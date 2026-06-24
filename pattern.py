from __future__ import annotations
from typing import Self

import numpy as np
import numpy.typing as npt
import xarray as xr
import skrf as rf

from antlib.read_pats import read_nsi_measurements, read_feko_ffe, read_hfss_ffd, get_nsi_nth_nph_nfreqs, write_feko_ffe
from antlib.matching import match_l_network
from antlib.constants import ETA0

# Pattern represents a radiation pattern, sampled on a sphere
class Pattern:
    # Construct a radiation pattern from sampled points gridded in theta-phi
    # Eth and Eph are ndarray's of peak (NOT rms) complex electric field intensity values (in V), indexed as [freq, theta, phi]
    # freq is a list of the frequencies of measurement in Hz
    # theta and phi are the angles in radians of measured points, which should be [0,pi] and [0,2*pi]
    # rad_eff is radiation efficiency, used for calculating gain
    def __init__(self, Eth: npt.NDArray[np.complex128], Eph:  npt.NDArray[np.complex128], freq: npt.NDArray[np.float64], theta: npt.NDArray[np.float64], phi: npt.NDArray[np.float64], rad_eff: npt.NDArray[np.float64] = None):
        # remove duplicate points at beginning / end of data
        if np.isclose(theta[-1], theta[0] + np.pi):
            theta = theta[0:-1]
            Eth = Eth[:,0:-1,:]
            Eph = Eph[:,0:-1,:]
        if np.isclose(phi[-1], phi[0] + 2*np.pi):
            phi = phi[0:-1]
            Eth = Eth[:,:,0:-1]
            Eph = Eph[:,:,0:-1]

        self.Eth = xr.DataArray(Eth, dims=("freq", "th", "ph"), coords={"freq": freq, "th": theta, "ph": phi})
        self.Eph = xr.DataArray(Eph, dims=("freq", "th", "ph"), coords={"freq": freq, "th": theta, "ph": phi})
        # if no radiation efficiency, set to 1
        if rad_eff is None:
            rad_eff = np.ones(np.size(freq))

        self.rad_eff = xr.DataArray(rad_eff, dims=("freq"), coords={"freq": freq})

    # Round frequency to the nearest Hz
    # This might be needed to match sparameters, which are often rounded to nearest Hz
    def round_freq_to_Hz(self):
        self.Eth.coords["freq"] = self.Eth.coords["freq"].round()
        self.Eph.coords["freq"] = self.Eph.coords["freq"].round()
        self.rad_eff.coords["freq"] = self.rad_eff.coords["freq"].round()

    # return a pattern from Eth and Eph dataarray
    @classmethod
    def from_dataarrays(cls, Eth, Eph, rad_eff=None) -> Self:
        return cls(Eth.values, Eph.values, Eth.coords["freq"], Eth.coords["th"], Eth.coords["ph"], rad_eff)

    # compute the radiant intensity of the pattern (in W/rad^2)
    def radiant_intensity(self) -> xr.DataArray:
        # compute total radiated power intensity
        Esq = np.abs(self.Eth**2) + np.abs(self.Eph ** 2)
        U = Esq / (2.0 * ETA0)

        return U

    # compute the radiated power contained in the pattern (in W)
    def power(self) -> xr.DataArray:
        U = self.radiant_intensity()
        # integrate radiant intensity to get total power
        Ptot = (U * np.abs(np.sin(U.coords["th"]))).integrate(coord="ph").integrate(coord="th")
        return Ptot

    # compute directivity (linear scale, 10*log10 for dB)
    def directivity(self) -> xr.DataArray:
        return self.radiant_intensity() * 4 * np.pi / self.power()
    
    # compute gain (linear scale, 10*log10 for dB)
    def gain(self) -> xr.DataArray:
        return self.rad_eff * self.directivity()
    
    # return the pattern rotated in phi by the specified amount (in radians)
    # the rotation will be rounded to the nearest sampling point
    def rotate_phi(self, delta_phi: float) -> Pattern:
        iphi = round(delta_phi / (self.Eth.coords["ph"][1].values - self.Eth.coords["ph"][0].values))
        Ethr = self.Eth.roll(ph=iphi, roll_coords=False)
        Ephr = self.Eph.roll(ph=iphi, roll_coords=False)

        return Pattern.from_dataarrays(Ethr, Ephr, self.rad_eff.data)

    # return a pattern with just a single frequency point
    # returns the closest data to the specified frequency
    def select_freq(self, freq: float) -> Self:
        return Pattern.from_dataarrays(self.Eth.sel(freq=freq, method="nearest").expand_dims("freq"), self.Eph.sel(freq=freq, method="nearest").expand_dims("freq"), self.rad_eff.sel(freq=freq, method="nearest").expand_dims("freq"))

    # normalize pattern intensity and compute radiation efficiency based on gain cal
    def normalize_gain_cal(self, std: GainStandardPattern):
        fac = std.power_normalize_factor()
        self.Eth *= fac
        self.Eph *= fac

    # take a theta cut of the given data
    @classmethod
    def theta_cut(cls, data: xr.DataArray, th: float) -> xr.DataArray:
        return data.sel(th=th, method="nearest")
    
    # take a phi cut of the given data
    @classmethod
    def phi_cut(cls, data: xr.DataArray, ph: float) -> xr.DataArray:
        # normalize ph to data range
        ph = ph % (2*np.pi)
        # select both ph and opposite ph cuts
        cut1 = data.sel(ph=ph, method="nearest")
        cut2 = data.sel(ph=(ph+np.pi)%(2*np.pi), method="nearest")
        # reverse and relabel cut2 coordinates
        cut2 = cut2.isel(th=slice(None, None, -1))
        cut2.coords["th"] = 2*np.pi - cut2.coords["th"]
        cut2.coords["ph"] = cut1.coords["ph"]
        return xr.concat([cut1, cut2], dim="th", coords="different", compat="equals")

    # take an fft of the given cut data and return the fourier coefficients and mode indices
    @classmethod
    def fft_cut(cls, data: xr.DataArray, coord) -> xr.DataArray:
        coord_axis = data.get_axis_num(coord)
        fcomp = np.fft.fftshift(np.fft.fft(data.data, axis=coord_axis), axes=coord_axis)
        p = CircularAntennaArray.mode_indices_N(np.size(data.coords[coord]))
        return xr.DataArray(fcomp, dims=("freq", "p"), coords={"freq": data.coords["freq"], "p": p})
    
    # return the least squares phase center in kx for the given data (probably E-field data)
    @classmethod
    def phase_center_kx(cls, data: xr.DataArray) -> xr.DataArray:
        phase = np.unwrap(xr.ufuncs.angle(data))
        phase = phase - np.mean(phase)
        weight = np.cos(data.coords["ph"]).expand_dims("dummy_num_dim").T

        kx, _, _, _ =  np.linalg.lstsq(weight.data, phase.T.data)

        return xr.DataArray(kx[0,:], dims=("freq"), coords={"freq": data.coords["freq"]})

    # return an z-oriented Hertzian electric dipole pattern containing 1 W at the specified spacing
    @classmethod
    def hertzian_electric_dipole(cls, freq: npt.NDArray[np.float64]=np.array([1e9, 2e9, 3e9]), th_space: float=2*np.pi/180, ph_space: float=2*np.pi/180) -> Self:
        theta = np.arange(0, np.pi, th_space)
        phi = np.arange(0, 2*np.pi, ph_space)

        ph, th = np.meshgrid(phi, theta)

        Eph = np.zeros((np.size(freq), np.size(theta), np.size(phi)), dtype=np.complex128)
        Eth = np.zeros((np.size(freq), np.size(theta), np.size(phi)), dtype=np.complex128)
        
        Eth[:,:,:] = np.sqrt(ETA0 * 3/(8*np.pi)) * np.sin(th)

        return Pattern(Eth, Eph, freq, theta, phi)

    # Return the pattern contained in a NSI txt file
    @classmethod
    def from_NSI_txt(cls, path: str) -> Self:
        # read data
        nth, nph, nfreq = get_nsi_nth_nph_nfreqs(path)
        freqs, th, ph, Eth, Eph = read_nsi_measurements(path, nth, nph, nfreq)
        # TODO: normalize with gain calibration
        return Pattern(Eth, Eph, freqs, np.unique(th), np.unique(ph), rad_eff=None)

    # Return the pattern contained in a FFE file from Altair Feko
    @classmethod
    def from_FEKO_FFE(cls, path: str) -> Self:
        # read data
        freqs, th, ph, Eth, Eph = read_feko_ffe(path)
        return Pattern(Eth, Eph, freqs, np.unique(th), np.unique(ph), rad_eff=None)
    
    def write_FEKO_FFE(self, path: str):
        write_feko_ffe(path, self.Eth.coords["freq"].data, self.Eth.coords["th"].data, self.Eth.coords["ph"].data, self.Eth.data, self.Eph.data)

    # Return the pattern contained in a FFD file from Ansys HFSS
    @classmethod
    def from_HFSS_FFD(cls, path: str) -> Self:
        # read data
        freqs, th, ph, Eth, Eph = read_hfss_ffd(path)
        return Pattern(Eth, Eph, freqs, np.unique(th), np.unique(ph), rad_eff=None)

# A pattern measurement of a gain standard antenna
# ie, a pattern and a gain table
class GainStandardPattern(Pattern):
    def __init__(self, Eth: npt.NDArray[np.complex128], Eph:  npt.NDArray[np.complex128], freq: npt.NDArray[np.float64], theta: npt.NDArray[np.float64], phi: npt.NDArray[np.float64], realized_gain_f: npt.NDArray[np.float64], realized_gain: npt.NDArray[np.float64]):
        super().__init__(Eth, Eph, freq, theta, phi, rad_eff=None)
        # interpolate gain standard onto frequency range
        # TODO: consider if this is the best way to do the interpolation
        gain_int = np.interp(freq, realized_gain_f, realized_gain)
        self.gain_standard = xr.DataArray(gain_int, dims="freq", coords={"freq": freq})
    
    # compute the field normalization factor (as a function of frequency) for 1 W incident
    def power_normalize_factor(self):
        pwr_norm = self.directivity().max(dim=("th", "ph")) * self.power() / self.gain_standard
        return np.sqrt(1/pwr_norm)
        
    @classmethod
    def from_pattern_with_standard_txt(cls, pat: Pattern, standard_txt_path: str, freq_unit=1e9, gain_dBi=True) -> Self:
        data = np.loadtxt(standard_txt_path)
        rgfreq = data[:,0] * freq_unit
        rg_gain = data[:,1]
        if gain_dBi:
            rg_gain = 10.0**(rg_gain/10.0)
        
        return GainStandardPattern(pat.Eth.values, pat.Eph.values, pat.Eth.coords["freq"], pat.Eth.coords["th"], pat.Eth.coords["ph"], rgfreq, rg_gain)


# A collection of patterns sampled at the same points, which can be excited togeather
# This would typically be used to represent a collection of embedded element patterns of an array
class PatternArray:
    def __init__(self, patterns: list[Pattern]):
        self.patterns = patterns
    
    def excite(self, weights: npt.NDArray[np.complex128]) -> Pattern:
        Eth = xr.zeros_like(self.patterns[0].Eth)
        Eph = xr.zeros_like(self.patterns[0].Eph)

        for n in range(len(weights)):
            Eth += (weights[n] * self.patterns[n].Eth.T).T
            Eph += (weights[n] * self.patterns[n].Eph.T).T

        return Pattern.from_dataarrays(Eth, Eph)

    # save patterns to an hdf5 file on disk
    def to_hdf5(self, path: str):
        entries = {}
        for n in range(len(self.patterns)):
            entries[f"element_{n}_Eth"] = self.patterns[n].Eth
            entries[f"element_{n}_Eph"] = self.patterns[n].Eph
            entries[f"element_{n}_rad_eff"] = self.patterns[n].rad_eff
        ds = xr.Dataset(entries)
        ds.to_netcdf(path, engine="h5netcdf")

    # read patterns from an hdf5 file on disk
    # the file should have element_N_Eth, element_N_Eph, and element_N_rad_eff entries, where N is the zero-indexed element number
    @classmethod
    def from_hdf5(cls, path: str) -> Self:
        ds = xr.open_dataset(path, engine="h5netcdf")
        patterns = []
        n = 0
        while True:
            if not f"element_{n}_Eth" in ds:
                break

            patterns.append(Pattern.from_dataarrays(ds.get(f"element_{n}_Eth"), ds.get(f"element_{n}_Eph"), ds.get(f"element_{n}_rad_eff")))
            n += 1

        return PatternArray(patterns)

# An antenna array, containing patterns and scattering parameters
class AntennaArray:
    # Initialize array from element patterns and scattering parameter matrix
    # Element patterns should be scaled for 1W incident on the port
    def __init__(self, patterns: PatternArray, sp: rf.Network):
        self.patterns = patterns
        self.sp = sp
    
    # Return the number of elements
    def N(self) -> int:
        return len(self.patterns.patterns)
    
    # Compute active reflection coefficients for a given excitation
    def active_gamma(self, a: npt.NDArray[np.complex128]) -> xr.DataArray:
        b = self.sp.s @ a.T
        active_gamma = b / a
        return xr.DataArray(active_gamma, dims=["freq", "element"], coords={"freq": self.sp.f, "element": np.arange(0, np.size(a), 1)})

    # compute the total active reflection coefficient for a given excitation
    # result is in linear scale, use 20*log10 for dB
    def tarc(self, a: npt.NDArray[np.complex128]) -> xr.DataArray:
        b = self.sp.s @ a.T
        tarc = np.sqrt(np.sum(np.abs(b)**2, axis=1) / np.sum(np.abs(a)**2))
        return xr.DataArray(tarc, dims=["freq"], coords={"freq": self.sp.f})

    def tarc_freq_dependent(self, a: xr.DataArray) -> xr.DataArray:
        res = np.zeros(np.size(self.sp.f), dtype=np.complex128)

        # add dummy dimension to a and order properly
        a = a.expand_dims("dummy")
        a = a.transpose("freq", "element", "dummy")

        b = (self.sp.s @ a.values).squeeze(-1)
        tarc = np.sqrt(np.sum(np.abs(b)**2, axis=1) / np.sum(np.abs(a.values.squeeze(-1))**2, axis=1))

        return xr.DataArray(tarc, dims=["freq"], coords={"freq": self.sp.f})

    # excite the array, returning active pattern (with computed radiation efficiency)
    def excite(self, a: npt.NDArray[np.complex128]) -> Pattern:
        # power incident on ports
        Pinc = np.sum(np.abs(a)**2)
        # normalize excitations for incident power to be 1 W
        a = a / np.sqrt(Pinc)
        # power transmitted into antenna
        Ptransmit = (1 - self.tarc(a)**2)
        # excite patterns
        pat = self.patterns.excite(a)
        # total power radiated
        Prad = pat.power()
        # radiation efficiency
        rad_eff = Prad / Ptransmit
        pat.rad_eff = rad_eff

        return pat
    
    # excite the array with a different excitation a at each frequency
    # a should have dimensions element and freq
    def excite_freq_dependent(self, a: xr.DataArray) -> Pattern:
        # power incident on ports
        Pinc = (np.abs(a)**2).sum(dim="element")
        # normalize excitations for incident power to be 1 W
        a = a / np.sqrt(Pinc)
        # power transmitted into antenna
        tarc = self.tarc_freq_dependent(a)
        Ptransmit = (1 - tarc**2)
        # excite patterns
        pat = self.patterns.excite(a.transpose("element", "freq"))
        # total power radiated
        Prad = pat.power()
        # radiation efficiency
        rad_eff = Prad / Ptransmit
        pat.rad_eff = rad_eff

        return pat

    # generate a network of resistors which makes each active impedance passive
    # i.e., add enough loss to any element with negative active resistance
    # returns the network and active impedances with resistors added
    def make_lossy_negative_active_r_compensation(self, a: npt.NDArray[np.complex128], f0: float, z0: float=50) -> tuple[rf.Network, npt.NDArray[np.complex128]]:
        netwrks = []
        agamma = self.active_gamma(a).sel(freq=f0, method="nearest")
        netwrks = []
        f = rf.Frequency.from_f(self.sp.f)
        line = rf.media.DefinedGammaZ0(frequency=f, z0=z0)

        act_z = np.zeros(self.N(), dtype=np.complex128)

        for n in range(self.N()):
            gam = agamma.sel(element=n).item()
            z = z0 * (1 + gam) / (1 - gam)
            if np.real(z) >= 0:
                netwrks.append(line.resistor(R=0))
                act_z[n] = z
            else:
                Rcomp = -np.real(z)+1
                netwrks.append(line.resistor(R=Rcomp))
                act_z[n] = z + Rcomp
        
        return (rf.network.concat_ports(netwrks, port_order="second"), act_z)

    # generate a matching network for the array's active impedances at frequency f0
    # the first N port are inputs, the last N are connected to array inputs
    # return the matching network and compensated excitations to matching network
    def make_l_network_match_active_gamma(self, a: npt.NDArray[np.complex128], f0: float) -> tuple[rf.Network, npt.NDArray[np.complex128]]:
        f0_index = np.argmin(np.abs(self.sp.f - f0))
        # make impedances passive with lossy network
        net_lossy, act_z = self.make_lossy_negative_active_r_compensation(a, f0)
        # compute active impedances with loss added
        agamma = (act_z - 50) / (act_z + 50)
        # create matching networks
        netwrks = []
        for n in range(self.N()):
            gam = agamma[n]
            gam_match = match_l_network(self.sp.f, f0, gam)
            netwrks.append(gam_match)
        
        match_net = rf.network.concat_ports(netwrks, port_order="second")
        # add lossy section
        match_net = match_net ** net_lossy

        # Now we have to compensate the excitations for the effect of the matching network
        # Equations from S. Yen and D. Filipovic, "Theoretical Considerations for Tuning of HF Arrays of Electrically Small Monopoles", IEEE PAST 2024

        # extract s matrix at design frequency
        Sarray = self.sp.s[f0_index, :, :]
        # extract relevant blocks of s matrix
        Smm = match_net.s[f0_index, 0:self.N(), 0:self.N()]
        Snn = match_net.s[f0_index, self.N():, self.N():]
        Snm = match_net.s[f0_index, self.N():, 0:self.N()]
        Smn = match_net.s[f0_index, 0:self.N(), self.N():]

        I = np.eye(self.N())
        # compensated excitations for matching network
        acomp = np.linalg.solve(Snm, (I - Snn @ Sarray) @ a)
        # reflected waves from matching network
        bcomp = Smm @ acomp + (Smn @ Sarray @ np.linalg.solve(I - Snn @ Sarray, Snm @ acomp))

        return [match_net , acomp]
    
    def cascade_match(self, match_net: rf.Network) -> Self:
        Smm = match_net.s[:, 0:self.N(), 0:self.N()]
        Snn = match_net.s[:, self.N():, self.N():]
        Snm = match_net.s[:, self.N():, 0:self.N()]
        Smn = match_net.s[:, 0:self.N(), self.N():]
        Sarray = self.sp.s

        I = np.eye(self.N())

        new_pats = []
        for p in range(self.N()):
            Vin = np.zeros((self.N(),1), dtype=np.complex128)
            Vin[p] = 1
            a = np.linalg.solve((I - Snn @ Sarray), Snm @ Vin)
            # get a pattern with zero fields
            pat = self.patterns.excite(np.zeros(self.N()))
            for n in range(self.N()):
                pat.Eth += (a[:,n,0] * self.patterns.patterns[n].Eth.T).T
                pat.Eph += (a[:,n,0] * self.patterns.patterns[n].Eph.T).T
            
            new_pats.append(pat)

        return AntennaArray(PatternArray(new_pats), match_net.copy() ** self.sp.copy())

    # return the array with embedded element patterns if the array is terminated with a source network Sg
    def adjust_element_patterns_for_source(self, Sg: rf.Network) -> Self:
        assert(np.shape(Sg.s)[1] == self.N())
        assert(np.shape(Sg.s)[2] == self.N())

        I = np.eye(self.N())
        # convert element patterns to open circuited thevenin sources
        sa_to_oc_pat = np.linalg.inv(I - self.sp.s)
        sg_to_oc_pat = np.linalg.solve(np.swapaxes((I - self.sp.s), -1, -2), np.swapaxes((I - Sg.s @ self.sp.s),  -1, -2))

        to_sg_pat = np.linalg.solve(sg_to_oc_pat, sa_to_oc_pat)

        new_pats = []
        for p in range(self.N()):
            # get a pattern with zero fields
            pat = self.patterns.excite(np.zeros(self.N()))
            for n in range(self.N()):
                pat.Eth += (to_sg_pat[:,p,n] * self.patterns.patterns[n].Eth.T).T
                pat.Eph += (to_sg_pat[:,p,n] * self.patterns.patterns[n].Eph.T).T
            
            new_pats.append(pat)

        return AntennaArray(PatternArray(new_pats), self.sp.copy())




    # construct an array from s-params and pattern files (either NSI .txt, FEKO .ffe, or HFSS .ffd)
    # type is a list of "nsi", "feko", or "hfss", or None (type will be inferred from file extension)
    @classmethod
    def from_touchstone_pattern_files(cls, sp_path: str, pattern_path: list[str], types: list[str] = None) -> Self:
        # infer pattern file types
        if types is None:
            types = [""] * len(pattern_path)
            for n in range(len(pattern_path)):
                if pattern_path[n].endswith(".txt"):
                    types[n] = "nsi"
                elif pattern_path[n].endswith(".ffe"):
                    types[n] = "feko"
                elif pattern_path[n].endswith(".ffd"):
                    types[n] = "hfss"

        assert(len(pattern_path) == len(types))
        patterns = []
        for n in range(len(pattern_path)):
            print(f"Reading {pattern_path[n]}")
            if types[n] == "nsi":
                patterns.append(Pattern.from_NSI_txt(pattern_path[n]))
            elif types[n] == "feko":
                patterns.append(Pattern.from_FEKO_FFE(pattern_path[n]))
            elif types[n] == "hfss":
                patterns.append(Pattern.from_HFSS_FFD(pattern_path[n]))
            else:
                assert False, f"Cannot recognize extensions of {pattern_path[n]}"
        
        return AntennaArray(PatternArray(patterns), rf.Network(sp_path))

    # construct an array from s-params and patterns stored in an hdf5 file
    @classmethod
    def from_touchstone_hdf5(cls, sp_path: str, patterns_path: str) -> Self:
        return AntennaArray(PatternArray.from_hdf5(patterns_path), rf.Network(sp_path))

class CircularAntennaArray(AntennaArray):
    # Convert an AntennaArray into a CircularAntennaArray
    # The elements should be laid out on a circle in a counterclockwise orientation, with element 0 at phi=0
    @classmethod
    def from_antenna_array(cls, array: AntennaArray) -> Self:
        return CircularAntennaArray(array.patterns, array.sp)

    @staticmethod
    def mode_indices_N(N: float) -> list[int]:
        return np.linspace(0, N-1, N) - int(N / 2)

    # Return the unaliased circular mode indices available in this array
    def mode_indices(self) -> list[int]:
        return CircularAntennaArray.mode_indices_N(N=self.N())

    # Return the excitation vector for mode m
    def mode_m_excitation(self, m: int) -> npt.NDArray[np.complex128]:
        return np.exp(1j * 2*np.pi * np.linspace(0, self.N()-1, self.N()) * m / self.N())

    # Return the pattern for the array excited in mode m
    def mode_m(self, m: int) -> Pattern:
        return self.excite(self.mode_m_excitation(m))

    # return the excitation for exciting the modal spectrum modes,
    # with compensation for the array pattern given by fourier coefficients Dp (expanded about origin of array)
    def modal_excitation_pat_compensated(self, modes: npt.NDArray[np.complex128], Dp: xr.DataArray) -> npt.NDArray[np.complex128]:
        m = self.mode_indices()
        m = xr.DataArray(m, dims=("p"), coords={"p": m})
        # label modes to sum over pattern fourier coefficients
        modes = xr.DataArray(modes, dims=("p"), coords={"p": m})
        mode_excite = modes / Dp.sel(p=m) 

        n = xr.DataArray(np.linspace(0, self.N()-1, self.N()), dims=("element"), coords={"element": np.linspace(0, self.N()-1, self.N())})

        excite = (mode_excite * np.exp(1j*2*np.pi * n * m / self.N())).sum(dim="p")

        return excite
        


