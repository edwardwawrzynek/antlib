from typing import Self

import numpy as np
import numpy.typing as npt
import xarray as xr
import skrf as rf

from antlib.read_pats import read_nsi_measurements, read_feko_ffe, read_hfss_ffd, get_nsi_nth_nph_nfreqs, write_feko_ffe
from antlib.constants import ETA0

# Pattern represents a radiation pattern, sampled on a sphere
class Pattern:
    # Construct a radiation pattern from sampled points gridded in theta-phi
    # Eth and Eph are ndarray's of complex electric field intensity values (in V), indexed as [freq, theta, phi]
    # freq is a list of the frequencies of measurement in Hz
    # theta and phi are the angles in radians of measured points
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

    # return a pattern from Eth and Eph dataarray
    @classmethod
    def from_dataarrays(cls, Eth, Eph, rad_eff=None) -> Self:
        return cls(Eth.values, Eph.values, Eth.coords["freq"], Eth.coords["th"], Eth.coords["ph"], rad_eff)

    # compute the radiant intensity of the pattern (in W/rad^2)
    def radiant_intensity(self) -> xr.DataArray:
        # compute total radiated power intensity
        Esq = np.abs(self.Eth**2) + np.abs(self.Eph ** 2)
        U = Esq / ETA0

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
    def rotate_phi(self, delta_phi: float) -> Self:
        iphi = round(delta_phi / (self.Eth.coords["ph"][1] - self.Eth.coords["ph"][0]))
        Ethr = self.Eth.roll(ph=iphi, roll_coords=False)
        Ephr = self.Eph.roll(ph=iphi, roll_coords=False)

        return Self.from_dataarrays(Ethr, Ephr, self.rad_eff.data)

    # return a pattern with just a single frequency point
    # returns the closest data to the specified frequency
    def select_freq(self, freq: float) -> Self:
        return Pattern.from_dataarrays(self.Eth.sel(freq=freq, method="nearest").expand_dims("freq"), self.Eph.sel(freq=freq, method="nearest").expand_dims("freq"), self.rad_eff.sel(freq=freq, method="nearest").expand_dims("freq"))

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

# A collection of patterns sampled at the same points, which can be excited togeather
# This would typically be used to represent a collection of embedded element patterns of an array
class PatternArray:
    def __init__(self, patterns: list[Pattern]):
        self.patterns = patterns
    
    def excite(self, weights: npt.NDArray[np.complex128]) -> Pattern:
        Eth = xr.zeros_like(self.patterns[0].Eth)
        Eph = xr.zeros_like(self.patterns[0].Eph)

        for n in range(len(weights)):
            Eth += weights[n] * self.patterns[n].Eth
            Eph += weights[n] * self.patterns[n].Eph

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
    
    # Compute active reflection coefficients for a given excitation
    def active_gamma(self, a: npt.NDArray[np.complex128]) -> xr.DataArray:
        b = self.sp.s @ a.T
        active_gamma = b / a
        return xr.DataArray(active_gamma, dims=["freq", "elements"], coords={"freq": self.sp.f, "element_num": np.arange(0, np.size(a), 1)})

    # compute the total active reflection coefficient for a given excitation
    def tarc(self, a: npt.NDArray[np.complex128]) -> xr.DataArray:
        b = self.sp.s @ a.T
        tarc = np.sqrt(np.sum(np.abs(b)**2, axis=1) / np.sum(np.abs(a)**2))
        return xr.DataArray(tarc, dims=["freq"], coords={"freq": self.sp.f})

    # excite the array, returning active pattern and radiation efficiency
    def excited_pat_rad_eff(self, a: npt.NDArray[np.complex128]) -> tuple[Pattern, xr.DataArray]:
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

        return pat, rad_eff

    # Return the number of elements
    def N(self) -> int:
        return len(self.patterns)

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
        
        return AntennaArray(PatternArray(patterns), rf.Network(sp_path))

    # construct an array from s-params and patterns stored in an hdf5 file
    @classmethod
    def from_touchstone_hdf5(cls, sp_path: str, patterns_path: str) -> Self:
        return AntennaArray(PatternArray.from_hdf5(patterns_path), rf.Network(sp_path))

class CircularAntennaArray(AntennaArray):
    # Convert an AntennaArray into a CircularAntennaArray
    # The elements should be laid out on a circle in a counterclockwise orientation, with element 0 at phi=0
    @classmethod
    def from_antenna_array(cls, array: AntennaArray) -> CircularAntennaArray:
        return CircularAntennaArray(array.patterns, array.sp)

    # Return the unaliased circular mode indices available in this array
    def mode_indices(self) -> list[int]:
        return np.linspace(0, self.N(), 1) - int(self.N() / 2)

    
