#! /usr/bin/env python3
import numpy as np
import numpy.typing as npt
import scipy as sp
from scipy.special import hankel2
import matplotlib.pyplot as plt
import xarray as xr
from antlib.constants import C0
from antlib.pattern import Pattern

# Convert cylindrical near field measurements to far fields
# Ez and Eph are near field values, with dims "freq", "ph" and "z"
# n is order of azimuthal mode to expand in
# ff_th and ff_ph are desired sampling points in far field
def cylindrical_nf_to_ff(r0: float, Ez: xr.DataArray, Eph: xr.DataArray, ff_th: npt.NDArray[np.float64], ff_ph: npt.NDArray[np.float64], N: int = 20) -> Pattern:
    freq = Ez.coords["freq"].values
    k0 = 2*np.pi*freq / C0
    k0xr = 2*np.pi*Ez.coords["freq"] / C0
    # order of azimuthal mode
    n = np.arange(-N, N+1, 1)
    # elevations modes
    h = np.multiply.outer(k0, np.cos(ff_th))
    # produce grid of modes to evaluate
    # NN and HH are indexed [freq, h, n]
    NN = np.apply_along_axis(lambda hi: np.meshgrid(n, hi)[0], axis=1, arr=h)
    HH = np.apply_along_axis(lambda hi: np.meshgrid(n, hi)[1], axis=1, arr=h)

    NNxr = xr.DataArray(NN, dims=("freq", "th", "n"), coords={"freq": Ez.coords["freq"].values, "th": ff_th, "n": n})
    HHxr = xr.DataArray(HH, dims=("freq", "th", "n"), coords={"freq": Ez.coords["freq"].values, "th": ff_th, "n": n})

    # compute Iphi and Iz
    Iphi = 1/(4*np.pi**2) * (Eph * 
                             np.exp(-1j*NNxr*Eph.coords["ph"]) * 
                             np.exp(-1j*HHxr*Eph.coords["z"])).integrate(coord="ph").integrate(coord="z")
    Iz = 1/(4*np.pi**2) * (Ez * 
                           np.exp(-1j*NNxr*Ez.coords["ph"]) * 
                           np.exp(-1j*HHxr*Ez.coords["z"])).integrate(coord="ph").integrate(coord="z")

    # compute modal coefficients
    Lam = np.sqrt(k0xr**2 - HHxr**2)
    dHdr = Lam * 0.5 * (hankel2(NNxr-1, Lam*r0) - hankel2(NNxr+1, Lam*r0))

    an = (-Iphi / dHdr) + (NNxr*HHxr*Iz / (Lam**2*r0*dHdr)) 
    bn = Iz / (Lam**2 / k0xr * hankel2(NNxr, Lam*r0))

    # compute far field samping grid
    thg, phg = np.meshgrid(ff_th, ff_ph, indexing="ij")

    thxr = xr.DataArray(thg, dims=("th", "ph"), coords={"th": ff_th, "ph": ff_ph})
    phxr = xr.DataArray(phg, dims=("th", "ph"), coords={"th": ff_th, "ph": ff_ph})

    # compute far fields through fft
    Eth_ff = -1j*2*np.sin(thxr) / r0 * np.exp(-1j*k0xr*r0) * ((1j**NNxr)*bn*np.exp(1j*NNxr*phxr)).sum(dim="n")
    Eph_ff = -2*np.sin(thxr) / r0 * np.exp(-1j*k0xr*r0) * ((1j**NNxr)*an*np.exp(1j*NNxr*phxr)).sum(dim="n")

    return Pattern.from_dataarrays(Eth_ff, Eph_ff)

#sample_E =  xr.DataArray(np.array([[[1,1,1,1],[1,1,1,1],[1,1,1,1]], [[1,1,1,1],[1,1,1,1],[1,1,1,1]]]), dims=("freq", "ph", "z"), coords={"freq": np.array([1e9,2e9]), "ph":np.array([0,1,2]), "z": np.array([1,2,3,4])})
#cylindrical_nf_to_ff(1e-4, sample_E, sample_E, np.linspace(0, np.pi, 21), np.linspace(0, 2*np.pi, 21))