#! /usr/bin/env python3
import numpy as np
import numpy.typing as npt
import scipy as sp
import matplotlib.pyplot as plt
import xarray as xr
from antlib.constants import C0

# Convert cylindrical near field measurements to far fields
# Eth and Eph are near field values, with dims "freq", "ph" and "z"
# n is order of azimuthal mode to expand in
# ff_th and ff_ph are desired sampling points in far field
def cylindrical_nf_to_ff(r0: float, Eth: xr.DataArray, Eph: xr.DataArray, ff_th: npt.NDArray[np.float64], ff_ph: npt.NDArray[np.float64], N: int = 20) -> tuple[xr.DataArray, xr.DataArray]:
    freq = Eth.coords["freq"].values
    k0 = 2*np.pi*freq / C0
    # order of azimuthal mode
    n = np.arange(-N, N+1, 1)
    # elevations modes
    h = np.multiply.outer(k0, np.cos(ff_th))
    # produce grid of modes to evaluate
    # NN and HH are index [freq, h, n]
    NN = np.apply_along_axis(lambda hi: np.meshgrid(n, hi)[0], axis=1, arr=h)
    HH = np.apply_along_axis(lambda hi: np.meshgrid(n, hi)[1], axis=1, arr=h)

    print(HH[0,:,0])

sample_E =  xr.DataArray(np.array([[[1,1,1,1],[1,1,1,1],[1,1,1,1]], [[1,1,1,1],[1,1,1,1],[1,1,1,1]]]), dims=("freq", "th", "ph"), coords={"freq": np.array([1e9,2e9]), "th":np.array([0,1,2]), "ph": np.array([1,2,3,4])})
cylindrical_nf_to_ff(1, sample_E, sample_E, np.linspace(0, np.pi, 21), np.linspace(0, 2*np.pi, 21))