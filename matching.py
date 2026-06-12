from typing import Self

import numpy as np
import numpy.typing as npt
import xarray as xr
import skrf as rf

# Produce a L matching network for the given impedance (over frequency)
# The load must be passive, i.e., |load_gamma| < 1
def match_l_network(freq: npt.NDArray[np.float64], f0: float, load_gamma: complex, use_2nd_sol=False) -> rf.Network:
    z0 = 50
    zl = z0 * (1 + load_gamma) / (1 - load_gamma)
    rl = np.real(zl)
    xl = np.imag(zl)

    # matching network
    f = rf.Frequency.from_f(freq)
    line = rf.media.DefinedGammaZ0(frequency=f, z0=z0)
    
    el1 = None
    el2 = None

    if rl > z0:
        b = (xl + np.sqrt(rl/z0)*np.sqrt(rl**2 + xl**2 - z0*rl))/(rl**2 + xl**2)
        if use_2nd_sol:
            b = (xl - np.sqrt(rl/z0)*np.sqrt(rl**2 + xl**2 - z0*rl))/(rl**2 + xl**2)
        
        x = 1/b + (xl*z0)/rl - z0/(b*rl)

        # find capacitors and inductors
        if x > 0:
            el1 = line.inductor(L = x / (2*np.pi*f0))
        else:
            el1 = line.capacitor(C = -1 / (x * 2*np.pi*f0))
        
        if b > 0:
            el2 = line.shunt_capacitor(C = b / (2*np.pi*f0))
        else:
            el2 = line.shunt_inductor(L = -1 / (b * 2*np.pi*f0))

    if rl <= z0:
        x = np.sqrt(rl*(z0-rl)) - xl
        b = np.sqrt((z0-rl)/rl) / z0

        if use_2nd_sol:
            x = -np.sqrt(rl*(z0-rl)) - xl
            b = -np.sqrt((z0-rl)/rl) / z0
        
        # find capacitors and inductors
        if b > 0:
            el1 = line.shunt_capacitor(C = b / (2*np.pi*f0))
        else:
            el1 = line.shunt_inductor(L = -1 / (b * 2*np.pi*f0))

        if x > 0:
            el2 = line.inductor(L = x / (2*np.pi*f0))
        else:
            el2 = line.capacitor(C = -1 / (x * 2*np.pi*f0))
    
    net = el1 ** el2

    return net


