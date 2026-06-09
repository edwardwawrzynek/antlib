import numpy as np

# Permittivity of free space (F/m)
EP0 = 8.85418782e-12
# Permeability of free space (A/m)
MU0 = 4*np.pi * 1e-7
# Impedance of free space (ohm)
ETA0 = np.sqrt(MU0 / EP0)
# Phase velocity of free space (m/s)
C0 = 1/np.sqrt(MU0*EP0)