# Tries to load an HDF file, convert the prims to cons, and go back

import sys
from scipy.linalg import norm

from pyHARM.defs import Loci

from pyHARM.grmhd.u_to_p import U_to_P
from pyHARM.grmhd.phys import prim_to_flux, get_state
from pyHARM.io.dump import read_dump
from pyHARM.grid import Grid

P, params = read_dump(sys.argv[1])

# Set anything else we need to specify to get a Grid
params['ng'] = 0
params['debug'] = True
G = Grid(params)

D = get_state(params, G, P, Loci.CENT)
U = prim_to_flux(params, G, P, D, 0, Loci.CENT)

# Set everything else we need for U_to_P
params['invert_err_tol'] = 1e-8
params['invert_iter_max'] = 8
params['invert_iter_delta'] = 1e-5
params['gamma_max'] = 50
P_test, _ = U_to_P(params, G, U, 1.05*P)

print("Norm difference: ", norm((P - P_test).get()))
