# Coordination file for advancing fluid by one timestep

import numpy as np
from scipy import linalg as la

import pyopencl as cl
import pyopencl.array as cl_array
import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

from pyHARM.loopy_tools import tune_prims_kernel, add_ghosts, primsArrayArgs

from pyHARM.defs import Loci
from pyHARM.bounds import set_bounds  # , fix_flux
from pyHARM.fixup import fixup, fixup_utoprim
from pyHARM.fluxes import get_flux, flux_ct
from pyHARM.phys import get_state, prim_to_flux, get_fluid_source

from pyHARM.u_to_p import U_to_P
from pyHARM.iharmc.iharmc import Iharmc

from debug_tools import plot_prims, plot_var

knl_finite_diff = None

P_d = None
def step(params, G, P, dt, prep_dump=False):
    # Accommodate either an ndarray or cl_array for later
    # Cache a device prims array to update with ndarray contents
    global P_d
    if P_d is None:
        if isinstance(P, np.ndarray):
            P_d = cl_array.to_device(params['queue'], P)

    if isinstance(P, np.ndarray):
        P_d.set(P)
        P = P_d

    # Need both P_n and P_n+1 to calculate current
    if prep_dump:
        Psave = P.copy()

    # Predictor setup
    Ptmp, pflag, _ = advance_fluid(params, G, P, P, 0.5 * dt)

    if params['electrons']:
        pass
        #heat_electrons(G, P, Ptmp)

    # Ptmp_h = Ptmp.get()
    # pflag_h = pflag.get()
    # set_bounds(params, G, Ptmp_h, pflag=pflag_h)
    # Ptmp.set(Ptmp_h)
    # pflag.set(pflag_h)

    # Returning allows callee to change backing storage
    # TODO either always or (preferably) never use this convention...
    Ptmp = fixup_utoprim(params, pflag, G, Ptmp)
    Ptmp = fixup(params, G, Ptmp)

    Ptmp_h = Ptmp.get()
    pflag_h = pflag.get()
    set_bounds(params, G, Ptmp_h, pflag=pflag_h)
    Ptmp.set(Ptmp_h)
    pflag.set(pflag_h)

    # Corrector step
    P, pflag, new_dt = advance_fluid(params, G, P, Ptmp, dt)

    if params['electrons']:
        pass
        #heat_electrons(G, Ptmp, P)

    # P_h = Ptmp.get()
    # pflag_h = pflag.get()
    # set_bounds(params, G, P_h, pflag=pflag_h)
    # P.set(P_h)
    # pflag.set(pflag_h)

    P = fixup_utoprim(params, pflag, G, P)
    P = fixup(params, G, P)

    # Final stage and in-between is host-side
    P_h = P.get()
    set_bounds(params, G, P_h)
    P.set(P_h)

    # If we're dumping this step, update the current
    if prep_dump:
        #current_calc(G, P, Psave, dt)  # TODO Currents output
        pass

    # Limit next timestep.  Also see calculation in ndt_min
    if new_dt > params['safe_step_increase'] * dt:
        new_dt = params['safe_step_increase'] * dt

    # Returns a cl_array
    return P, new_dt


Ui = None
dU = None
Uf = None
Ds = None
ihc = None
def advance_fluid(params, G, Pi, Ps, dt):
    s = G.slices
    sh = G.shapes
    global Ui, dU, Uf, Ds
    if Ui is None:
        Ui = cl_array.zeros(params['queue'], sh.grid_primitives, dtype=np.float64)
        dU = cl_array.zeros(params['queue'], sh.grid_primitives, dtype=np.float64)
        Uf = cl_array.zeros(params['queue'], sh.grid_primitives, dtype=np.float64)
        Ds = get_state(params, G, Ps, Loci.CENT)

    global ihc
    if ihc is None and 'use_ctypes' in params and params['use_ctypes']:
        ihc = Iharmc()

    F, ndt = get_flux(params, G, Ps)

    # plot_all_prims(G, F[1], "F1")
    # plot_all_prims(G, F[2], "F2")
    # plot_all_prims(G, F[3], "F3")

    # TODO RESTORE following
    # if params['metric'][-3:] == "mks":
    #     fix_flux(F)

    # Constrained transport for B
    flux_ct(params, G, F)

    # Flux diagnostic globals
    # diag_flux(F)

    # Get conserved variables & source term
    get_state(params, G, Ps, Loci.CENT, out=Ds)
    get_fluid_source(params, G, Ps, Ds, out=dU)

    if Pi is not Ps:
        Di = get_state(params, G, Pi, Loci.CENT)
    else:
        Di = Ds

    prim_to_flux(params, G, Pi, Di, 0, Loci.CENT, out=Ui)
    if 'P_U_P_test' in params and params['P_U_P_test']:
        # Test U_to_P by recovering a prims array we know (Ps),
        # starting from a close but not identical array (Pi)
        if Pi is not Ps:
            Us = prim_to_flux(params, G, Ps, Ds, 0, Loci.CENT)
            Ps_fromU, _ = ihc.U_to_P(params, G, Us, Pi)
            print("Pi vs Ps change: ", la.norm((Ps - Pi).get()[s.bulk]))
            print("Ps->U->Ps Absolute Error: ", la.norm((Ps - Ps_fromU).get()[s.bulk]))
        else:
            print("Pi is Ps")

    global knl_finite_diff
    if knl_finite_diff is None:
        code = add_ghosts("""
        Uf[p,i,j,k] = Ui[p,i,j,k] + \
                      dt * ((F1[p,i,j,k] - F1[p,i+1,j,k]) / dx1 + \
                            (F2[p,i,j,k] - F2[p,i,j+1,k]) / dx2 + \
                            (F3[p,i,j,k] - F3[p,i,j,k+1]) / dx3 + dU[p,i,j,k])
        """)
        knl_finite_diff = lp.make_kernel(sh.isl_grid_primitives, code,
                                         [lp.ValueArg("dt", dtype=np.float64),
                                          *primsArrayArgs("Ui", "F1", "F2", "F3", "dU", "Uf"),
                                          ...],
                                         assumptions=sh.assume_grid)
        knl_finite_diff = lp.fix_parameters(knl_finite_diff, dx1=G.dx[1], dx2=G.dx[2], dx3=G.dx[3])
        knl_finite_diff = tune_prims_kernel(knl_finite_diff, sh.bulk_primitives, ng=G.NG)
        print("Compiled finite_diff")

    if isinstance(dt, cl_array.Array):
        dt = dt.get()
    evt, _ = knl_finite_diff(params['queue'], dt=float(dt), Ui=Ui, F1=F[1], F2=F[2], F3=F[3], dU=dU, Uf=Uf)

    # Newton-Raphson
    if ihc is not None:
        # If we're using iharm3d's C U_to_P fn, negotiate memory.  Note eflag is not preserved!
        Pf, pflag = ihc.U_to_P(params, G, Uf.get(), Pi.get())
        Pf = cl_array.to_device(params['queue'], Pf)
        pflag = cl_array.to_device(params['queue'], pflag)
    else:
        # Loopy version
        Pf, pflag = U_to_P(params, G, Uf, Pi)

    if params['debug']:
        print("Uf - Ui: ", la.norm((Uf - Ui).get()))
        print("Pf - Pi: ", la.norm((Pf - Pi).get()))

    return Pf, pflag, ndt
