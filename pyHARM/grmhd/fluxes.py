# FLUXES

import numpy as np

import pyopencl.array as cl_array
import pyopencl.clmath as clm

import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
from pyHARM.loopy_tools import *

from pyHARM.defs import Loci
from pyHARM.phys import get_state, prim_to_flux, mhd_vchar
from pyHARM.reconstruction import reconstruct

from pyHARM.debug_tools import plot_var, plot_prims

# If there's a better way to do statics than globals, I'm all ears
knl_flux_ct_emf = None; knl_flux_ct = None
knl_addfluxes = None; knl_ctop = None

emf1 = None; emf2 = None; emf3 = None
Pl = None; Pr = None
Ul = None; Ur = None
fluxL = None; fluxR = None

knl_ndt = None
ndt = None
def ndt_min(params, G, ctop):
    sh = G.shapes

    global knl_ndt
    if knl_ndt is None:
        # Note that ghost zones are already added where necessary!
        # ndt is kept bulk only as this is all that's needed/makes sense
        code = """
        ndt[i,j,k] = 1 / ( 1/(cour * dx[1] / ctop[1,i+ng,j+ng,k+ng]) +
                           1/(cour * dx[2] / ctop[2,i+ng,j+ng,k+ng]) +
                           1/(cour * dx[3] / ctop[3,i+ng,j+ng,k+ng]) )
        """
        knl_ndt = lp.make_kernel(sh.isl_grid_scalar, code,
                                 [*vecArrayArgs("ctop"), ...])
        knl_ndt = lp.fix_parameters(knl_ndt, cour=params['cour'])
        knl_ndt = lp.fix_parameters(knl_ndt, ndim=4)
        knl_ndt = tune_grid_kernel(knl_ndt, sh.bulk_scalar, ng=G.NG)
        print("Compiled ndt min")

    global ndt
    if ndt is None:
        ndt = cl_array.zeros(params['queue'], sh.bulk_scalar, dtype=np.float64)

    # TODO if debug print/record argmin?
    evt, _ = knl_ndt(params['queue'], ctop=ctop, dx=G.dx_d, ndt=ndt)

    # TODO manual reduce this?  Loopy doesn't like reductions...
    return cl_array.min(ndt)


ctop = None
def get_flux(params, G, P):
    sh = G.shapes

    # Just need 4 elements -- filled below
    F = [0] * 4

    global Pl, Pr, ctop
    if Pl is None:
        Pl = cl_array.empty(params['queue'], sh.grid_primitives, dtype=np.float64)
        Pr = cl_array.empty(params['queue'], sh.grid_primitives, dtype=np.float64)
        ctop = cl_array.empty(params['queue'], sh.grid_vector, dtype=np.float64)

    # reconstruct left- and right-going components
    reconstruct(params, G, P, 1, lout=Pl, rout=Pr)
    # turn these into a net flux
    F[1], ctop[1] = lr_to_flux(params, G, Pl, Pr, 1, Loci.FACE1)

    reconstruct(params, G, P, 2, lout=Pl, rout=Pr)
    F[2], ctop[2] = lr_to_flux(params, G, Pl, Pr, 2, Loci.FACE2)

    reconstruct(params, G, P, 3, lout=Pl, rout=Pr)
    F[3], ctop[3] = lr_to_flux(params, G, Pl, Pr, 3, Loci.FACE3)

    if params['dt_static']:
        ndt = params['dt_start']
    else:
        ndt = ndt_min(params, G, ctop)

    return F, ndt


# Note that the sense of L/R flips from zone to interface during function call
Dl = None; Dr = None
def lr_to_flux(params, G, Pr, Pl, dir, loc):
    s = G.slices
    sh = G.shapes

    # These return dicts of clArrays
    global Dl, Dr
    if Dl is None:
        Dl = get_state(params, G, Pl, loc)
        Dr = get_state(params, G, Pr, loc)
    else:
        get_state(params, G, Pl, loc, out=Dl)
        get_state(params, G, Pr, loc, out=Dr)

    cmaxL, cminL = mhd_vchar(params, G, Pl, Dl, loc, dir)
    cmaxR, cminR = mhd_vchar(params, G, Pr, Dr, loc, dir)

    global knl_ctop
    if knl_ctop is None:
        code = add_ghosts("""
        <> cmax = if( cmaxL[i,j,k] >  cmaxR[i,j,k], cmaxL[i,j,k], cmaxR[i,j,k])
        <> cmin = if(-cminL[i,j,k] > -cminR[i,j,k], -cminL[i,j,k], -cminR[i,j,k])
        cmax0 := if(0. > cmax, 0., cmax)
        cmin0 := if(0. > cmin, 0., cmin)
        ctop[i,j,k] = if(cmax0 > cmin0, cmax0, cmin0)
        """)
        knl_ctop = lp.make_kernel(sh.isl_grid_scalar, code,
                                  [*scalarArrayArgs("ctop", "cmaxL", "cmaxR", "cminL", "cminR"), ...],
                                  assumptions=sh.assume_grid, default_offset=lp.auto,
                                  silenced_warnings='inferred_iname')
        # for var in ["cmax", "cmin", "cmax0", "cmin0"]:
        #     knl_ctop = lp.assignment_to_subst(knl_ctop, var)
        knl_ctop = tune_grid_kernel(knl_ctop, sh.halo1_primitives, ng=G.NG-1)
        print("Compiled ctop")

    ctop = cl_array.empty_like(cmaxL)
    evt, _ = knl_ctop(params['queue'], cmaxL=cmaxL, cmaxR=cmaxR, cminL=cminL, cminR=cminR, ctop=ctop)
    evt.wait()
    del cmaxL, cmaxR, cminL, cminR

    if 'debug' in params and params['debug']:
        # This is slow as it's host-bound
        if np.any(np.isnan((1/ctop).get()[s.bulkrh1])):
            raise ValueError("Ctop is 0 or NaN at {}".format(np.argwhere(np.isnan(ctop.get()[s.bulkrh1]))[0]))

    global Ul, Ur, fluxL, fluxR
    if Ul is None:
        Ul = cl_array.empty(params['queue'], sh.grid_primitives, dtype=np.float64)
        Ur = cl_array.empty(params['queue'], sh.grid_primitives, dtype=np.float64)
        fluxL = cl_array.empty(params['queue'], sh.grid_primitives, dtype=np.float64)
        fluxR = cl_array.empty(params['queue'], sh.grid_primitives, dtype=np.float64)

    prim_to_flux(params, G, Pl, Dl, 0, loc, out=Ul)
    prim_to_flux(params, G, Pr, Dr, 0, loc, out=Ur)

    prim_to_flux(params, G, Pl, Dl, dir, loc, out=fluxL)
    prim_to_flux(params, G, Pr, Dr, dir, loc, out=fluxR)

    flux = cl_array.empty(params['queue'], sh.grid_primitives, dtype=np.float64)
    global knl_addfluxes
    if knl_addfluxes is None:
        code = add_ghosts("""
        flux[p,i,j,k] = 0.5 * (fluxL[p,i,j,k] + fluxR[p,i,j,k] - ctop[i,j,k] * (Ur[p,i,j,k] - Ul[p,i,j,k]))
        """)
        knl_addfluxes = lp.make_kernel(sh.isl_grid_primitives, code,
                                       [*primsArrayArgs("flux", "fluxL", "fluxR", "Ul", "Ur"),
                                        *scalarArrayArgs("ctop"), ...],
                                       assumptions=sh.assume_grid,
                                       default_offset=lp.auto)
        knl_addfluxes = tune_prims_kernel(knl_addfluxes, sh.halo1_primitives, ng=G.NG-1)
        print("Compiled addfluxes")

    evt, _ = knl_addfluxes(params['queue'], flux=flux, fluxL=fluxL, fluxR=fluxR, ctop=ctop, Ur=Ur, Ul=Ul)
    if params['profile']:
        evt.wait()

    return flux, ctop


def flux_ct(params, G, F):
    sh = G.shapes

    global knl_flux_ct, knl_flux_ct_emf
    global emf1, emf2, emf3
    if knl_flux_ct is None:
        # Flux constrained transport from Toth 2000
        # Implementation adapted from Ben Ryan's ebhlight code
        # TODO can use locals here if careful
        code_emf = replace_prim_names(add_ghosts("""
        emf3[i, j, k] =  0.25*(F1[B2, i, j, k] + F1[B2, i, j - 1, k] \
                             - F2[B1, i, j, k] - F2[B1, i - 1, j, k])
        emf2[i, j, k] = -0.25*(F1[B3, i, j, k] + F1[B3, i, j, k - 1] \
                             - F3[B1, i, j, k] - F3[B1, i - 1, j, k])
        emf1[i, j, k] =  0.25*(F2[B3, i, j, k] + F2[B3, i, j, k - 1] \
                             - F3[B2, i, j, k] - F3[B2, i, j - 1, k])
        """))
        code_flux = replace_prim_names(add_ghosts("""
        F1[B1, i, j, k] = 0
        F1[B2, i, j, k] = 0.5 * (emf3[i, j, k] + emf3[i, j + 1, k])
        F1[B3, i, j, k] = -0.5 * (emf2[i, j, k] + emf2[i, j, k + 1])
        
        F2[B1, i, j, k] = -0.5 * (emf3[i, j, k] + emf3[i + 1, j, k])
        F2[B2, i, j, k] = 0
        F2[B3, i, j, k] = 0.5 * (emf1[i, j, k] + emf1[i, j, k + 1])
        
        F3[B1, i, j, k] = 0.5 * (emf2[i, j, k] + emf2[i + 1, j, k])
        F3[B2, i, j, k] = -0.5 * (emf1[i, j, k] + emf1[i, j + 1, k])
        F3[B3, i, j, k] = 0
        """))

        knl_flux_ct_emf = lp.make_kernel(sh.isl_grid_primitives, code_emf,
                                     [*primsArrayArgs("F1", "F2", "F3"),
                                      *scalarArrayArgs("emf1", "emf2", "emf3"), ...],
                                     assumptions=sh.assume_grid)
        knl_flux_ct_emf = tune_prims_kernel(knl_flux_ct_emf, sh.halo1_primitives, ng=G.NG-1)

        knl_flux_ct = lp.make_kernel(sh.isl_grid_primitives, code_flux,
                                     [*primsArrayArgs("F1", "F2", "F3"),
                                      *scalarArrayArgs("emf1", "emf2", "emf3"), ...],
                                     assumptions=sh.assume_grid)
        knl_flux_ct = tune_prims_kernel(knl_flux_ct, sh.halo1_primitives, ng=G.NG-1)

        #knl_flux_ct = lp.set_options(knl_flux_ct, "print_cl")
        #print(knl_flux_ct)
        print("Compiled flux_ct")

        emf1 = cl_array.empty(params['queue'], sh.grid_scalar, dtype=np.float64)
        emf2 = cl_array.empty(params['queue'], sh.grid_scalar, dtype=np.float64)
        emf3 = cl_array.empty(params['queue'], sh.grid_scalar, dtype=np.float64)

    # The queue is in-order for the foreseeable future. When it's not, we'll need a .wait() in here
    evt, _ = knl_flux_ct_emf(params['queue'], F1=F[1], F2=F[2], F3=F[3], emf1=emf1, emf2=emf2, emf3=emf3)
    evt, _ = knl_flux_ct(params['queue'], F1=F[1], F2=F[2], F3=F[3],  emf1=emf1, emf2=emf2, emf3=emf3)

    # Similarly, we might care about whether everything's up-to-date when we relinquish control
    return F
