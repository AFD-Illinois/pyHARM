# Fix egration/inversion failures

import numpy as np
import pyopencl.array as cl_array

import loopy as lp
from pyHARM.loopy_tools import *

from pyHARM.defs import Loci
from pyHARM.phys import get_state, mhd_gamma_calc, prim_to_flux
from pyHARM.u_to_p import U_to_P

from pyHARM.debug_tools import plot_prims, plot_var

# Floor Codes: bit masks
HIT_FLOOR_GEOM_RHO = 1
HIT_FLOOR_GEOM_U = 2
HIT_FLOOR_B_RHO = 4
HIT_FLOOR_B_U = 8
HIT_FLOOR_TEMP = 16
HIT_FLOOR_GAMMA = 32
HIT_FLOOR_KTOT = 64
FLOOR_UTOP_FAIL = 128

# Apply floors to density, internal energy
fflag = None
def fixup(params, G, P):
    sh = G.shapes

    global fflag
    if fflag is None:
        fflag = cl_array.empty(params['queue'], sh.grid_scalar, dtype=np.int16)

    fflag.fill(0)
    P, fflag = fixup_ceiling(params, fflag, G, P)
    P, fflag = fixup_floor(params, fflag, G, P)

    return P

    # Some debug info about floors
    #if DEBUG
    # if flag & HIT_FLOOR_GEOM_RHO: n_geom_rho += 1
    # if flag & HIT_FLOOR_GEOM_U: n_geom_u += 1
    # if flag & HIT_FLOOR_B_RHO: n_b_rho += 1
    # if flag & HIT_FLOOR_B_U: n_b_u += 1
    # if flag & HIT_FLOOR_TEMP: n_temp += 1
    # if flag & HIT_FLOOR_GAMMA: n_gamma += 1
    # if flag & HIT_FLOOR_KTOT: n_ktot += 1
    #
    # # n_geom_rho = mpi_reduce_(n_geom_rho)
    # # n_geom_u = mpi_reduce_(n_geom_u)
    # # n_b_rho = mpi_reduce_(n_b_rho)
    # # n_b_u = mpi_reduce_(n_b_u)
    # # n_temp = mpi_reduce_(n_temp)
    # # n_gamma = mpi_reduce_(n_gamma)
    # # n_ktot = mpi_reduce_(n_ktot)
    #
    # if n_geom_rho > 0: print("Hit %d GEOM_RHO".format(n_geom_rho))
    # if n_geom_u > 0: print("Hit %d GEOM_U".format(n_geom_u))
    # if n_b_rho > 0: print("Hit %d B_RHO".format(n_b_rho))
    # if n_b_u > 0: print("Hit %d B_U".format(n_b_u))
    # if n_temp > 0: print("Hit %d TEMPERATURE".format(n_temp))
    # if n_gamma > 0: print("Hit %d GAMMA".format(n_gamma))
    # if n_ktot > 0: print("Hit %d KTOT".format(n_ktot))


def fixup_ceiling(params, fflag, G, P):
    s = G.slices

    # First apply ceilings:
    # 1. Limit gamma with respect to normal observer

    # TODO is there a softer touch here?
    gamma = mhd_gamma_calc(params['queue'], G, P, Loci.CENT)
    f = cl_array.if_positive(gamma - params['gamma_max'],
                             ((params['gamma_max']**2 - 1.) / (gamma**2 - 1.))**(1/2),
                             cl_array.empty_like(gamma).fill(1))
    P[s.U1] *= f
    P[s.U2] *= f
    P[s.U3] *= f

    # 2. Limit KTOT
    if params['electrons']:
        # Keep to KTOTMAX by controlling u, to avoid anomalous cooling from funnel wall
        # TODO This operates on last iteration's KTOT, meaning the effective value can escape the ceiling. Rethink
        u_max_ent = params['entropy_max'] * (P[s.RHO] ** params['gam'])/(params['gam']-1.)
        P[s.UU] = cl_array.if_positive(P[s.UU] - u_max_ent, u_max_ent, P[s.UU])
        P[s.KTOT] = cl_array.if_positive(P[s.KTOT] - params['entropy_max'], params['entropy_max'], P[s.KTOT])
        pass

    # TODO keep track of hits
    #fflag |= cl_array.if_positive(gamma - params['gamma_max'], temp.fill(HIT_FLOOR_GAMMA), zero)

    return P, fflag


rhoflr_geom = None
uflr_geom = None
U = None
D = None
Padd = None
Uadd = None
Dadd = None
dzero = None
knl_floors = None
def fixup_floor(params, fflag, G, P):
    s = G.slices
    sh = G.shapes
    # Then apply floors:
    # 1. Precalculate geometric hard floors, not based on fluid relationships
    global rhoflr_geom, uflr_geom, U, D, Padd, Uadd, Dadd, dzero
    if rhoflr_geom is None:
        if "mks" in params['coordinates']:
            # New, steeper floor in rho
            # Previously raw r^-2 or r^-1.5
            r = G.coords.r(G.coord_all())
            rhoscal = 1/(r**2) * 1 / (1 + r/params['floor_char_r'])
            # Impose minimum rho with scaling above, minimum u as rho**gam
            rhoflr_geom = cl_array.to_device(params['queue'],
                                             np.maximum(params['rho_min'] * rhoscal, params['rho_min_limit']))
            uflr_geom = cl_array.to_device(params['queue'],
                                           np.maximum(params['u_min']*(rhoscal**params['gam']), params['u_min_limit']))
        elif "minkowski" in params['coordinates']:
            rhoflr_geom = cl_array.empty(params['queue'], sh.grid_scalar, dtype=np.float64).fill(params['rho_min']*1.e-2)
            uflr_geom = cl_array.empty(params['queue'], sh.grid_scalar, dtype=np.float64).fill(params['u_min']*1.e-2)
        # Arrays we should keep on hand: derived values and values for additional fluid packet
        U = cl_array.zeros_like(P)
        D = get_state(params, G, P, Loci.CENT)
        Padd = cl_array.zeros_like(P)
        Uadd = cl_array.zeros_like(P)
        Dadd = get_state(params, G, P, Loci.CENT)
        dzero = cl_array.zeros_like(P)

    global knl_floors
    if knl_floors is None:
        code = add_ghosts(replace_prim_names("""
        # 2. Magnetic floors: impose maximum magnetization sigma = bsq/rho, inverse beta prop. to bsq/U
        rhoflr_b := bsq[i,j,k] / sig_max
        uflr_b := bsq[i,j,k] / (sig_max * temp_max)
        
        # Maximum U floor
        uflr_max := if(uflr_b > uflr_geom[i,j,k], uflr_b, uflr_geom[i,j,k])
        
        # 3. Temperature ceiling: impose maximum temperature
        temp_real := P[UU,i,j,k] / temp_max
        temp_floor := uflr_max / temp_max
        rhoflr_temp := if(temp_real > temp_floor, temp_real, temp_floor)
        
        # Maximum rho floor
        rhoflr_max1 := if(rhoflr_geom[i,j,k] > rhoflr_b, rhoflr_geom[i,j,k], rhoflr_b)
        rhoflr_max := if(rhoflr_max1 > rhoflr_temp, rhoflr_max1, rhoflr_temp)
        
        # Initialize a dummy fluid parcel with any missing mass and internal energy, but not velocity
        rho_add := rhoflr_max - P[RHO,i,j,k]
        u_add := uflr_max - P[UU,i,j,k]
        Padd[RHO,i,j,k] = if(rho_add > 0, rho_add, 0) {id=p1,nosync=p2}
        Padd[UU,i,j,k] = if(u_add > 0, u_add, 0)      {id=p2,nosync=p1}
        """))
        knl_floors = lp.make_kernel(sh.isl_grid_scalar, code,
                                                [*primsArrayArgs("P", "Padd"),
                                                 *scalarArrayArgs("bsq", "rhoflr_geom", "uflr_geom"),
                                                 ...])
        knl_floors = lp.fix_parameters(knl_floors, sig_max=params['sigma_max'], temp_max=params['u_over_rho_max'],
                                       nprim=params['n_prim'])
        knl_floors = tune_grid_kernel(knl_floors, shape=sh.bulk_scalar, ng=G.NG,
                                      prefetch_args=["bsq", "rhoflr_geom", "uflr_geom"])
        print("Compiled fixup_floors")

    # Bulk call before bsq calculation below
    get_state(params, G, P, Loci.CENT, out=D)  # Reused below!
    bsq = G.dot(D['bcon'], D['bcov'])

    Padd.fill(0)
    evt, _ = knl_floors(params['queue'], P=P, bsq=bsq, rhoflr_geom=rhoflr_geom, uflr_geom=uflr_geom, Padd=Padd)
    evt.wait()

    if params['debug']:
        rhits = np.count_nonzero(Padd[s.RHO].get())
        uhits = np.count_nonzero(Padd[s.UU].get())
        print("Rho floors hit {} times".format(rhits))
        print("U floors hit {} times".format(uhits))

    if cl_array.max(Padd) > 0.:
        print("Fixing up")
        # Get conserved variables for the parcel
        get_state(params, G, Padd, Loci.CENT, out=Dadd)
        prim_to_flux(params, G, Padd, Dadd, 0, Loci.CENT, out=Uadd)

        # And for the current state
        #get_state(G, S, i, j, k, CENT, out=D) # Called just above
        prim_to_flux(params, G, P, D, 0, Loci.CENT, out=U)

        # Recover primitive variables.  Don't touch what we don't have to
        Ptmp, pflag = U_to_P(params, G, U+Uadd, P+Padd)
        P = cl_array.if_positive(Padd, Ptmp, P)
        del Ptmp

    # TODO Record specific floor hits/U_to_P fails
    #fflag |= cl_array.if_positive(rhoflr_geom - P[s.RHO], temp.fill(HIT_FLOOR_GEOM_RHO), zero)
    #fflag |= cl_array.if_positive(uflr_geom - P[s.UU], temp.fill(HIT_FLOOR_GEOM_U), zero)
    #fflag |= cl_array.if_positive(rhoflr_b - P[s.RHO], temp.fill(HIT_FLOOR_B_RHO), zero)
    #fflag |= cl_array.if_positive(uflr_b - P[s.UU], temp.fill(HIT_FLOOR_B_U), zero)
    #fflag |= cl_array.if_positive(rhoflr_temp - P[s.RHO], temp.fill(HIT_FLOOR_TEMP), zero)
    #fflag |= cl_array.if_positive(pflag, temp.fill(FLOOR_UTOP_FAIL), zero)

    if params['electrons']:
        # Reset the entropy after floor application
        P[s.KTOT] = (params['gam'] - 1.) * P[s.UU] / (P[s.RHO]**params['gam'])

    return P, fflag


# Replace bad pos with values interpolated from neighbors
knl_fixup_utoprim_sums = None
knl_fixup_utoprim_fix = None
def fixup_utoprim(params, pflag, G, P):
    sh = G.shapes
    s = G.slices

    if params['debug']:
        nbad_utop = np.sum(pflag.get()[s.bulk] != 0)
        print("Fixing {} bad cells".format(nbad_utop))

    # Make sure we are not using ill defined physical corner regions
    # TODO can this be forgotten?  U_to_P only updates the bulk, and bounds should not touch physical corners
    #zero_corners(params, G, pflag)

    sum = cl_array.zeros(params['queue'], sh.grid_primitives, dtype=np.float64)
    wsum = cl_array.zeros(params['queue'], sh.grid_scalar, dtype=np.float64)

    global knl_fixup_utoprim_sums, knl_fixup_utoprim_fix
    if knl_fixup_utoprim_sums is None:
        # TODO these should really be combined and the check on wsum inlined
        # That's gonna be a project
        code_sums = add_ghosts("""
        # TODO if statements here to speed up evaluation?
        w(l, m, n) := not(pflag[i+l,j+m,k+n]) / (abs(l) + abs(m) + abs(n) + 1)
        wsum[i, j, k] = reduce(sum, (l,m,n), w(l,m,n))
        sum[p, i, j, k] = reduce(sum, (l,m,n), w(l,m,n) * P[p, i+l, j+m, k+n])
        """)
        code_fixup = add_ghosts("""
        P[p, i, j, k] = if(pflag[i, j, k] == 0, P[p, i, j, k], sum[p, i, j, k] / wsum[i, j, k])
        """)
        knl_fixup_utoprim_sums = lp.make_kernel(sh.isl_grid_primitives_fixup, code_sums,
                                                [*primsArrayArgs("P", "sum"), *scalarArrayArgs("wsum"),
                                                 *scalarArrayArgs("pflag", dtype=np.int32)],
                                                assumptions=sh.assume_grid)
        knl_fixup_utoprim_sums = spec_prims_kernel(knl_fixup_utoprim_sums, sh.bulk_primitives, ng=G.NG)
        # Roll our own optimization here as this is the only convolution kernel we got
        knl_fixup_utoprim_sums = lp.split_iname(knl_fixup_utoprim_sums, "k", 8, outer_tag="g.0", inner_tag="l.0")
        knl_fixup_utoprim_sums = lp.split_iname(knl_fixup_utoprim_sums, "j", 8, outer_tag="g.1", inner_tag="l.1")
        knl_fixup_utoprim_sums = lp.split_iname(knl_fixup_utoprim_sums, "i", 8, outer_tag="g.2", inner_tag="l.2")
        knl_fixup_utoprim_sums = lp.make_reduction_inames_unique(knl_fixup_utoprim_sums)

        # TODO these are some feisty prefetches. Leaving them for later
        # knl_fixup_utoprim_sums = lp.tag_inames(knl_fixup_utoprim_sums, "p:unr")
        # knl_fixup_utoprim_sums = lp.add_prefetch(knl_fixup_utoprim_sums, "pflag", "i_inner,j_inner,k_inner",
        #                                          default_tag="l.auto")
        # knl_fixup_utoprim_sums = lp.add_prefetch(knl_fixup_utoprim_sums, "P", "i_inner,j_inner,k_inner,l,m,n",
        #                                          default_tag="l.auto")

        # TODO The prefetches on this are not working either, look at that
        knl_fixup_utoprim_fix = lp.make_kernel(sh.isl_grid_primitives, code_fixup,
                                                [*primsArrayArgs("P", "sum"), *scalarArrayArgs("wsum"),
                                                 *scalarArrayArgs("pflag", dtype=np.int32)],
                                               assumptions=sh.assume_grid)
        knl_fixup_utoprim_fix = tune_prims_kernel(knl_fixup_utoprim_fix, shape=sh.bulk_primitives, ng=G.NG)
        print("Compiled fixup_utoprim")

    evt, _ = knl_fixup_utoprim_sums(params['queue'], P=P, pflag=pflag, sum=sum, wsum=wsum)
    evt.wait()
    if params['debug']:
        if np.any(wsum.get()[s.bulk] < 1.e-10):
            # TODO don't die on this when we hit prod
            raise ValueError("fixup_utoprim found no usable neighbors!")
    evt, _ = knl_fixup_utoprim_fix(params['queue'], P=P, pflag=pflag, sum=sum, wsum=wsum)

    if params['debug']:
        # TODO count what we fixed
        nleft_utop = nbad_utop - nbad_utop
        if nleft_utop > 0:
            print("Cells STILL BAD after fixup_utoprim: {}".format(nleft_utop))

    # Reset the pflag, because we tried our best and that's what counts
    # TODO necessary? See above about new copy
    #pflag.fill(0)

    return P

def zero_corners(params, G, pflag):
    s = G.slices
    pflag = pflag.get()
    if G.global_start[2] == 0 and G.global_start[1] == 0 and G.global_start[0] == 0:
        pflag[s.ghostl, s.ghostl, s.ghostl] = 0
    if G.global_start[2] == 0 and G.global_start[1] == 0 and G.global_stop[0] == G.NTOT[1]:
        pflag[s.ghostl, s.ghostl, s.ghostr] = 0
    if G.global_start[2] == 0 and G.global_stop[1] == G.NTOT[2] and G.global_start[0] == 0:
        pflag[s.ghostl, s.ghostr, s.ghostl] = 0
    if G.global_stop[2] == G.NTOT[3] and G.global_start[1] == 0 and G.global_start[0] == 0:
        pflag[s.ghostr, s.ghostl, s.ghostl] = 0
    if G.global_start[2] == 0 and G.global_stop[1] == G.NTOT[2] and G.global_stop[0] == G.NTOT[1]:
        pflag[s.ghostl, s.ghostr, s.ghostr] = 0
    if G.global_stop[2] == G.NTOT[3] and G.global_start[1] == 0 and G.global_stop[0] == G.NTOT[1]:
        pflag[s.ghostr, s.ghostl, s.ghostr] = 0
    if G.global_stop[2] == G.NTOT[3] and G.global_stop[1] == G.NTOT[2] and G.global_start[0] == 0:
        pflag[s.ghostr, s.ghostr, s.ghostl] = 0
    if G.global_stop[2] == G.NTOT[3] and G.global_stop[1] == G.NTOT[2] and G.global_stop[0] == G.NTOT[1]:
        pflag[s.ghostr, s.ghostr, s.ghostr] = 0
    return cl_array.to_device(params['queue'], pflag)
