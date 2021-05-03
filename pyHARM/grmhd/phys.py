# Physics functions

import numpy as np

from pyHARM.defs import Var, Loci


def Tmhd(params, G, P, D, dir, slc=None):
    """Compute the MHD stress-energy tensor with first index up, second index down.
    Note a factor of sqrt(4 pi) is absorbed into the definition of b.
    """
    # TODO sort this out/unify with T_mixed in ana.
    # phys.py and variables.py can actually share most stuff...
    s = G.slices
    sh = G.shapes
    if slc is None:
        slc = s.all

    w = P[s.RHO+slc] + params['gam'] * P[s.UU+slc]
    bsq = G.dot(D['bcov'], D['bcon'])
    eta = w + bsq
    ptot = (params['gam'] - 1) * P[s.UU+slc] + 0.5 * bsq

    T = np.zeros(sh.grid_vector)
    for mu in range(4):
        T[mu][slc] = (eta * D['ucon'][dir][slc] * D['ucov'][mu][slc] +
                            ptot * (mu == dir) -
                            D['bcon'][dir][slc] * D['bcov'][mu][slc])

    return T

def set_fourvel_t(gcov, ucon):
    AA = gcov[0][0]
    BB = 2. * (gcov[0][1] * ucon[1] + \
               gcov[0][2] * ucon[2] + \
               gcov[0][3] * ucon[3])
    CC = 1. + gcov[1][1] * ucon[1] * ucon[1] + \
         gcov[2][2] * ucon[2] * ucon[2] + \
         gcov[3][3] * ucon[3] * ucon[3] + \
         2. * (gcov[1][2] * ucon[1] * ucon[2] + \
               gcov[1][3] * ucon[1] * ucon[3] + \
               gcov[2][3] * ucon[2] * ucon[3])

    discr = BB * BB - 4. * AA * CC
    ucon[0] = (-BB - np.sqrt(discr)) / (2. * AA)

def fourvel_to_prim(gcon, ucon, u_prim):
    alpha2 = -1.0 / gcon[0][0]
    # Note gamma/alpha is ucon[0]
    u_prim[1] = ucon[1] + ucon[0] * alpha2 * gcon[0][1]
    u_prim[2] = ucon[2] + ucon[0] * alpha2 * gcon[0][2]
    u_prim[3] = ucon[3] + ucon[0] * alpha2 * gcon[0][3]


def prim_to_flux(params, G, P, D=None, dir=0, loc=Loci.CENT, slc=None):
    """Calculate the Lax-Friedrichs flux or """
    s = G.slices
    if slc is None:
        slc = s.all
    gslc = s.geom_slc(slc)

    if D is None:
        D = get_state(G, P, loc, slc)

    flux = np.zeros_like(P)

    # Particle number flux
    flux[s.RHO + slc] = P[s.RHO + slc] * D['ucon'][dir][slc]

    # MHD stress-energy tensor w/ first index up, second index down
    T = Tmhd(params, G, P, D, dir, slc)
    flux[s.UU + slc] = T[0][slc] + flux[s.RHO + slc]
    flux[s.U3VEC + slc] = T[s.VEC3 + slc]

    # Dual of Maxwell tensor
    flux[s.B1 + slc] = (D['bcon'][1] * D['ucon'][dir] - D['bcon'][dir] * D['ucon'][1])[slc]
    flux[s.B2 + slc] = (D['bcon'][2] * D['ucon'][dir] - D['bcon'][dir] * D['ucon'][2])[slc]
    flux[s.B3 + slc] = (D['bcon'][3] * D['ucon'][dir] - D['bcon'][dir] * D['ucon'][3])[slc]

    if 'electrons' in params and params['electrons']:
        flux[s.KEL] = flux[s.RHO]*P[s.KEL]
        flux[s.KTOT] = flux[s.RHO]*P[s.KTOT]

    flux[s.allv + slc] *= G.gdet[loc.value][gslc]
    return flux


def mhd_gamma_calc(G, P, loc=Loci.CENT, slc=None):
    """Find relativistic gamma-factor w.r.t. normal observer"""
    s = G.slices
    if slc is None:
        slc = s.all
    gslc = s.geom_slc(slc)

    qsq = np.zeros_like(P[s.RHO])
    qsq[slc] = (G.gcov[(loc.value, 1, 1) + gslc] * P[s.U1 + slc] ** 2 +
                G.gcov[(loc.value, 2, 2) + gslc] * P[s.U2 + slc] ** 2 +
                G.gcov[(loc.value, 3, 3) + gslc] * P[s.U3 + slc] ** 2) + \
                2. * (G.gcov[(loc.value, 1, 2) + gslc] * P[s.U1 + slc] * P[s.U2 + slc] +
                      G.gcov[(loc.value, 1, 3) + gslc] * P[s.U1 + slc] * P[s.U3 + slc] +
                      G.gcov[(loc.value, 2, 3) + gslc] * P[s.U2 + slc] * P[s.U3 + slc])

    return np.sqrt(1. + qsq)


def ucon_calc(G, P, loc, slc=None):
    """Find contravariant fluid four-velocity"""
    s = G.slices
    if slc is None:
        slc = s.all
    gslc = s.geom_slc(slc)

    gamma = mhd_gamma_calc(G, P, loc, slc)

    ucon = np.zeros([4, *gamma.shape])
    ucon[0][slc] = gamma[slc] / G.lapse[loc.value][gslc]
    for mu in range(1, 4):
        ucon[mu][slc] = P[Var.U1.value + mu - 1][slc] - \
                        gamma[slc] * G.lapse[loc.value][gslc] * G.gcon[loc.value, 0, mu][gslc]

    return ucon


def bcon_calc(G, P, ucon, ucov, slc=None):
    """Calculate magnetic field four-vector"""
    s = G.slices
    if slc is None:
        slc = s.all

    bcon = np.zeros_like(ucon)
    bcon[0][slc] = P[s.B1 + slc] * ucov[1][slc] + \
                    P[s.B2 + slc] * ucov[2][slc] + \
                    P[s.B3 + slc] * ucov[3][slc]
    for mu in range(1, 4):
        bcon[mu][slc] = (P[Var.B1.value - 1 + mu][slc] +
                    bcon[0][slc] * ucon[mu][slc]) / ucon[0][slc]

    return bcon


def get_state(params, G, P, loc=Loci.CENT, slc=None):
    """
    Calculate ucon, ucov, bcon, bcov from primitive variables
    Returns a dict D ("derived") of state variables
    :param params: Only taken for compatibility with OpenCL version. Remove...
    :param G:
    :param P:
    :param loc:
    :return:
    """
    D = {'ucon': ucon_calc(G, P, loc, slc)}
    D['ucov'] = G.lower_grid(D['ucon'], loc)
    D['bcon'] = bcon_calc(G, P, D['ucon'], D['ucov'], slc)
    D['bcov'] = G.lower_grid(D['bcon'], loc)
    return D


# Calculate components of magnetosonic velocity from primitive variables
def mhd_vchar(params, G, P, D, loc, dir, slc=None):
    s = G.slices
    sh = G.shapes
    if slc is None:
        slc = s.all

    rho = np.abs(P[s.RHO + slc])
    u = np.abs(P[s.UU + slc])

    Acov = np.zeros([4, *rho.shape])
    Acov[dir] = 1.

    Bcov = np.zeros_like(Acov)
    Bcov[0] = 1

    Acon = G.raise_grid(Acov, loc)
    Bcon = G.raise_grid(Bcov, loc)

    # Find fast magnetosonic speed
    # Trim vectors as we dot them to save lines
    gam = params['gam']
    bsq = G.dot(D['bcon'], D['bcov'])
    ef = rho + gam * u
    ee = bsq + ef
    va2 = bsq / ee
    cs2 = gam * (gam - 1.) * u / ef

    cms2 = cs2 + va2 - cs2 * va2

    cms2[np.where(cms2 < 0)] = 1e-20
    cms2[np.where(cms2 > 1)] = 1

    # Require that speed of wave measured by observer q->ucon is cms2
    Asq = G.dot(Acon, Acov)
    Bsq = G.dot(Bcon, Bcov)
    Au = G.dot(Acov, D['ucon'][s.allv+slc])
    Bu = G.dot(Bcov, D['ucon'][s.allv+slc])

    AB = G.dot(Acon, Bcov)

    A = Bu**2 - (Bsq + Bu**2) * cms2
    B = 2. * (Au*Bu - (AB + Au*Bu) * cms2)
    C = Au**2 - (Asq + Au**2) * cms2

    discr = B**2 - 4. * A * C
    discr[discr < 0] = 0
    np.sqrt(discr, out=discr)

    vp = np.zeros(sh.grid_scalar)
    vm = np.zeros(sh.grid_scalar)
    vp[slc] = -(-B + discr) / (2. * A)
    vm[slc] = -(-B - discr) / (2. * A)

    return np.maximum(vp, vm), np.minimum(vp, vm)


def get_fluid_source(params, G, P, D, slc=None):
    """Calculate a small fluid source term, added to conserved variables for stability"""
    s = G.slices
    if slc is None:
        slc = s.all
    gslc = s.geom_slc(slc)

    T = np.zeros([4, 4, *P[s.RHO].shape])
    for mu in range(4):
        T[mu] = Tmhd(params, G, P, D, mu)

    # Contract mhd stress tensor with connection
    dU = np.zeros_like(P)
    for mu in range(4):
        for nu in range(4):
            for gam in range(4):
                dU[Var.UU.value + gam][slc] += T[mu, nu][slc] * G.conn[nu, gam, mu][gslc]

    dU[slc] *= G.gdet[Loci.CENT.value][gslc]

    return dU