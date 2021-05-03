# Inverts from conserved variables to primitives.
# Based on Mignone & McKinney 2007, Noble et al 2006

import numpy as np
import pyopencl.array as cl_array

from pyHARM.defs import Loci


def U_to_P(params, G, U, P, loc=Loci.CENT, slc=None):
    err_tol = params['invert_err_tol']
    iter_max = params['invert_iter_max']
    delta = params['invert_iter_delta']
    gamma_max = params['gamma_max']
    gam = params['gam']

    if 'queue' in params:
        U = U.get()
        P = P.get()

    # This method defaults to bulk-only, being outside the flux calculation
    s = G.slices
    sh = G.shapes
    gslc = s.geom_slc(s.all)

    # TODO avoid re-slicing this.  Currently convenient for adding the 3rd index
    gdet = G.gdet[Loci.CENT.value]
    lapse = G.lapse[Loci.CENT.value]

    # Update the primitive B-fields
    P[s.B3VEC] = U[s.B3VEC]/gdet[gslc]

    # Eflag will indicate inversion failures --
    # Like all temporary variables in here, it is sized to bulk only
    eflag = np.zeros(sh.grid_scalar, dtype=np.int32)
    # Catch negative density early
    # TODO this without effectively testing double equality
    eflag[np.where(U[s.RHO] < 0.)] = -100

    # Convert from conserved variables to four-vectors
    # All temporary variables are bulk-sized only!
    D = U[s.RHO]*lapse[gslc]/gdet[gslc]

    # TODO shape to actual slice we're called with...
    Bcon = np.zeros(sh.grid_vector)
    Bcon[1:] = U[s.B3VEC] * lapse[gslc] / gdet[gslc]

    Qcov = np.zeros_like(Bcon)
    Qcov[0] = (U[s.UU] - U[s.RHO]) * lapse[gslc] / gdet[gslc]
    Qcov[1:] = U[s.U3VEC] * lapse[gslc] / gdet[gslc]

    ncov = np.zeros_like(Qcov)
    ncov[0] = -lapse[gslc]

    Bcov = G.lower_grid(Bcon, ocl=False)
    Qcon = G.raise_grid(Qcov, ocl=False)
    ncon = G.raise_grid(ncov, ocl=False)

    # This will have fringes of zeros still!
    Bsq = np.sum(Bcon * Bcov, axis=0)
    QdB = np.sum(Bcon * Qcov, axis=0)
    Qdotn = np.sum(Qcon * ncov, axis=0)
    Qsq = np.sum(Qcon * Qcov, axis=0)

    Qtsq = Qsq + Qdotn**2

    # Set up eqn for W' this is the energy density
    Ep = -Qdotn - D

    # Numerical rootfinding
    # Take guesses from primitive
    Wp = Wp_func(G, P, loc, gam, gamma_max, eflag)
    if np.any(eflag != 0):
        raise ValueError("Unexpected flag set!")

    # Step around the guess & evaluate errors
    Wpm = (1 - delta)*Wp
    h = Wp - Wpm
    Wpp = Wp + h
    errp = err_eqn(Bsq, D, Ep, QdB, Qtsq, Wpp, gam, gamma_max, eflag)
    err = err_eqn(Bsq, D, Ep, QdB, Qtsq, Wp, gam, gamma_max, eflag)
    errm = err_eqn(Bsq, D, Ep, QdB, Qtsq, Wpm, gam, gamma_max, eflag)

    # Attempt a Halley/Muller/Bailey/Press step
    dedW = (errp - errm) / (Wpp - Wpm)
    dedW2 = (errp - 2.*err + errm) / (h**2)

    f = 0.5*err*dedW2/(dedW**2)
    # Limit size of 2nd derivative correction
    np.clip(f, -0.3, 0.3, out=f)

    dW = -err/dedW/(1. - f)
    Wp1 = np.copy(Wp)
    err1 = np.copy(err)
    # Limit size of step
    np.clip(dW, -0.5*Wp, 2.0*Wp, out=dW)

    Wp += dW
    err = err_eqn(Bsq, D, Ep, QdB, Qtsq, Wp, gam, gamma_max, eflag)

    # Not good enough?  apply secant method
    #iter_flag = np.logical_not(np.logical_or(np.abs(dW/Wp) < err_tol, np.abs(err/Wp) < err_tol))
    iter_flag = np.zeros_like(eflag)
    iter_flag[s.bulk] = True
    for niter in range(iter_max):
        print("U_to_P iteration")
        dW[iter_flag] = (Wp1[iter_flag] - Wp[iter_flag]) * err[iter_flag]/\
                        (err[iter_flag] - err1[iter_flag])

        # Preserve last values
        Wp1[iter_flag] = Wp[iter_flag]
        err1[iter_flag] = err[iter_flag]

        # Normal secant increment is dW. Also limit guess to between 0.5 and 2
        # times the current value
        np.clip(dW, -0.5*Wp, 2.0*Wp, out=dW)
        Wp[iter_flag] += dW[iter_flag]

        # Set flag not to continue in zones that have converged
        iter_flag[np.abs(dW/Wp) < err_tol] = False

        err = err_eqn(Bsq, D, Ep, QdB, Qtsq, Wp, gam, gamma_max, eflag)
        iter_flag[np.abs(err/Wp) < err_tol] = False

        # Break only if nobody needs another iteration
        if not np.any(iter_flag):
            break

    # If secant method failed to converge, do not set primitives other than B
    eflag += iter_flag

    # Find utsq, gamma, rho0 from Wp
    gamma = gamma_func(Bsq, D, QdB, Qtsq, Wp, gamma_max, eflag)
    if np.any(gamma < 1.):
        raise ValueError("gamma < 1 failure!")

    # Find the scalars
    rho0 = D / gamma
    W = Wp + D
    w = W / (gamma**2)
    pres = pressure_rho0_w(rho0, w, gam)
    u = w - (rho0 + pres)

    # Return without updating non-B primitives
    eflag[rho0 < 0] = 6
    eflag[u < 0] = 7
    eflag[np.logical_and(u < 0, rho0 < 0)] = 8

    # Set primitives only where eflag is 0
    # Newly calculated values are all of size slc already, due to trimming above
    P[s.RHO][eflag == 0] = rho0[eflag == 0]
    P[s.UU][eflag == 0] = u[eflag == 0]

    # Find u(tilde) Eqn. 31 of Noble et al.
    Qtcon = Qcon + ncon * Qdotn
    P[s.U1][eflag == 0] = ((gamma / (W + Bsq))*(Qtcon[1] + QdB*Bcon[1]/W))[eflag == 0]
    P[s.U2][eflag == 0] = ((gamma / (W + Bsq))*(Qtcon[2] + QdB*Bcon[2]/W))[eflag == 0]
    P[s.U3][eflag == 0] = ((gamma / (W + Bsq))*(Qtcon[3] + QdB*Bcon[3]/W))[eflag == 0]

    if 'queue' in params:
        return cl_array.to_device(params['queue'], P), cl_array.to_device(params['queue'], eflag)
    else:
        return P, eflag


def err_eqn(Bsq,  D,  Ep,  QdB,  Qtsq, Wp, gam, gamma_max, eflag):
    W = Wp + D
    gamma = gamma_func(Bsq, D, QdB, Qtsq, Wp, gamma_max, eflag)
    w = W / (gamma**2)
    rho0 = D/gamma
    p = pressure_rho0_w(rho0, w, gam)

    return -Ep + Wp - p + 0.5*Bsq + 0.5*(Bsq*Qtsq - QdB**2)/((Bsq + W)**2)


def gamma_func(Bsq,  D,  QdB,  Qtsq,  Wp, gamma_max, eflag, slc=None):
    W = D + Wp
    WB = W + Bsq

    # This is basically inversion of eq. A7 of MM
    utsq = -((W + WB) * QdB**2 + W**2 * Qtsq) / (QdB**2 * (W + WB) + W**2 * (Qtsq - WB**2))
    gamma = np.sqrt(1. + np.abs(utsq))

    # Catch utsq < 0
    eflag[np.where(np.logical_or(utsq < 0., utsq > 1.e3 * gamma_max**2))] = 2

    return gamma


def Wp_func(G, P, loc, gam, gamma_max, eflag):
    s = G.slices
    sh = G.shapes

    rho0 = P[s.RHO]
    u = P[s.UU]

    # Again, vectors are done full-grid
    utcon = np.zeros(sh.grid_vector)
    utcon[0] = 0
    utcon[1:] = P[s.U3VEC]
    utcov = G.lower_grid(utcon, loc, ocl=False)

    utsq = G.dot(utcon, utcov, ocl=False)

    # Catch utsq < 0
    slc_utsq = np.where(np.logical_and(utsq < 0., np.abs(utsq) < 1.e-13))
    utsq[slc_utsq] = np.abs(utsq[slc_utsq])

    slc_utsq = np.where(np.logical_or(utsq < 0., utsq > 1.e3*gamma_max**2))
    utsq[slc_utsq] = rho0[slc_utsq] + u[slc_utsq]  # Not sure what to do here...
    eflag[slc_utsq] = 2

    # TODO why do we abs() twice, here and above?
    gamma = np.sqrt(1. + np.abs(utsq))
    Wp = (rho0 + u + pressure_rho0_u(rho0, u, gam)) * gamma ** 2 - rho0 * gamma
    # print(rho0)
    return Wp


# Equation of state
def pressure_rho0_u(rho0, u, gam):
    return (gam - 1.) * u


def pressure_rho0_w(rho0, w, gam):
    return (w - rho0)*(gam - 1.)/gam

