# Data reductions (integrals, averages, statistics) for fluid simulation output data

import numpy as np
import scipy.fftpack as fft

# This is too darn useful
from pyHARM.util import i_of

# General interface for reductions:
# (dump, var, options)
# dump: IharmDump object (or anything which provides dump['gdet'] and dump.header['nx'] members).  Sometimes omitted.
# var: ndarray of 3 dimensions, or a vector with vector index //first// (mu,i,j,k)
# other options as described below.  Certain zone arguments are necessary when a reduction is otherwise ambiguous

# TODO PDFs


## Correlation functions/lengths ##


def corr_midplane(var, norm=True, at_i1=None):
    """Angular correlation function at the midplane,
     of an array representing a variable in spherical-like coordinates r,th,phi
     """
    if at_i1 is None:
        at_i1 = range(var.shape[0])
    if isinstance(at_i1,int):
        at_i1 = [at_i1]

    # This selects the midplane-adjacent zones N2/2-1 & N2/2
    jmin = var.shape[1] // 2 - 1
    jmax = var.shape[1] // 2 + 1

    R = np.zeros((len(at_i1), var.shape[2]))

    # TODO is there a way to vectorize over R? Also, are we going to average over adjacent r ever?
    for i_out,i1 in enumerate(at_i1):
        # Average over small angle around midplane
        var_phi = np.mean(var[i1, jmin:jmax, :], axis=0)
        # Calculate autocorrelation
        var_phi_normal = (var_phi - np.mean(var_phi)) / np.std(var_phi)
        var_corr = fft.ifft(np.abs(fft.fft(var_phi_normal)) ** 2)
        R[i_out] = np.real(var_corr) / var_corr.size

    if norm:
        normR = R[:, 0]
        for k in range(var.shape[2]):
            R[:, k] /= normR

    if R.shape[0] == 1:
        return R[0,:]
    else:
        return R


def corr_midplane_direct(var, norm=True):
    """Alternate more volatile implementation of corr_midplane"""
    jmin = var.shape[1] // 2 - 1
    jmax = var.shape[1] // 2 + 1

    var_norm = np.ones((var.shape[0], 2, var.shape[2]))
    # Normalize radii separately
    for i in range(var.shape[0]):
        vmean = np.mean(var[i, jmin:jmax, :])
        var_norm[i, :, :] = var[i, jmin:jmax, :] - vmean

    R = np.ones((var.shape[0], var.shape[2]))
    for k in range(var.shape[2]):
        R[:, k] = np.sum(var_norm * np.roll(var_norm, k, axis=-1), axis=(1, 2)) / 2

    if norm:
        normR = R[:, 0]
        for k in range(var.shape[2]):
            R[:, k] /= normR

    return R


def corr_length_phi(R, interpolate=True):
    """Correlation "length" (angle) given a correlation function of r,phi"""
    lam = np.zeros(R.shape[0])
    for i in range(R.shape[0]):
        k = 0
        target = R[i, 0] / np.exp(1)
        while k < R.shape[1] and R[i, k] >= target:
            k += 1

        if interpolate and k < R.shape[1]:
            k = (target + (k - 1) * R[i, k] - k * R[i, k - 1]) / (R[i, k] - R[i, k-1])

        # This is specific to phi
        lam[i] = k * (2 * np.pi / R.shape[1])
    return lam


## Power Spectra ##


def pspec(var, dt=5, window=0.33, half_overlap=False, bin="fib"):
    """Power spectrum of a 1D timeseries, with various windows/binning.
    Currently only supports equally spaced times dt.
    """
    if not np.any(var[var.size // 2:]):
        return np.zeros_like(var), np.zeros_like(var)

    data = var[var.size // 2:]
    data = data[np.nonzero(data)] - np.mean(data[np.nonzero(data)])

    if window < 1:
        window = int(window * data.size)
    print("FFT window is {} of {} samples".format(window, data.size))

    print("Sampling time is {}".format(dt))
    out_freq = np.abs(fft.fftfreq(window, dt))

    if half_overlap:
        # Hanning w/50% overlap
        spacing = (window // 2)
        nsamples = data.size // spacing

        out = np.zeros(window)
        for i in range(nsamples - 1):
            windowed = np.hanning(window) * data[i * spacing:(i + window // spacing) * spacing]
            out += np.abs(fft.fft(windowed)) ** 2

        # TODO binning?

        freqs = out_freq

    else:
        # Hamming no overlap, like comparison paper
        nsamples = data.size // window

        for i in range(nsamples):
            windowed = np.hamming(window) * data[i * window:(i + 1) * window]
            pspec = np.abs(fft.fft(windowed)) ** 2

            # Bin data, declare accumulator output when we know its size
            if bin == "fib":
                # Modify pspec, allocate for modified form
                pspec, freqs = _fib_bin(pspec, out_freq)

                if i == 0:
                    out = np.zeros_like(np.array(pspec))
            else:
                if i == 0:
                    out = np.zeros(window)

            out += pspec

    print("PSD using ", nsamples, " segments.")
    out /= nsamples
    out_freq = freqs

    return out, out_freq


def _fib_bin(data, freqs):
    # Fibonacci binning.  Why is this a thing.
    j = 0
    fib_a = 1
    fib_b = 1
    pspec = []
    pspec_freq = []
    while j + fib_b < data.size:
        pspec.append(np.mean(data[j:j + fib_b]))
        pspec_freq.append(np.mean(freqs[j:j + fib_b]))
        j = j + fib_b
        fib_c = fib_a + fib_b
        fib_a = fib_b
        fib_b = fib_c

    return np.array(pspec), np.array(pspec_freq)


## Sums and Averages ##


def get_eht_disk_j_vals(dump, th_min=np.pi / 3., th_max=2*np.pi / 3.):
    """Calculate jmin, jmax in theta coordinate for EHT disk profiles (pi/3 - 2pi/3)"""
    # Calculate jmin, jmax for EHT radial profiles
    # Use values at large R to get even split in radially-dependent coordinates
    if len(dump['th'].shape) == 3:
        ths = dump['th'][-1, :, 0]
    elif len(dump['th'].shape) == 2:
        ths = dump['th'][-1, :]

    return (i_of(ths, th_min), i_of(ths, th_max))


def shell_sum(dump, var, at_r=None, at_zone=None, th_slice=None, j_slice=None, mask=None):
    """Sum a variable over spherical shells. Returns a radial profile (array length N1) or single-shell sum
    @param at_r: Single radius at which to sum (nearest-neighbor smaller zone is used)
    @param at_zone: Specific radial zone at which to sum, for compatibility
    @param th_slice: Tuple of minimum and maximum theta value to sum
    @param j_slice: Tuple of x2 indices instead of specifying theta
    @param mask: array of 1/0 of remaining size which is multiplied with the result
    """
    if isinstance(var, str):
        var = dump[var]

    # Translate coordinates to zone numbers.
    # TODO Xtoijk for slices?
    # TODO slice dx2, dx3 if they're matrices for exotic coordinates
    if th_slice is not None:
        j_slice = get_eht_disk_j_vals(dump, th_slice[0], th_slice[1])
    if j_slice is not None:
        var = var[:,j_slice[0]:j_slice[1],:]
        gdet = dump['gdet'][:,j_slice[0]:j_slice[1]]
    else:
        gdet = dump['gdet']

    if at_r is not None:
        at_zone = i_of(dump['r'][:,0,0], at_r)
    if at_zone is not None:
        # Keep integrand "3D" and deal with it below
        var = var[at_zone:at_zone+1]
        gdet = gdet[at_zone:at_zone+1]

    integrand = var * gdet[:, :, None] * dump.header['dx2'] * dump.header['dx3']
    if mask is not None:
        integrand *= mask

    ret = np.sum(integrand, axis=(-2, -1))
    if ret.shape == (1,):
        # Don't return a scalar result as a length-1 array
        return ret[0]
    else:
        return ret


def shell_avg(dump, var, **kwargs):
    """Average a variable over spherical shells. Returns a radial profile (array length N1) or single-shell average.
    See shell_sum for arguments.
    """
    if isinstance(var, str):
        var = dump[var]
    return shell_sum(dump, var, **kwargs) / shell_sum(dump, np.ones_like(var), **kwargs)


def sphere_sum(dump, var, r_slice=None, i_slice=None, th_slice=None, j_slice=None, mask=None):
    """Sum everything within a sphere, semi-sphere, or thick spherical shell
    Extent can be specified in r and/or theta, or i and/or j
    Mask is multiplied at the end
    """
    # TODO see sum for problems with this
    if th_slice is not None:
        j_slice = get_eht_disk_j_vals(dump, th_slice[0], th_slice[1])
    if j_slice is not None:
        var = var[:,j_slice[0]:j_slice[1],:]
        gdet = dump['gdet'][:,j_slice[0]:j_slice[1]]
    else:
        gdet = dump['gdet']

    if r_slice is not None:
        i_slice = (i_of(dump['r'][:,0,0], r_slice[0]), i_of(dump['r'][:,0,0], r_slice[1]))
    if i_slice is not None:
        var = var[i_slice[0]:i_slice[1]]
        gdet = gdet[i_slice[0]:i_slice[1]]

    return np.sum(var * gdet[:, :, None] * dump.header['dx1'] * dump.header['dx2'] * dump.header['dx3'])


def sphere_av(dump, var, **kwargs):
    if isinstance(var, str):
        var = dump[var]
    return sphere_sum(dump, var, **kwargs) / sphere_sum(dump, ones_like(var), **kwargs)


def midplane_sum(dump, var, zones=2, **kwargs):
    """Average a few zones adjacent to midplane and sum.
    Allows specifying an r_slice or i_slice within which to sum
    """
    if isinstance(var, str):
        var = dump[var]

    jmin = var.shape[1] // 2 - zones//2
    jmax = var.shape[1] // 2 + zones//2
    return sphere_sum(dump, var, j_slice=(jmin, jmax), **kwargs) / (jmax - jmin)


def theta_av(dump, var, start, zones_to_av=1, use_gdet=False, fold=True):
    """Profile in theta by averaging over phi, and optionally also:
    hemispheres: set fold=True
    radial zones: set zones_to_av > 1
    """
    if isinstance(var, str):
        var = dump[var]

    N2 = var.shape[1]
    if use_gdet:
        # TODO currently implies fold -- also does this do anything different from below?
        return (var[start:start + zones_to_av, :N2//2, :] * dump['gdet'][start:start + zones_to_av, :N2//2, None] *
                dump.header['dx1'] * dump.header['dx3'] +
                var[start:start + zones_to_av, :N2//2-1:-1, :] *
                dump['gdet'][start:start + zones_to_av, :N2//2-1:-1, None] * dump.header['dx1'] * dump.header['dx3']
                ).sum(axis=(0, 2)) \
               / ((dump['gdet'][start:start + zones_to_av, :N2 // 2] * dump.header['dx1']).sum(axis=0) * 2 * np.pi)
    else:
        if fold:
            return (var[start:start + zones_to_av, :N2 // 2, :].mean(axis=(0, 2)) +
                    var[start:start + zones_to_av, :N2 // 2 - 1:-1, :].mean(axis=(0, 2))) / 2
        else:
            return var[start:start + zones_to_av, :, :].mean(axis=(0, 2))
