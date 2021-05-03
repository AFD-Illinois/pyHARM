# Module defining coordinate grids

import numpy as np

try:
    import pyopencl.array as cl_array
    import loopy as lp

    from pyHARM.loopy_tools import *
    use_2018_2()
except ModuleNotFoundError:
    #print("Loading grid.py without OpenCL array support.")
    pass

from pyHARM.defs import Loci, Slices, Shapes
from pyHARM.coordinates import *


def make_some_grid(type, n1=128, n2=128, n3=128, a=0, hslope=0.3, r_in=None, r_out=50, params=None):
    """Convenience function for generating grids with default parameters used at Illinois.
    Type should be one of 'minkowski', 'mks', 'fmks'
    Size and coordinate parameters are optional with somewhat reasonable defaults.
    """
    if params is None:
        params = {}

    params['coordinates'] = type
    params['n1tot'] = n1
    params['n2tot'] = n2
    params['n3tot'] = n3
    params['n1'] = n1
    params['n2'] = n2
    params['n3'] = n3

    # Things which should ideally be optional in grid creation,
    # but are not for one reason or another
    params['ng'] = 0
    params['n_prim'] = 8

    if type == 'minkowski':
        params['x1min'] = 0
        params['x2min'] = 0
        params['x3min'] = 0
        params['x1max'] = 1
        params['x2max'] = 1
        params['x3max'] = 1
    elif 'mks' in type:
        params['a'] = a
        params['hslope'] = hslope
        params['r_out'] = r_out
        if r_in is not None:
            params['r_in'] = r_in
        if type == 'fmks' or type == 'mmks':
            params['poly_xt'] = 0.82
            params['poly_alpha'] = 14.0
            params['mks_smooth'] = 0.5

    return Grid(params)



class Grid:
    """Holds all information about the (potentially a) grid of zones:
    size, shape, zones' global locations, metric tensor
    """

    def __init__(self, params, caches=True):
        """
        Initialize a Grid object.  This object divides a domain in native coordinates into zones, and caches the
        local metric (and some other convenient information) at several locations in each zone.

        Primarily, this object should be used to consult the grid size/shape for global calculations, and raise and
        lower the indices of fluid 4-vectors.

        :param params: Dictionary containing the following parameters, depending on coordinate system:
        n{1,2,3}tot: total number of physical grid zones in each direction
        ng: number of "ghost" zones in each direction to add for boundary conditions/domain decomposition
        coordinates: name of the coordinate system (value in parens below)
        === For Minkowski coordinates ("minkowski"): ===
        x{1,2,3}min: location of nearest corner of first grid zone [0,0,0]
        x{1,2,3}max: location of farthest corner of last grid zone [n1tot, n2tot, n3tot]
        === For Spherical Kerr-Schild coordinates ("ks"): ===
        a: black hole spin
        r_out: desired location of the outer edge of the farthest radial zone on the grid
        === For Modified Kerr-Schild coordinates ("mks"): ===
        hslope: narrowing parameter for X2, defined in Gammie et. al. '03 `here <https://doi.org/10.1086/374594>`_
        === For Modified Modified Kerr-Schild coordinates ("mmks"): ===
        All of the MKS parameters, plus
        poly_xt, poly_alpha: See HARM docs `wiki <https://github.com/AFD-Illinois/docs/wiki/Coordinates>`_
        === For "Funky" Modified Kerr-Schild coordinates ("fmks"): ===
        All of the MMKS parameters, plus
        mks_smooth: See HARM docs wiki
        """
        # Set the basic grid parameters
        # Total grid size (all MPI processes)
        if 'n1tot' in params:
            self.NTOT = np.array([1, params['n1tot'], params['n2tot'], params['n3tot']])
        else:
            self.NTOT = np.array([1, params['n1'], params['n2'], params['n3']])
        # Indices of first & last zone in global grid
        self.global_start = np.array([0, 0, 0])  # TODO someday this and N need modification
        self.global_stop = self.NTOT[1:]
        # Size of this grid
        self.N = self.NTOT  # TODO do I really want to carry around a useless first index?
        # Number of ghost zones
        if 'ng' in params:
            self.NG = params['ng']
        else:
            self.NG = 0
        # Size of grid in-memory (with ghost zones where appropriate)
        self.GN = self.N + (self.N > 1) * 2*self.NG

        # Slices and shapes sub-objects: to hold numbers in convenient namespaces
        self.slices = Slices(self.NG)
        self.shapes = Shapes(self, params)

        if params['coordinates'] in ["minkowski", "cartesian"]:
            # There are no parameters to Minkowski coordinates, so this is a class, not an object
            self.coords = Minkowski
        # MKS, FMKS expect the parameters a, hslope
        elif params['coordinates'] == "fmks":
            # FMKS additionally requires poly_xt, poly_alpha, mks_smooth
            self.coords = FMKS(params)
        elif params['coordinates'] == "mmks" or params['coordinates'] == "cmks":
            # MMKS additionally requires poly_xt and poly_alpha
            self.coords = CMKS(params)
        elif params['coordinates'] == "bhac_mks":
            # BHAC's MKS
            self.coords = BHAC_MKS(params)
        elif params['coordinates'] == "mks":
            self.coords = MKS(params)
        else:
            raise ValueError("metric is {}!! must be minkowski, mks, mmks, or fmks".format(params['coordinates']))

        # If we got native coordinates, use those
        if 'x1min' in params:
            self.startx = np.array([0, params['x1min'], params['x2min'], params['x3min']])
        else:
            # Ask our new coordinate system where to start/stop the native grid,
            # so it aligns with the KS boundaries we've been assigned
            self.startx = self.coords.native_startx(params)
            if params['coordinates'] not in ["minkowski", "cartesian"] and self.startx[1] < 0.0:
                raise ValueError("Not enough radial zones! Increase N1!")

        self.stopx = self.coords.native_stopx(params)

        # Finally, set up the grid
        self.dx = (self.stopx - self.startx) / self.NTOT
        self.dV = self.dx[1]*self.dx[2]*self.dx[3]

        if caches:
            self.gcov = np.zeros(self.shapes.locus_geom_tensor)
            self.gcon = np.zeros_like(self.gcov)

            self.gdet = np.zeros(self.shapes.locus_geom_scalar)
            self.lapse = np.zeros_like(self.gdet)

            for loc in Loci:
                ilist = np.arange(self.GN[1])
                jlist = np.arange(self.GN[2])
                x = self.coord(ilist, jlist, 0, loc)

                # Save zone centers to calculate connection coefficients
                if loc == Loci.CENT:
                    x_cent = x

                gcov_loc = self.coords.gcov(x)
                gcon_loc = self.coords.gcon(gcov_loc)
                gdet_loc = self.coords.gdet(gcov_loc)
                self.gcov[loc.value] = gcov_loc
                self.gcon[loc.value] = gcon_loc
                self.gdet[loc.value] = gdet_loc
                self.lapse[loc.value] = 1./np.sqrt(-gcon_loc[0, 0])

            self.conn = self.coords.conn_func(x_cent)

    def coord(self, i, j, k, loc=Loci.CENT):
        """Get the position x of zone(s) i,j,k, in _native_ coordinates

        If given lists of i,j,k, this returns x[NDIM,len(i),len(j),len(k)] via np.meshgrid().
        Any index given as a single value is suppressed on output, down to x[NDIM]

        All functions in coordinates.py which take coordinates "x" also accept a grid of the form this returns.
        """
        i += self.global_start[0]
        j += self.global_start[1]
        k += self.global_start[2]

        # Create list of 3 lists of coordinates
        x = [np.zeros(1)]
        for index in i, j, k:
            if isinstance(index, np.ndarray) and len(index.shape) == 1:
                x.append(np.zeros_like(index))
            else:
                x.append(np.zeros(1))

        for mu in range(4):
            x[mu] = self.startx[mu]

        NG = self.NG
        dx = self.dx
        if loc == Loci.FACE1:
            x[1] += (i - NG) * dx[1]
            x[2] += (j + 0.5 - NG) * dx[2]
            x[3] += (k + 0.5 - NG) * dx[3]
        elif loc == Loci.FACE2:
            x[1] += (i + 0.5 - NG) * dx[1]
            x[2] += (j - NG) * dx[2]
            x[3] += (k + 0.5 - NG) * dx[3]
        elif loc == Loci.FACE3:
            x[1] += (i + 0.5 - NG) * dx[1]
            x[2] += (j + 0.5 - NG) * dx[2]
            x[3] += (k - NG) * dx[3]
        elif loc == Loci.CENT:
            x[1] += (i + 0.5 - NG) * dx[1]
            x[2] += (j + 0.5 - NG) * dx[2]
            x[3] += (k + 0.5 - NG) * dx[3]
        elif loc == Loci.CORN:
            x[1] += (i - NG) * dx[1]
            x[2] += (j - NG) * dx[2]
            x[3] += (k - NG) * dx[3]
        else:
            raise ValueError("Invalid coordinate location!")

        return np.squeeze(np.array(np.meshgrid(x[0], x[1], x[2], x[3])))

    def coord_bulk(self, loc=Loci.CENT):
        """Return a 3D array of all position vectors X within the physical zones.
        See coord() for use.
        """
        return self.coord(np.arange(self.N[1])+self.NG,
                          np.arange(self.N[2])+self.NG,
                          np.arange(self.N[3])+self.NG, loc=loc)

    def coord_bulk_mesh(self):
        """Returns zone corners for plotting a variable in the bulk
        """
        return self.coord(np.arange(self.N[1]+1)+self.NG,
                          np.arange(self.N[2]+1)+self.NG,
                          np.arange(self.N[3]+1)+self.NG, loc=Loci.CORN)

    def coord_all(self, loc=Loci.CENT):
        """Like coord_bulk, but including ghost zones"""
        return self.coord(np.arange(self.GN[1]),
                          np.arange(self.GN[2]),
                          np.arange(self.GN[3]), loc=loc)

    def coord_all_mesh(self):
        """Like coord_bulk_mesh, but including ghost zones"""
        return self.coord(np.arange(self.GN[1]+1),
                          np.arange(self.GN[2]+1),
                          np.arange(self.GN[3]+1), loc=Loci.CORN)

    def lower_grid(self, vcon, loc=Loci.CENT):
        """Lower a grid of contravariant rank-1 tensors to covariant ones."""
        return np.einsum("ij...,j...->i...", self.gcov[loc.value, :, :, :, :, None], vcon)

    def raise_grid(self, vcov, loc=Loci.CENT):
        """Raise a grid of covariant rank-1 tensors to contravariant ones."""
        return np.einsum("ij...,j...->i...", self.gcon[loc.value, :, :, :, :, None], vcov)

    def dot(self, ucon, ucov):
        """Inner product along first index. Exists to make other code OpenCL-agnostic"""
        return np.einsum("i...,i...", ucon, ucov)


    def dt_light(self):
        """Returns the light crossing time of the smallest zone in the grid"""
        # Following stolen from bhlight's dt_light calculation

        dt_light_local = np.zeros((self.N[1], self.N[2]), dtype=np.float64)
        gcon = self.gcon
        CENT = Loci.CENT.value
        for mu in range(1,4):
            cplus = np.abs((-gcon[CENT, 0, mu, :, :] +
                            np.sqrt(gcon[CENT, 0, mu, :, :]**2 -
                                    gcon[CENT, mu, mu, :, :] * gcon[CENT, 0, 0, :, :])) /
                           gcon[CENT, 0, 0, :, :])

            cminus = np.abs((-gcon[CENT, 0, mu, :, :] -
                             np.sqrt(gcon[CENT, 0, mu, :, :]**2 -
                                     gcon[CENT, mu, mu, :, :] * gcon[CENT, 0, 0, :, :])) /
                            gcon[CENT, 0, 0, :, :])
            light_phase_speed = np.maximum.reduce([cplus, cminus])

            light_phase_speed = np.where(gcon[CENT, 0, mu, :, :] ** 2 -
                                         gcon[CENT, mu, mu, :, :] * gcon[CENT, 0, 0, :, :] >= 0.,
                                         light_phase_speed, 1e-20)

            dt_light_local += 1. / (self.dx[mu] / light_phase_speed)

        dt_light_local = 1. / dt_light_local

        return np.min(dt_light_local)
    
    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        elif key in ['n1', 'n2', 'n3']:
            return self.NTOT[int(key[-1:])]
            # TODO tot
        elif key in ['r', 'th', 'phi']:
            return getattr(self.coords, key)(self.coord_all())
        elif key in ['x', 'y', 'z']:
            return getattr(self.coords, 'cart_' + key)(self.coord_all())
        elif key in ['X1', 'X2', 'X3']:
            return self.coord_all()[int(key[-1:])]
        else:
            raise ValueError("Grid cannot find or compute {}".format(key))
