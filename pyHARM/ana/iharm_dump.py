# Object representing an iharm dump file
# Contains fluid state, and definitions of some common derived fields

import os
import sys
import numpy as np

from pyHARM.defs import Loci
from pyHARM.grmhd import phys
from pyHARM.grid import Grid

import pyHARM.io as io

import pyHARM.ana.variables as vars
from pyHARM.checks import divB

class IharmDump:
    """Read and cache data from a fluid dump file in HARM HDF5 format, and allow accessing
    various derived variables directly.
    """

    def __init__(self, fname, params=None, calc_cons=False, calc_derived=False, calc_divB=False,
                 add_jcon=False, add_floors=False, add_fails=False, add_ghosts=False, add_divB=False,
                 add_psi_cd=False, add_grid_caches=True, tag="", zones_first=False):
        """Read the HDF5 file 'fname' into memory, and pre-calculate/cache useful variables
        @param calc_cons: calculate the conserved variables U, i.e. run 'prim_to_flux(...,0)' from HARM
        @param calc_derived: calculate the derived 4-vectors u, b and fluid Lorentz factor gamma
        @param add_jcon: Read the current jcon from the file if it exists, fail if it doesn't

        For debugging:
        @param add_floors: Read the applied floors bitflag from the file if it exists, fail if it doesn't
        @param add_fails: Read the inversion failures bitflag from the file if it exists, fail if it doesn't
        @param add_divB: Read the B-field divergence *present in the file*, divB.  pyHARM can still *calculate* the
        divergence for itself if this parameter is set to False, so this is mostly useful for debugging.

        @param zones_first: keep arrays and vectors in i,j,k,p order rather than native p,i,j,k.  This breaks
        most pyHARM functions, so only use it if you plan to do most manipulations yourself.  Even then,
        consider the more flexible functions in io.* instead.
        """
        self.fname = fname
        self.tag = tag
        if params is None:
            params = {}

        # Choose an importer based on what we know of filenames
        # TODO option to add U from dumps
        my_filter = io.get_filter(fname)
        P, params = my_filter.read_dump(fname, add_ghosts=add_ghosts, params=params)
        if add_jcon:
            self.jcon = my_filter.read_jcon(fname, add_ghosts=add_ghosts)
        if add_fails:
            self.fails = my_filter.read_fail_flags(fname, add_ghosts=add_ghosts)
        if add_floors:
            self.floors = my_filter.read_floor_flags(fname, add_ghosts=add_ghosts)
        if add_divB:
            self.divB = my_filter.read_divb(fname, add_ghosts=add_ghosts)
        if add_psi_cd:
            self.psi_cd = my_filter.read_psi_cd(fname, add_ghosts=add_ghosts)

        self.header = self.params = params

        if not add_ghosts:
            self.header['ng'] = 0

        self.grid = G = Grid(self.header, caches=add_grid_caches)
        if add_grid_caches:
            if zones_first:
                self.gcon = G.gcon[Loci.CENT.value].transpose(2, 3, 0, 1)
                self.gcov = G.gcov[Loci.CENT.value].transpose(2, 3, 0, 1)
            else:
                self.gcon = G.gcon[Loci.CENT.value]
                self.gcov = G.gcov[Loci.CENT.value]
            self.gdet = G.gdet[Loci.CENT.value]
            self.lapse = G.lapse[Loci.CENT.value]

        self.N1 = G.NTOT[1]
        self.N2 = G.NTOT[2]
        self.N3 = G.NTOT[3]

        if calc_divB:
            self.divB = divB(G, P)

        if calc_derived or calc_cons:
            # Derived variables
            # Calculating w/temp P avoids issues with the variable reordering
            D = phys.get_state(self.header, G, P)
            self.gamma = phys.mhd_gamma_calc(G, P)
            if calc_cons:
                U = phys.prim_to_flux(self.header, self.grid, P, D=D)

            self.ucon = D['ucon']
            self.ucov = D['ucov']
            self.bcon = D['bcon']
            self.bcov = D['bcov']

            del D

        self._zones_first = zones_first
        if zones_first and (calc_derived or calc_cons):
            # Turn around all our lovely zones-last arrays
            self.prims = np.ascontiguousarray(np.einsum("p...->...p", P))
            if calc_cons:
                self.cons = np.einsum("p...->...p", U)
            if calc_derived or calc_cons:
                self.ucon = np.einsum("p...->...p", self.ucon)
                self.ucov = np.einsum("p...->...p", self.ucov)
                self.bcon = np.einsum("p...->...p", self.bcon)
                self.bcov = np.einsum("p...->...p", self.bcov)
        else:
            self.prims = np.ascontiguousarray(P)
            if calc_cons:
                self.cons = U
        # We no longer need the originals
        del P,G
        if calc_cons:
            del U

    # Act like a dict when retrieving lots of different things --
    # just compute/retrieve them on the fly!
    def __getitem__(self, key):
        if key in self.header['prim_names']:
            i = self.header['prim_names'].index(key)
            if self._zones_first:
                return self.prims[:, :, :, i]
            else:
                return self.prims[i]
        elif key in self.__dict__:
            return self.__dict__[key]
        elif key in ['r', 'th', 'phi']:
            return getattr(self.grid.coords, key)(self.grid.coord_all())
        elif key in ['x', 'y', 'z']:
            return getattr(self.grid.coords, 'cart_' + key)(self.grid.coord_all())
        elif key in ['X1', 'X2', 'X3']:
            return self.grid.coord_all()[int(key[-1:])]
        elif key in self.header:
            return self.header[key]
        elif key in vars.fns_dict:
            return vars.fns_dict[key](self)
        elif key[:4] == "log_":
            return np.log10(self[key[4:]])
        elif key[:3] == "ln_":
            return np.log(self[key[3:]])
        elif key[:4] == "pdf_":
            var_og = self[key[4:]]
            pdf_window=(np.min(var_og), np.max(var_og))
            return np.histogram(var_og, bins=100, range=pdf_window,
                weights=np.repeat(self.gdet, self.N3).reshape(var_og.shape), density=True)
        # TODO transformed full vectors, with e.g. 'ucon_ks'
        # Return vector components
        elif key[-2:] == "_0" or key[-2:] == "_1" or key[-2:] == "_2" or key[-2:] == "_3":
            return self[key[0]+"cov"][int(key[-1])]
        elif key[-2:] == "^0" or key[-2:] == "^1" or key[-2:] == "^2" or key[-2:] == "^3":
            return self[key[0]+"con"][int(key[-1])]
        # Return transformed vector components
        elif key[-2:] == "_t" or key[-2:] == "_r" or key[-3:] == "_th" or key[-4:] == "_phi":
            return np.einsum("i...,ij...->j...",
                                self[key[0]+"cov"],
                                self.grid.coords.dxdX(self.grid.coord_all())
                            )[["t", "r", "th", "phi"].index(key.split("_")[-1])]
        elif key[-2:] == "^t" or key[-2:] == "^r" or key[-3:] == "^th" or key[-4:] == "^phi":
            return np.einsum("i...,ij...->j...",
                                self[key[0]+"con"],
                                self.grid.coords.dXdx(self.grid.coord_all())
                            )[["t", "r", "th", "phi"].index(key.split("^")[-1])]
        elif key[-2:] == "_x" or key[-2:] == "_y" or key[-2:] == "_z":
            return np.einsum("i...,ij...->j...",
                                self[key[0]+"cov"],
                                np.einsum("ij...,jk...->ik...",
                                    self.grid.coords.dxdX(self.grid.coord_all()),
                                    self.grid.coords.dxdX(self.grid.coord_all())
                                )
                            )[["t", "r", "th", "phi"].index(key.split("_")[-1])]
        elif key[-2:] == "^x" or key[-2:] == "^y" or key[-2:] == "^z":
            return np.einsum("i...,ij...->j...",
                                self[key[0]+"con"],
                                np.einsum("ij...,jk...->ik...",
                                    self.grid.coords.dxdX(self.grid.coord_all()),
                                    self.grid.coords.dxdX(self.grid.coord_all())
                                )
                            )[["t", "r", "th", "phi"].index(key.split("^")[-1])]
        else:
            try:
                # Reshape number inputs.  I swear this is useful for properly-sized constant arrays for e.g. area
                nkey = float(key)
                return nkey*np.ones_like(self['RHO'])
            except ValueError:
                raise ValueError("IharmDump cannot find or compute {}".format(key))

    # TODO does __del__ need to do anything?


