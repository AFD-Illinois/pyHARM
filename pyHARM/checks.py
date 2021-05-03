# Various debugging tools: mapping areas of the grid, checks for NaNs, etc.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pyHARM.defs import Loci, Slices

def check_nan_prims(params, P):
    # Print the surrounding areas of the first 10 NaNs found
    # TODO allow to specify one prim to check, slice to check
    nidxs = np.argwhere(np.isnan(P))
    if nidxs.shape[0] > 10:
        nidxs = nidxs[:10]
    for idx in nidxs:
        print_area_prims(params, P, idx)


def check_nan_array(arr, slc=None):
    if slc is not None:
        nidxs = np.argwhere(np.isnan(arr))
    else:
        nidxs = np.argwhere(np.isnan(arr[slc]))
    print(nidxs)
    if nidxs.shape[0] > 10:
        nidxs = nidxs[:10]
    for idx in nidxs:
        print_area_array(arr, idx)


def print_area_prims(params, P, idx):
    # Print surrounding zones of a location idx in form [prim_number, i, j, k]
    dbg_slice = P[idx[0], idx[1]-1:idx[1]+1, idx[2]-1:idx[2]+1, idx[3]-1:idx[3]+1]
    print(params['var_names'][idx[0]], ":", dbg_slice)


def print_area_array(arr, idx):
    dbg_slice = arr[idx[0]-1:idx[0]+1, idx[1]-1:idx[1]+1, idx[2]-1:idx[2]+1]
    print(dbg_slice)


def shape_of_nonzero_portion(arr, tag=""):
    for i, idx_list in enumerate(np.nonzero(arr)):
        print("{} axis {} {}:{}".format(tag, i, np.min(idx_list), np.max(idx_list)))


def divB(G, P):
    gdet = G.gdet[Loci.CENT.value]

    # If we don't have ghost zones, make our own slices
    if G.NG > 0:
        s = G.slices
    else:
        s = Slices(ng=1)

    divB = np.abs(0.25 * (
            P[s.B1][s.b, s.b, s.b] * gdet[s.b, s.b, None]
            + P[s.B1][s.b, s.l1, s.b] * gdet[s.b, s.l1, None]
            + P[s.B1][s.b, s.b, s.l1] * gdet[s.b, s.b, None]
            + P[s.B1][s.b, s.l1, s.l1] * gdet[s.b, s.l1, None]
            - P[s.B1][s.l1, s.b, s.b] * gdet[s.l1, s.b, None]
            - P[s.B1][s.l1, s.l1, s.b] * gdet[s.l1, s.l1, None]
            - P[s.B1][s.l1, s.b, s.l1] * gdet[s.l1, s.b, None]
            - P[s.B1][s.l1, s.l1, s.l1] * gdet[s.l1, s.l1, None]
            ) / G.dx[1] + 0.25 * (
            P[s.B2][s.b, s.b, s.b] * gdet[s.b, s.b, None]
            + P[s.B2][s.l1, s.b, s.b] * gdet[s.l1, s.b, None]
            + P[s.B2][s.b, s.b, s.l1] * gdet[s.b, s.b, None]
            + P[s.B2][s.l1, s.b, s.l1] * gdet[s.l1, s.b, None]
            - P[s.B2][s.b, s.l1, s.b] * gdet[s.b, s.l1, None]
            - P[s.B2][s.l1, s.l1, s.b] * gdet[s.l1, s.l1, None]
            - P[s.B2][s.b, s.l1, s.l1] * gdet[s.b, s.l1, None]
            - P[s.B2][s.l1, s.l1, s.l1] * gdet[s.l1, s.l1, None]
            ) / G.dx[2] + 0.25 * (
            P[s.B3][s.b, s.b, s.b] * gdet[s.b, s.b, None]
            + P[s.B3][s.b, s.l1, s.b] * gdet[s.b, s.l1, None]
            + P[s.B3][s.l1, s.b, s.b] * gdet[s.l1, s.b, None]
            + P[s.B3][s.l1, s.l1, s.b] * gdet[s.l1, s.l1, None]
            - P[s.B3][s.b, s.b, s.l1] * gdet[s.b, s.b, None]
            - P[s.B3][s.b, s.l1, s.l1] * gdet[s.b, s.l1, None]
            - P[s.B3][s.l1, s.b, s.l1] * gdet[s.l1, s.b, None]
            - P[s.B3][s.l1, s.l1, s.l1] * gdet[s.l1, s.l1, None]
            ) / G.dx[3])
    if G.NG > 0:
        return divB
    else:
        divB_full = np.zeros(G.shapes.grid_scalar)
        divB_full[1:-1, 1:-1, 1:-1] += divB
        return divB_full
