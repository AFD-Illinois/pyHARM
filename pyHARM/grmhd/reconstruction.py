# OpenCL version of reconstruction as a test
# This is written with heavy reliance on a Python package called loopy (formerly loo.py) for
# doing code transformations

# it's effectively an interactive compiler -- you can make substitutions, loop transformations/vectorizations,
# and optimization passes, all as functions of the form
# knl = lp.do_x(knl, options)

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

from pyHARM.loopy_tools import primsArrayArgs, add_ghosts

from datetime import datetime

knl1, knl2, knl3 = None, None, None
body_weno5 = """
    vr0 := (3. / 8.) * x1 - (5. / 4.) * x2 + (15. / 8.) * x3
    vr1 := (-1. / 8.) * x2 + (3. / 4.) * x3 + (3. / 8.) * x4
    vr2 := (3. / 8.) * x3 + (3. / 4.) * x4 - (1. / 8.) * x5

    vl0 := (3. / 8.) * x5 - (5. / 4.) * x4 + (15. / 8.) * x3
    vl1 := (-1. / 8.) * x4 + (3. / 4.) * x3 + (3. / 8.) * x2
    vl2 := (3. / 8.) * x3 + (3. / 4.) * x2 - (1. / 8.) * x1

    beta0 := (13. / 12.) * (x1 - 2. * x2 + x3) * (x1 - 2. * x2 + x3) + \
                (1. / 4.) * (x1 - 4. * x2 + 3. * x3) * (x1 - 4. * x2 + 3. * x3)
    beta1 := (13. / 12.) * (x2 - 2. * x3 + x4) * (x2 - 2. * x3 + x4) + \
                (1. / 4.) * (x4 - x2) * (x4 - x2)
    beta2 := (13. / 12.) * (x3 - 2. * x4 + x5) * (x3 - 2. * x4 + x5) + \
                (1. / 4.) * (x5 - 4. * x4 + 3. * x3) * (x5 - 4. * x4 + 3. * x3)

    wtr0 := (1. / 16.) / ((eps + beta0) * (eps + beta0))
    wtr1 := (5. / 8.) / ((eps + beta1) * (eps + beta1))
    wtr2 := (5. / 16.) / ((eps + beta2) * (eps + beta2))

    Wr := wtr0 + wtr1 + wtr2

    wr0 := wtr0 / Wr
    wr1 := wtr1 / Wr
    wr2 := wtr2 / Wr

    wtl0 := (1. / 16.) / ((eps + beta2) * (eps + beta2))
    wtl1 := (5. / 8.) / ((eps + beta1) * (eps + beta1))
    wtl2 := (5. / 16.) / ((eps + beta0) * (eps + beta0))

    Wl := wtl0 + wtl1 + wtl2

    wl0 := wtl0 / Wl
    wl1 := wtl1 / Wl
    wl2 := wtl2 / Wl

    lout = vl0 * wl0 + vl1 * wl1 + vl2 * wl2
    rout = vr0 * wr0 + vr1 * wr1 + vr2 * wr2
    """

def compile_recon_kernel(params, G, dir, subs):
    sh = G.shapes
    body_local = body_weno5
    for sub in subs:
        body_local = body_local.replace(sub[0], sub[1])
    body_local = add_ghosts(body_local)

    knl = lp.make_kernel(sh.isl_grid_primitives, body_local, [*primsArrayArgs("P", "lout", "rout"), ...],
                         assumptions=sh.assume_grid_primitives, silenced_warnings='inferred_iname')

    # This is the template for general optimization but will get more complicated for the patterns
    # in different directions.
    knl = lp.fix_parameters(knl, eps=1.e-26, nprim=params['n_prim'])
    knl = lp.fix_parameters(knl, n1=int(G.N[1]+2), n2=int(G.N[2]+2), n3=int(G.N[3]+2), ng=G.NG-1)
    knl = lp.split_iname(knl, "k", 64, outer_tag="g.0", inner_tag="l.0")
    knl = lp.split_iname(knl, "j", 1, outer_tag="g.1", inner_tag="unr")
    knl = lp.split_iname(knl, "i", 1, outer_tag="g.2", inner_tag="unr")

    # knl = lp.add_prefetch(knl, "P", ["k_inner"], fetch_bounding_box=True, temporary_name="_P")
    # knl = lp.tag_inames(knl, "P_dim_0:l.0")

    print("Compiled reconstruction {}".format(dir))

    return knl


def reconstruct(params, G, P, dir, rout=None, lout=None):
    sh = G.shapes

    global knl1, knl2, knl3
    if knl1 is None:
        # CACHE EVERYTHING
        subs = [["x1", "P[p,i-2,j,k]"],
                ["x2", "P[p,i-1,j,k]"],
                ["x3", "P[p,i,j,k]"],
                ["x4", "P[p,i+1,j,k]"],
                ["x5", "P[p,i+2,j,k]"],
                ["lout", "lout[p,i,j,k]"],
                ["rout", "rout[p,i+1,j,k]"]]
        knl1 = compile_recon_kernel(params, G, 1, subs)
        subs = [["x1", "P[p,i,j-2,k]"],
                ["x2", "P[p,i,j-1,k]"],
                ["x3", "P[p,i,j,k]"],
                ["x4", "P[p,i,j+1,k]"],
                ["x5", "P[p,i,j+2,k]"],
                ["lout", "lout[p,i,j,k]"],
                ["rout", "rout[p,i,j+1,k]"]]
        knl2 = compile_recon_kernel(params, G, 2, subs)
        subs = [["x1", "P[p,i,j,k-2]"],
                ["x2", "P[p,i,j,k-1]"],
                ["x3", "P[p,i,j,k]"],
                ["x4", "P[p,i,j,k+1]"],
                ["x5", "P[p,i,j,k+2]"],
                ["lout", "lout[p,i,j,k]"],
                ["rout", "rout[p,i,j,k+1]"]]
        knl3 = compile_recon_kernel(params, G, 3, subs)

    if lout is None:
        lout = cl_array.empty(params['queue'], sh.grid_primitives, dtype=np.float64)
    if rout is None:
        rout = cl_array.empty(params['queue'], sh.grid_primitives, dtype=np.float64)

    if dir == 1:
        evt, _ = knl1(params['queue'], P=P, lout=lout, rout=rout)
    elif dir == 2:
        evt, _ = knl2(params['queue'], P=P, lout=lout, rout=rout)
    elif dir == 3:
        evt, _ = knl3(params['queue'], P=P, lout=lout, rout=rout)

    return lout, rout
