

import numpy as np
import h5py

from pyHARM.defs import Loci

def write_grid(G, fname="dumps/grid.h5"):
    """Dump a file containing grid zones.
    This will primarily be of archival use soon -- see grid.py, coordinates.py for
    a good way of reconstructing all common grids on the fly.
    """
    outf = h5py.File(fname, "w")
    # Output grid in double precision
    out_type = np.float32

    x = G.coord_bulk(Loci.CENT).reshape(4, G.N[1], G.N[2], G.N[3])
    coords = G.coords

    outf['x'] = coords.cart_x(x).astype(out_type)
    outf['y'] = coords.cart_y(x).astype(out_type)
    outf['z'] = coords.cart_z(x).astype(out_type)
    outf['r'] = coords.r(x).astype(out_type)
    outf['th'] = coords.th(x).astype(out_type)
    outf['phi'] = coords.phi(x).astype(out_type)

    # Native coordinate output
    outf['X1'] = x[1].astype(out_type)
    outf['X2'] = x[2].astype(out_type)
    outf['X3'] = x[3].astype(out_type)

    # Return only the CENT values, repeated over the N3 axis
    if G.NG > 0:
        b = slice(G.NG, -G.NG)
    else:
        b = slice(None, None)
    gcon3 = G.gcon[Loci.CENT.value, :, :, b, b, None].repeat(G.NTOT[3], axis=-1).transpose((2, 3, 4, 0, 1))
    gcov3 = G.gcov[Loci.CENT.value, :, :, b, b, None].repeat(G.NTOT[3], axis=-1).transpose((2, 3, 4, 0, 1))
    gdet3 = G.gdet[Loci.CENT.value, b, b, None].repeat(G.NTOT[3], axis=-1)
    lapse3 = G.lapse[Loci.CENT.value, b, b, None].repeat(G.NTOT[3], axis=-1)

    outf['gcon'] = gcon3.astype(out_type)
    outf['gcov'] = gcov3.astype(out_type)
    outf['gdet'] = gdet3.astype(out_type)
    outf['lapse'] = lapse3.astype(out_type)

    outf.close()