# Load a selection of useful functions and aliases to serve as a high-level interface

from .ana.iharm_dump import IharmDump
from .ana.reductions import *
from .ana.variables import pretty

from .ana import plot

from .defs import Loci, Var

def load_dump(fname, **kwargs):
    return IharmDump(fname, **kwargs)