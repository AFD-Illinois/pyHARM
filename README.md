# pyHARM
Python tools for HARM analysis

pyHARM is a set of Python functions for analyzing and plotting the output of General-Relativistic Magnetohydrodynamic (GRMHD) simulations, including functions for derivation of a number of different quantities (pressure, stress-energy tensor, etc) from the simulation's output, reductions & integrations in Kerr spacetime using a range of different coordinate systems, and tools for plotting simulation results in standard x,y,z coordinates. 

The primary target is simulations based on the HARM scheme (Gammie et al. 2003) -- [iharm3d](https://github.com/AFD-Illinois/iharm3d), [ebhlight](https://github.com/AFD-Illinois/ebhlight), [KHARMA](https://github.com/AFD-Illinois/kharma), and others.  It includes Python re-implementations of most of the scheme, useful for deriving everything about the simulation not directly present in output files.  Output from certain other codes can be converted to a readable format with [EHT-babel](https://github.com/AFD-Illinois/EHT-babel/).

As a consequence of needing to read GRMHD output, pyHARM includes definitions of various coordinate systems in Kerr spacetime, as well as tools for dealing with a logically Cartesian grid in various coordinate systems.  These might be independently useful, see `coordinates.py` & `grid.py` if you're looking for just coordinate tools.

## Installing:
The preferred installation method, for flexibility in changing the source as needed, is to run simply:
```bash
$ python3 setup.py develop
```
Thereafter pyHARM should be importable from any Python prompt or script run in the same environment.  It can also be installed as a user or system package with `pip`.


There is also an included Anaconda environment for users who would prefer Anaconda versions of the dependencies -- also note that any future (optional!) re-introduction of OpenCL integration will require the Anaconda environment.

```bash
$ conda env create -f environment.yml
$ conda activate pyHARM
$ python3 setup.py develop
```

## Examples:
The `notebooks` directory has a sample Jupyter notebook playing around with some basic reductions & plots.

The scripts in the `scripts/` directory are another good place to check -- they don't always reflect the easiest way to use the library as the interface changes, but they do generally work.
They are also useful in themselves -- `quick_plot.py` and `movie.py` are good for exploring output, and `analysis.py` provides a pipeline for arbitrary reductions over full GRMHD runs, which is in turn assumed by the `pyHARM.ana.results` module.
