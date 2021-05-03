.. module:: pyHARM
.. moduleauthor:: Ben Prather <bprathr2@illinois.edu>

.. _ref_dumps:

Reading HARM HDF5 Output
========================

These are functions for reading files of the form documented on the AFD docs `wiki <https://github.com/AFD-Illinois/docs/wiki/Output-Format>`_.  This is the format used by modern iterations of ``iharm3d`` and ``ebhlight``, as well as the HARM implementation in this package.  Several tools for converting output from other codes into this format can be found at `EHT-babel <https://github.com/AFD-Illinois/EHT-babel>`_.

There are two ways to read HARM HDF5 output with the pyHARM tools: the high level interface, :mod:`pyHARM.ana.iharm_dump`, which allows accessing derived variables of several types in a ``dict``-like interface, or :mod:`pyHARM.io`, which reads and writes arrays of the primitive variables used by HARM.

IharmDump interface
-------------------

.. automodule:: pyHARM.ana.iharm_dump
   :members:

h5io interface
--------------

.. automodule:: pyHARM.io
   :members: