# For some reason, I need to import torch before cppyy
# This did not use to be required, but has solved massive crashes lately
import torch as _

# We must specify the "-fopenmp" flag
# I also turn on O3 optimization (at the cost of compilation speed and bytecode size)
# O3 enables various loop optimizations (unrolling, splitting, peeling, ...)
import os
os.environ['EXTRA_CLING_ARGS'] = '-fopenmp=libomp -O3'
