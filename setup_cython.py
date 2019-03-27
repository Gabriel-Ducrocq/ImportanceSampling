from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name="log_weight_computing", ext_modules=cythonize('log_weight_computing.pyx'), include_dirs=[numpy.get_include()])