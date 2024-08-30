# This file is used to build the Cython extension module.
# Run it with the command python setup.py build_ext --inplace

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("has_winning_path", ["has_winning_path.pyx"],
              include_dirs=[numpy.get_include()]),
]

setup(
    name="My Project",
    ext_modules=cythonize(extensions),
)