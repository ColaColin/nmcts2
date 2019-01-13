from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(["core/cvectorMcts.pyx", "connect6/c6.pyx", "core/AbstractTorchLearner.pyx"], annotate=True, compiler_directives={
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
        'nonecheck': False
    }),
    include_dirs=[numpy.get_include()]
)
