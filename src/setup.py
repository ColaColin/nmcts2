from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(["core/cvectorMcts.pyx", "connect6/c6.pyx", "core/cMctsTree.pyx", "core/AbstractTorchLearner.pyx"], annotate=True, compiler_directives={
        'boundscheck': True,
        'wraparound': True,
        'cdivision': True,
        'nonecheck': True
    }),
    include_dirs=[numpy.get_include()]
)
