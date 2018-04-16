from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(["connect6/c6.pyx"], annotate=True, compiler_directives={
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
        'nonecheck': False
    }),

)
