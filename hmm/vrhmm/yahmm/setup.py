from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "yahmm",
        ["yahmm.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
    )
]

setup(
    name="yahmm",
    ext_modules=cythonize(extensions),
)