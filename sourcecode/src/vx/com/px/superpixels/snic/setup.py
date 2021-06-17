#https://medium.com/@xpl/protecting-python-sources-using-cython-dcd940bb188e
import setuptools  # important
from distutils.core import Extension, setup
from Cython.Build import cythonize

from Cython.Distutils import build_ext

#cimport numpy as np
import numpy



# define an extension that will be cythonized and compiled
extensions = [
        Extension(
            name="snic",
            sources=["SNIC.pyx"],
            libraries = ["pthread", "X11"],
            library_dirs=["/usr/X11/lib"],
            language="c++",
            include_dirs=[numpy.get_include()]
            ),
        
    ]
setup(
    name = 'SNIC',
    cmdclass = {'build_ext': build_ext},
    ext_modules=cythonize(extensions)
)
