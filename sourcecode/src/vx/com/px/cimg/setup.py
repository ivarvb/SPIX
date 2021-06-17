#https://medium.com/@xpl/protecting-python-sources-using-cython-dcd940bb188e
import setuptools  # important
from distutils.core import Extension, setup
from Cython.Build import cythonize

from Cython.Distutils import build_ext

#cimport numpy as np
import numpy


libraries = ["pthread", "X11"]

# define an extension that will be cythonized and compiled
extensions = [
        Extension(
            name="cimg",
            sources=["cimg.pyx"],
#            libraries=[],
            #library_dirs=["/usr/local/lib/","/usr/lib"],
            libraries = libraries,

            language="c++",
            include_dirs=[numpy.get_include()]
            ),
        
    ]
setup(
    name = 'CIMG',
    cmdclass = {'build_ext': build_ext},
    ext_modules=cythonize(extensions)
)
