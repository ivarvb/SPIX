# distutils: language = c

import os
from typing import List
import numpy as np
cimport numpy as np
#cimport cython


import matplotlib.pyplot as plt
#from libcpp.vector cimport vector
#from libcpp.string cimport string

#from libcpp.vector cimport vector

from libc.stdlib cimport calloc, free


from libc.stdlib cimport malloc, free
from libc.stdio cimport *

#import imageio

import random
import time
import math

import queue

   

cdef extern from "../../cp/image/cimg/CImg.h" namespace "cimg_library":
    cppclass CImg:
        CImg [T]()
 
"""
Class ImageT
"""

cdef class ImageIO:
    #attributes
    cdef _im_name
    cdef CImg _img

    def __cinit__(self):        
        pass

    def __dealloc__(self):
        pass
        

    def read(self, filed):
        pass
    
    