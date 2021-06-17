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


from queue import PriorityQueue
   

"""
cdef extern from "../../c/image/Image_t.h":
    ctypedef unsigned char PixelData
    ctypedef struct Color3RGB:
        PixelData R, G, B


    ctypedef struct Region_t:
        Pixel_t *next
        int size

    ctypedef struct Pixel_t:
        int i
        int x
        int y
        int label
        Pixel_t *next

    ctypedef struct Image_t:
        int width
        int height
        int channels
        int size_pixels
        int size_labels
        Pixel_t *pixel
        unsigned char *pixel_rgb
        double *pixel_lab

        Region_t *region

    Image_t* vx_image_read(char filename[])
    Image_t* vx_image_create(int w, int h, int intensity)
    Image_t* vx_image_from_seg(char filename[])
    void vx_image_setcolor(Image_t* img, int index, Color3RGB color)
    Color3RGB vx_image_getcolor(Image_t* img, int index)
    void vx_image_write(Image_t* img, char filename[])
    void vx_image_write_regions(Image_t* img, char filename[])
    void vx_image_free(Image_t* img)
    void vx_image_update_regions(Image_t *image)
"""

cdef extern from "../../cp/imageio/ImageIO.h":
    cdef cppclass ImageIO:
        ImageIO(std::string filename);
        ImageIO(std::string filename, int width, int height, int intensity, int channels);

        setColor(int i, Color3RGB c);
        Color3RGB getColor(int i);

        void setGray(int i, Color3RGB c);
        unsigned char getGray(int i);


cdef class CImageIO:
    #attributes
    cdef _im_name
    cdef ImageIO* _img

    def __cinit__(self):        
        pass

    def __dealloc__(self):
        vx_image_free(self._img)

    def read(self, filed):
        base = os.path.basename(filed)
        base = os.path.splitext(base)
        name = base[0]
        exte = base[1]

        filedX = filed.encode('UTF-8')
        cdef char* filedC = filedX

        self._im_name = name
        print(name, exte)
        
        
        if exte == '.tiff' or exte == '.jpg' or exte == '.png':
            self._img = vx_image_read(filedC)
        elif exte == '.seg':
            self._img = vx_image_from_seg(filedC)


    def create(self, h, w, intensity = 0, name="newimage"):
        self._im_name = name
        self._img = vx_image_create(h, w, intensity)


    def write_tiff(self, fileo):
        pass
        #vx_image_write(fileo, self._im)

    def write_jpg(self, fileo):
        pass
        #vx_image_write(fileo, self._im)

    def write_seg(self, fileo):
        pass
        #vx_image_write(fileo, self._im)


    """ 
    @staticmethod
    def snic(
            unsigned char[:,:] img,
            long[:,:] indices,
            unsigned char[:,:] mask,
            int side
            ):


        rows, cols = len(img),len(img[0])
        wrows, wcols = int(rows/side), int(cols/side)

        index = [ 0 for i in range(wcols*wrows+(wcols+wrows))]
        indexmap = {}
        for id in range(len(indices[0])):
            x0 = indices[0][id]
            y0 = indices[1][id]

            xt = int(x0/side)
            yt = int(y0/side)

            i = xt*wcols+yt;
            xc, yc = (x0+int(side/2)), (y0+int(side/2))

            if xc<rows and yc<cols and mask[xc][yc]==1 and index[i]!=1:
                indexmap[i] = (xc, yc)
                index[i] = 1


        Q = PriorityQueue()

        Cy = [ 0.0 for i in range(len(indexmap))]
        Cx = [ 0.0 for i in range(len(indexmap))]
        Cz = [ 0.0 for i in range(len(indexmap))]
        Cs = [ 0.0 for i in range(len(indexmap))]
        L = [ -1 for i in range(rows*cols)]
        kc = 0
        for key, val in indexmap.items():
            x, y = val[0], val[1]
            z = img[x][y]
            i = x*cols+y
            Q.put((-1.0, (i, x, y, z, kc)))
            kc=kc+1

        x8 = [-1,  0, 1, 0, -1,  1, 1, -1]
        y8 = [ 0, -1, 0, 1, -1, -1, 1,  1]

        t1 = time.time()

        while not Q.empty():
            w, e = Q.get()
            i, x, y, z, k = e[0], e[1], e[2], e[3], e[4]
            if (L[i]==-1):            
                L[i] = k
                Cx[k] += x
                Cy[k] += y
                Cz[k] += z
                Cs[k] += 1
                for p in range(8):
                    xj = x+x8[p]
                    yj = y+y8[p]
                    if xj<rows and yj<cols and mask[xj][yj]==1:
                        j = xj*cols+yj
                        if L[j]==-1:
                            zj = img[xj][yj]
                            ax, ay, az = Cx[k]-xj, Cy[k]-yj, Cz[k]-(zj*Cs[k])
                            ax, ay, az = ax*ax, ay*ay, az*az
                            #d = az/float(ay+ax)
                            d = az
                            Q.put((d, (j, xj, yj, zj, k)))
                        #print(d)
                #print(Q.qsize())
        print(Q.qsize())


        t2 = time.time()
        t = t2 - t1
        print("%.20f" % t)

        #print(len(indexmap))



            #except:
            #    print(x,y,i, wrows, wcols, indices[0][id], indices[1][id], rows, cols, side)

        #print(wrows, wtcols, wrows*wtcols)
    """