

cimport numpy as np

from libc.stdlib cimport calloc, free
from libc.stdlib cimport malloc, free
from libc.stdio cimport *




#from libc.math cimport pow

import time
import numpy
import math

#import numpy as np
cimport numpy as cnp

import time
from queue import PriorityQueue


cdef extern from "../../../cp/superpixels/SNIC.h":
    void vx_snic(unsigned char **img, long **indices, unsigned char **mask, int side)
    void vx_read();


class SNIC:
    @staticmethod
    def execute(
                unsigned char[:,:] img,
                long[:,:] indices,
                unsigned char[:,:] mask,
                int side    ):

        #cdef unsigned char **ptr1 = &img[0]
        #cdef long **ptr2 = <long**>indices
        #cdef unsigned char **ptr3 = <unsigned char**>mask


        #pass
        #vx_snic(ptr1, ptr2, ptr3, side);
        #vx_read()

        t1 = time.time()

        for id in range(len(indices[0])):
            x0 = indices[0][id]
            y0 = indices[1][id]
            v = img[x0][y0]
            
            #print(v)

        t2 = time.time()
        t = t2 - t1
        print("%.20f" % t)

        """ 
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
        """