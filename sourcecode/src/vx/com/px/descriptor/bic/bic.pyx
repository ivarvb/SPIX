# distutils: language = c++

import numpy as np
cimport numpy as np

from libc.stdlib cimport calloc, free
from libc.stdlib cimport malloc, free
from libc.stdio cimport *

from vx.com.px.image.cimageio import *



#from libc.math cimport pow

import time
import numpy
import math
#cimport numpy
#ctypedef np.int_t DTYPE_t

cdef extern from "../../../c/image/ImageIO_t.h":
    ctypedef struct ImageIO_t:
        int width, height, channels;
        unsigned long size_pixels, size_labels;

        unsigned char *pixel; 
        unsigned long *label;
        unsigned long *next;
        unsigned long *region;
        unsigned long *regionsize;

        unsigned long size_targets;
        unsigned long *target;
        char **name_targets;

cdef extern from "../../../c/descriptor/BIC.h":
    void vx_bic(
        ImageIO_t*& img,
        double **vector,
        int bins,
        int *dist,
        int size_dist
    );


def bic(img, int bins, long[:] dis, ):
    exporting_object = np.arange(111, 20, dtype=np.doublec)
    cdef double[:, :] vector = exporting_object

    vx_bic( (<ImageIO>(<CImageIO>img)._img)._img, vector,  bins, dist, len(dist))

    print(hola)

""" 
def bic(    unsigned char[:,:] img,
            long[:,:] indices,
            unsigned char[:,:] mask,
            long[:] dis, int bins):

    t1 = time.time()
    rows, cols = len(img),len(img[0])

    xn = [-1, 1, 1,-1]
    yn = [-1,-1, 1, 1]
    we = int(256/(bins))
    fc = (1.0/float(bins))
    #h = [ [0 for i in range(we*2)]  for k in range(len(dis)) ]
    h = [ [0 for i in range(bins*2)]  for k in range(len(dis)) ]
    for i in range(len(indices[0])):
        x = indices[0][i]
        y = indices[1][i]

        for k in range(len(dis)):
            d = dis[k]
            for j in range(len(xn)):
                xj=x+(xn[j]*d)
                yj=y+(yn[j]*d)
                
                if xj>=0 and xj<rows and yj>=0 and yj<cols and mask[xj][yj]==1:
                    pi=int((img[x][y]/256.0)/fc)
                    pj=int((img[xj][yj]/256.0)/fc)
                    
                    w = 1

                    #pi=int((img[x][y]/we))
                    #pj=int((img[xj][yj]/we))


                    if pi == pj:
                        #pass
                        h[k][pi]+=w
                    else:
                        #pass
                        try:
                            h[k][bins+pi]+=w
                            h[k][bins+pj]+=w
                    
                            #h[k][we+pi]+=1
                            #h[k][we+pj]+=1
                        except:
                            print("pi:{} pj: {} we:{} d:{} bins{}".format(pi, pj, bins))
                            exit()

    hr = []
    for d in h:
        hr +=d

    t2 = time.time()
    t = t2 - t1
    #print("%.20f" % t)
    return hr

def bicw(   unsigned char[:,:] img,
            long[:,:] indices,
            unsigned char[:,:] mask,
            long[:] dis, int bins):

    t1 = time.time()
    rows, cols = len(img),len(img[0])

    xn = [-1, 1, 1,-1]
    yn = [-1,-1, 1, 1]
    we = int(256/(bins))
    fc = (1.0/float(bins))
    #h = [ [0 for i in range(we*2)]  for k in range(len(dis)) ]
    h = [ [0 for i in range(bins*2)]  for k in range(len(dis)) ]
    for i in range(len(indices[0])):
        x = indices[0][i]
        y = indices[1][i]

        for k in range(len(dis)):
            d = dis[k]
            for j in range(len(xn)):
                xj=x+(xn[j]*d)
                yj=y+(yn[j]*d)
                
                #if xj>=0 and xj<rows and yj>=0 and yj<cols:
                if xj>=0 and xj<rows and yj>=0 and yj<cols and mask[xj][yj]==1:
                    pi=int((img[x][y]/256.0)/fc)
                    pj=int((img[xj][yj]/256.0)/fc)

                    w = pi-pj
                    w = math.sqrt(w*w)
                    #pi=int((img[x][y]/we))
                    #pj=int((img[xj][yj]/we))


                    if pi == pj:
                        #pass
                        h[k][pi]+=w
                    else:
                        #pass
                        try:
                            h[k][bins+pi]+=w
                            h[k][bins+pj]+=w
                    
                            #h[k][we+pi]+=1
                            #h[k][we+pj]+=1
                        except:
                            print("pi:{} pj: {} we:{} d:{} bins{}".format(pi, pj, bins))
                            exit()
    hr = []
    for d in h:
        hr +=d

    t2 = time.time()
    t = t2 - t1
    #print("%.20f" % t)
    return hr

 """