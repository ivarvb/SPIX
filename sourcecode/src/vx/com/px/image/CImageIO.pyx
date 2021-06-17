# distutils: language = c++

import os
from typing import List
import numpy as np
cimport numpy as np
#cimport cython


import matplotlib.pyplot as plt
import SimpleITK as sitk

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


cdef extern from "../../c/image/ImageIO_t.h":
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
        char *name;
        char **name_targets;

cdef extern from "../../c/image/Color3RGB.h":
    ctypedef struct Color3RGB:
        unsigned char R, G, B;
        
cdef extern from "../../cp/image/imageio/ImageIO.h":
    cdef cppclass ImageIO:
        ImageIO_t *_img;
        
        ImageIO(char filename_[]);
        ImageIO(char filename_[], int width, int height, int channels);

        int width();
        int height();
        int size_pixels();
        int size_labels();
        int channels();
        
        void setNameTarget(int i, char name[]);
        void setTarget(unsigned long i, unsigned long target);
        void setSizeTargets(int size);

        void snicr(int side, double factor);

        setColor(int i, Color3RGB c);
        Color3RGB getColor(int i);

        void setGray(int i, unsigned char c);
        unsigned char getGray(int i);

        void gray_update_from(
            unsigned char gray, unsigned long label,
            ImageIO *&ref, unsigned char gray_ref, unsigned long label_ref
        );
        
        void set_background_foreground(
            unsigned char bg_pixel,
            unsigned long bg_label,
            unsigned long fg_label
        );
        
        void read_seg(char filename[]);

        void copy_pixels_from(ImageIO *ref);
        void copy_labels_from(ImageIO *ref);

        void draw_regions_colors();
        void draw_regions_limits();
        void draw_and_write_segments(char filename[]);


        void write_seg(char filename_[]);
        void write_png(char filename_[]);
        void write_tiff(char filename_[]);


cdef extern from "../../c/descriptor/BIC.h":
    void vx_bic(
        ImageIO_t*& img,
        double *features,
        int size_rows_features,
        int size_cols_features,
        int bins,
        int *offsetx, int *offsety, int size_offset
    );
    void vx_bicw(
        ImageIO_t*& img,
        double *features,
        int size_rows_features,
        int size_cols_features,
        int bins,
        int *offsetx, int *offsety, int size_offset
    );

cdef extern from "../../c/descriptor/GLCM.h":
    void vx_glcm(
        ImageIO_t* img, double *features,
        int size_rows_features, int size_cols_features,
        int *offsetx, int *offsety, int size_offset
    );




cdef class CImageIO:
    cdef ImageIO* _img
    cdef _name
    def __cinit__(self):
        pass        

    def __dealloc__(self):
        del self._img
        #self._img

    def read(self, filed):
        #start_time = time.time()
        cdef char* filedC;
        base = os.path.basename(filed)
        base = os.path.splitext(base)
        name = base[0]
        exte = base[1]
        #print(name, exte)

        filedX = filed.encode('UTF-8')
        filedC = filedX
        
        self._name = name
        self._img = new ImageIO(filedC)
        #print("time cimaion: ",(time.time() - start_time))

    def create(self, namef="newimage", width=100, height=100, channels=3):
        name = namef.encode('UTF-8')
        cdef char* nameC = name;
            
        self._name = namef
        self._img = new ImageIO(nameC, width, height, channels);

    def width(self):
        return self._img.width()

    def height(self):
        return self._img.height()

    def size_pixels(self):
        return self._img.size_pixels()

    def size_labels(self):
        return self._img.size_labels()

    def channels(self):
        return self._img.channels()

    def setNameTarget(self, i, namet):
        name = namet.encode('UTF-8')
        cdef char* nameC = name;
        self._img.setNameTarget(i,  nameC)

    def setTarget(self, i, target):
        self._img.setTarget(i, target);

    def setSizeTargets(self, size):
        self._img.setSizeTargets(size);

    def snicr(self, side, factor):
        self._img.snicr(side, factor);


    def gray_update_from(self, gray, label, img_ref, gray_ref, label_ref):
        self._img.gray_update_from(gray, label, (<CImageIO>img_ref)._img, gray_ref, label_ref);

    def set_background_foreground(self, bg_pixel, bg_label, fg_label):
        self._img.set_background_foreground(bg_pixel,bg_label,fg_label)


    def read_seg(self, fileo):
        name = fileo.encode('UTF-8')
        cdef char* nameC = name
        self._img.read_seg(nameC)

    def copy_labels_from(self, img_ref):
        self._img.copy_labels_from( (<CImageIO>img_ref)._img );


    def draw_regions_limits(self):
        self._img.draw_regions_limits();

    def draw_regions_colors(self):
        self._img.draw_regions_colors();

    def draw_and_write_segments(self, fileo):
        name = fileo.encode('UTF-8')
        cdef char* nameC = name
        self._img.draw_and_write_segments(nameC);

    def write_seg(self, fileo):
        name = fileo.encode('UTF-8')
        cdef char* nameC = name
        self._img.write_seg(nameC)

    def write_png(self, fileo):
        name = fileo.encode('UTF-8')
        cdef char* nameC = name
        self._img.write_png(nameC)

    def write_tiff(self, fileo):
        name = fileo.encode('UTF-8')
        cdef char* nameC = name
        self._img.write_tiff(nameC)


    def write_nrrd(self, inputdir, outputdir, imagname, imgoname):

        filei = inputdir + imagname
        
        image = sitk.ReadImage(filei)
        im_size = np.array(image.GetSize())[::-1]
        ma_arr = np.zeros(im_size, dtype=int)        
        
        height = self._img.height()
        width = self._img.width()

        for x in range(width):
            for y in range(height):
                i = y*width+x;
                #i = x*height+y;
                ma_arr[y][x] = (self._img)._img.label[i]

        ma = sitk.GetImageFromArray(ma_arr)
        ma.CopyInformation(image)
        sitk.WriteImage(ma, outputdir+imgoname+".nrrd", True)  # True specifies it can use compression
        #sitk.WriteImage(sitk.LabelToRGB(ma), outputdir+imgoname+".png")

    """ 
    def write_npy(self, inputdir, outputdir, imagname, imgoname):
        height = self._img.height()
        width = self._img.width()

        superpixels = [[] for i in range(((self._img)._img.size_labels)+1) ]
        for r in range(1,(self._img)._img.size_labels):
            i = (self._img)._img.region[r]
            c=0
            while c< (self._img)._img.regionsize[r]:
                #i = y*width+x
                x = i % width;
                y = (i - x) / width;
                label = (self._img)._img.label[i]
                if label>0:
                    superpixels[label].append((x,y))
            i = (self._img)._img.next[i];
            c += 1

        an_array = np.array(superpixels)
        np.save(outputdir+imgoname+'.npy', an_array)
        del an_array
    """

class Descriptor():
    @staticmethod
    def formatresults(imgt, features, size_rows_features, size_cols_features ):
        #format results
        features_out = []
        for y in range(1,size_rows_features):
            row_feat = []

            name = <bytes>(imgt.name)
            uname = name.decode('UTF-8')
            row_feat.append(uname)

            row_feat.append(y)
            for x in range(size_cols_features):
                row_feat.append( features[y*size_cols_features+x] )
            features_out.append(row_feat)
            
            target = <bytes>(imgt.name_targets[imgt.target[y]])
            utarget = target.decode('UTF-8')
            row_feat.append(utarget)            
        
        return features_out

    @staticmethod
    def bic(img, bins, dist):
        angl = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        size_dist = len(dist) 
        size_angl = len(angl) 
        size_offset = size_angl*size_dist;
        cdef int* offsetx = <int*> malloc((size_offset)*sizeof(int))
        cdef int* offsety = <int*> malloc((size_offset)*sizeof(int))
        c=0
        for a in angl:
            for d in dist:
                offsetx[c] = np.cos(a)*d
                offsety[c] = np.sin(a)*d
                c+=1       



        cdef ImageIO_t *imgt = (<CImageIO>img)._img._img;
        size_cols_features = bins*2*size_dist
        size_rows_features = imgt.size_labels

        cdef double* features = <double*> malloc((size_rows_features*size_cols_features) * sizeof(double))
        vx_bic(imgt, features, size_rows_features, size_cols_features, bins, offsetx, offsety, size_offset)




        #format results
        features_out = []
        for y in range(1,size_rows_features):
            row_feat = []

            name = <bytes>(imgt.name)
            uname = name.decode('UTF-8')
            row_feat.append(uname)

            row_feat.append(y)
            for x in range(size_cols_features):
                row_feat.append( features[y*size_cols_features+x] )
            features_out.append(row_feat)
            
            target = <bytes>(imgt.name_targets[imgt.target[y]])
            utarget = target.decode('UTF-8')
            row_feat.append(utarget)            

        free(features)
        free(offsetx)
        free(offsety)
        
        return features_out


    @staticmethod
    def bicw(img, bins, dist):
        angl = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        size_dist = len(dist) 
        size_angl = len(angl) 
        size_offset = size_angl*size_dist;
        cdef int* offsetx = <int*> malloc((size_offset)*sizeof(int))
        cdef int* offsety = <int*> malloc((size_offset)*sizeof(int))
        c=0
        for a in angl:
            for d in dist:
                offsetx[c] = np.cos(a)*d
                offsety[c] = np.sin(a)*d
                c+=1       

        cdef ImageIO_t *imgt = (<CImageIO>img)._img._img;
        size_cols_features = bins*2*size_dist
        size_rows_features = imgt.size_labels

        cdef double* features = <double*> malloc((size_rows_features*size_cols_features) * sizeof(double))
        vx_bicw(imgt, features, size_rows_features, size_cols_features, bins, offsetx, offsety, size_offset)

        #format results
        features_out = []
        for y in range(1,size_rows_features):
            row_feat = []

            name = <bytes>(imgt.name)
            uname = name.decode('UTF-8')
            row_feat.append(uname)

            row_feat.append(y)
            for x in range(size_cols_features):
                row_feat.append( features[y*size_cols_features+x] )
            features_out.append(row_feat)
            
            target = <bytes>(imgt.name_targets[imgt.target[y]])
            utarget = target.decode('UTF-8')
            row_feat.append(utarget)            

        free(features)
        free(offsetx)
        free(offsety)
        
        return features_out
    
    def glcm(img, dist):
        #compute coefficient
        """ 
        coef_set =  {
                        "energy":0,
                        "homogeneity":1,
                        "dissimilarity":2,
                        "contrast":3,
                        "correlation":4,
                        "ASM":5
                    }
        coef_aux = [ coef_set[co] for co in coef]
        cdef int* coefs = <int*> malloc((len(coef_aux))*sizeof(int))    
        for i in range(len(coef_aux)):
            coefs[i] = coef_aux[i]
        """
        angl = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        size_coef = 6
        size_dist = len(dist) 
        size_angl = len(angl) 
        size_offset = size_angl*size_dist;

        #compute offsetx and offsety
        cdef int* offsetx = <int*> malloc((size_offset)*sizeof(int))
        cdef int* offsety = <int*> malloc((size_offset)*sizeof(int))
        c = 0
        for a in angl:
            for d in dist:
                offsetx[c] = np.cos(a)*d
                offsety[c] = np.sin(a)*d
                c+=1       
        #get image, rows and cols
        cdef ImageIO_t *imgt = (<CImageIO>img)._img._img;
        size_rows_features = imgt.size_labels
        size_cols_features = size_offset*size_coef

        #feature matrix
        cdef double* features = <double*> malloc((size_rows_features*size_cols_features)*sizeof(double))
        
        #compute feature for all segments
        vx_glcm(
            imgt, features,
            size_rows_features, size_cols_features,
            offsetx, offsety, size_offset);

        #format results
        features_out = []
        for y in range(1,size_rows_features):
            row_feat = []

            name = <bytes>(imgt.name)
            uname = name.decode('UTF-8')
            row_feat.append(uname)

            row_feat.append(y)
            for x in range(size_cols_features):
                row_feat.append( features[y*size_cols_features+x] )
            features_out.append(row_feat)
            
            target = <bytes>(imgt.name_targets[imgt.target[y]])
            utarget = target.decode('UTF-8')
            row_feat.append(utarget)            

        free(features)
        free(offsetx)
        free(offsety)
        #free(coefs)

        return features_out


