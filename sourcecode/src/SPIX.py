#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from vx.spix.Superpixels import *

if __name__ == "__main__":
    #mainpath = "/home/ivar/kaggle/"
    #mainpath = "/home/home/gbdi/Data/"
    mainpath = "/home/ivar/kaggle/"
    
    #mainpath = "/mnt/sda6/software/posdoc/kaggle/Data/"
    data =  [
                {   "sides":[50,100,200],
                    "inputdir": mainpath+"dataOriginal_RemoveBackground/test/",
                    "outputdir": mainpath+"superpixels/dataOriginal_RemoveBackground/test/",
                },
                {   "sides":[50,100,200],
                    "inputdir": mainpath+"dataOriginal_RemoveBackground/train/",
                    "outputdir": mainpath+"superpixels/dataOriginal_RemoveBackground/train/",
                }
                
                #{   "sides":[50,100,200],
                #    "inputdir": mainpath+"dataOriginal_RemoveBackground_sp/train/",
                #    "outputdir": outputdir+"dataOriginal_RemoveBackground_sp/train/",
                #},
            ]

    obj = Superpixels(data)
    obj.execute()
    #make_superpixels(data)
