#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import time
from multiprocessing import Pool, Manager, Process, Lock



from vx.spix.Util import *
#from vx.com.px.descriptor.bic.bic import *
from vx.com.px.image.cimageio import *
#from vx.com.px.superpixels.snic.snic import *


class Superpixels:
    def __init__(self, arg):
        self.arg = arg

    def process(self, arg):
        start_time = time.time()
        imagename = arg["imagename"]
        inputdir = arg["inputdir"]
        outputdir = arg["outputdir"]
        side = arg["side"]
        
        #name, ext = Util.splitname(imagename)
        name, ext = Util.split(imagename,"_")
        print(imagename, name)
                    
        img_o = CImageIO()
        img_o.read(inputdir+imagename)
        img_o.setNameTarget(0,"background")
        img_o.setNameTarget(1,"lung")
        img_o.setSizeTargets(2)

        img_o.set_background_foreground(0,0,1)
        img_o.snicr(int(side), 45.5)


        #img_o.write_seg(outputdir+str(side)+"/"+name+".seg")
        img_o.draw_and_write_segments(outputdir+str(side)+"/"+name+".png")
        img_o.write_nrrd(inputdir, outputdir+str(side)+"/", imagename, name)
        #################img_o.write_npy(inputdir, outputdir+str(side)+"/", imagename, name)

        """
        img_o.setNameTarget(0,"background")
        img_o.setNameTarget(1,"lung")
        img_o.snicr(int(s))
        img_o.write_png(outputdir+str(s)+"/"+name+".png");

        """
        #img_o.write_png(outputdir+str(s)+"/"+name+".png");
        #print(img_o.channels())
        del img_o;


    def execute(self):
        data = []
        for row in self.arg:
            #sides =  row["sides"]
            inputdir = row["inputdir"]
            outputdir = row["outputdir"]

            print("self.arg[sides]", row["sides"])
            for s in row["sides"]:
                Util.makedir(outputdir+str(s))
                dat = {}
                for imagename in os.listdir(inputdir):
                    dat = {}
                    dat["imagename"] = imagename
                    dat["side"] = s
                    dat["inputdir"] = inputdir
                    dat["outputdir"] = outputdir
                    data.append(dat)

        pool = Pool(processes=10)
        rr = pool.map(self.process, data)
        pool.close()

        #for row in data:
        #    self.process(row)

if __name__ == "__main__":
    mainpath = "/mnt/sda6/software/posdoc/kaggle/Data/"
    data=[
        {   "sides":[50,100,200],
            "inputdir": mainpath+"dataOriginal_RemoveBackground/test/",
            "outputdir": mainpath+"dataOriginal_RemoveBackground_sp/test/",
        }
        
        #{   "sides":[50,100,200],
        #    "inputdir": mainpath+"dataOriginal_RemoveBackground_sp/train/",
        #    "outputdir": outputdir+"dataOriginal_RemoveBackground_sp/train/",
        #},
    ]
    obj = Superpixels(data)
    obj.execute()
    #make_superpixels(data)


