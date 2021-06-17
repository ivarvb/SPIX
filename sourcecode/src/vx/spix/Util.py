#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ujson

from datetime import datetime

class Util:
    @staticmethod
    def write(file, obj):
        with open(file, "w") as filef:
            filef.write(ujson.dumps(obj))

    @staticmethod
    def read(file):
        data = {}
        with open(file,"r") as filef:
            data = (ujson.load(filef))
        return data

    @staticmethod
    def now():
        return datetime.now().strftime("%Y%m%d%H%M%S")

    @staticmethod
    def makedir(ndir):
        if not os.path.exists(ndir):
            os.makedirs(ndir)

    @staticmethod
    def splitname(filef):
        base = os.path.basename(filef)
        base = os.path.splitext(base)
        #name = base[0]
        return base
    
    @staticmethod
    def split(filef,separator):
        return filef.split(separator)