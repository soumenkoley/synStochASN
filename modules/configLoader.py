#!/usr/bin/env python
# coding: utf-8

# In[1]:


import configparser
import numpy as np

class Config:
    def __init__(self, path):
        cfg = configparser.ConfigParser()
        cfg.optionxform = str  # preserve case
        cfg.read(path)

        def arr(key):
            return np.array([float(v) for v in key.split(',') if v.strip() != ''])

        # Parse and store everything as attributes
        self.thickness = arr(cfg['model']['thickness'])
        self.vP = arr(cfg['model']['vp'])
        self.vS = arr(cfg['model']['vs'])
        self.rho = arr(cfg['model']['rho'])
        self.qP = arr(cfg['model']['qp'])
        self.qS = arr(cfg['model']['qs'])
        self.fMin = float(cfg['frequency']['fmin'])
        self.fMax = float(cfg['frequency']['fmax'])
        self.df = float(cfg['frequency']['df'])
        self.lambdaFrac = float(cfg['grid']['lambda_frac'])
        self.lambdaRes = int(cfg['grid']['lambda_res'])
        self.xMaxGF = float(cfg['geometry']['xmax_gf'])
        self.zMaxGF = float(cfg['geometry']['zmax_gf'])
        self.maxRec = int(cfg['geometry']['max_receivers'])
        self.tMax = int(cfg['geometry']['t_max'])
        self.nSamp = int(cfg['geometry']['n_samp'])
        self.templateFile = cfg['paths']['template_file']
        self.inputPath = cfg['paths']['input_folder']
        self.qseisExe = cfg['paths']['qseis_exe']
        self.outDispPath = cfg['paths']['out_disp_folder']
        self.outDispPathRea = cfg['paths']['out_disp_rea']
        self.delFlag = cfg['paths']['qseis_out_del']
        self.cpuCoresQseis = int(cfg['compute']['compute_cores_qseis'])
        self.cpuCoresDisp = int(cfg['compute']['compute_cores_disp'])
        self.cpuCoresNN = int(cfg['compute']['compute_cores_nn'])

# Global config instance
CONFIG = None

def load_config(path="configParse.ini"):
    global CONFIG
    CONFIG = Config(path)
    return CONFIG

