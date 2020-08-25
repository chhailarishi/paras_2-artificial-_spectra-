#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 11:58:51 2020

@author: paras
"""
#from numpy import genfromtxt
#import matplotlib.pyplot as plt
#from scipy.io import loadmat
#import numpy as np
#from astropy.io import fits
#import scipy.stats as st
from astropy.convolution import Gaussian1DKernel
#import math
import func_gen_spec
 
mat_file = "./CCD_array.mat"  #matlab file with wave info
sample_spec_file = 'sample_spectra.csv' # synthetic spectrum
kernel1 = Gaussian1DKernel(2.46)
rawim, wavelengths = func_gen_spec.generate_rawim(mat_file,sample_spec_file)

#%%
flatim = func_gen_spec.generate_flat(mat_file)

#%%
nord=73 #total number of orders
high_opti_ord = 162 # highest optical order number
blaze = 75.9 #degrees
inci = 77.0 #degres
groove_sep =  316455.6962025 #angstrm
final_eff_wvwise,fa, fc, fe, fg = func_gen_spec.final_effeciency_calc(wavelengths)
#%%
conv_disp_spatial_im = func_gen_spec.generate_conolved_spec(rawim, flatim, nord, mat_file, high_opti_ord, blaze, inci, groove_sep, kernel1, final_eff_wvwise,fa, fc, fe, fg)