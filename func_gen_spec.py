#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jun  9 14:44:14 2020

@author: paras
"""



from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from astropy.convolution import convolve
#import math
def create_fits(filename,pixelarray,nan=False):
    if nan:
        pixelarray[pixelarray==0] = np.nan
    fits.writeto(filename+".fits", pixelarray, overwrite=True)
    
def polynomialfit(xdata, ydata):
    xdata=np.array(xdata); ydata=np.array(ydata)
    x_min=xdata.min(); x_max=xdata.max()
    coeffs=np.polyfit(xdata, ydata, 4)
    poly=np.poly1d(coeffs)
    x=np.linspace(x_min, x_max, 6144)
    y=poly(x)
    return x, y, coeffs


def generate_flat(wave_file):
    theo_loc = loadmat(wave_file)['P']
    create_fits('wavelength_fitted',theo_loc)
    theo_loc = theo_loc.round(2)
    trgts = theo_loc.nonzero()
    rw,cl = trgts[0], trgts[1]
    for m in range(len(rw)):
        theo_loc[rw[m],cl[m]] = 1
    create_fits("flat", theo_loc)
    return theo_loc




def generate_rawim(wave_file,spec_file):
    features_template = genfromtxt(spec_file, delimiter=",")
    pixelarray = loadmat(wave_file)['P']
    pixelarray = pixelarray.round(2)
    wavelengths = features_template[:,0]
    wavelengths = wavelengths.round(2)
    features = features_template[:,1]
    spectra = dict( (wavelengths[i],features[i]) for i in  range(len(features)))
    targets = pixelarray.nonzero()
    row,col = targets[0], targets[1]
    

    for l in range(len(row)):
        pixelarray[row[l],col[l]] = spectra.get(pixelarray[row[l],col[l]],0)
    create_fits("raw_im", pixelarray)
    return pixelarray, wavelengths



def blaze_intensity(xar, yar, ord_no, wave_file, high_opti_ord, blaze, inci, groove_sep):
    wave_array = loadmat(wave_file)['P']
    new_wave = wave_array[np.int64(yar),np.int64(xar)]
    n = high_opti_ord - ord_no    
    angle = np.radians(blaze)
    alpha = np.radians(inci)
    d = groove_sep
    beta1 = np.arcsin(((n*(new_wave)/d)-np.sin(angle)))
    beta = beta1-angle   
    s = (np.cos(angle)/np.cos(alpha))*d
    part = (2.0*np.pi/new_wave)*(s*np.sin(beta/2.0)*(np.cos(angle-(beta/2.0))))
    intensity_factor = ((np.sin(part))**2)/(part)**2
    
    return intensity_factor, new_wave

def final_effeciency_calc(wavelengths):
    wwa,tta1 = np.loadtxt('atm_trans.txt',unpack=True)
    wwc,ttc = np.loadtxt('ccd_eff.txt',unpack=True)
    wwe,tte = np.loadtxt('ech_eff.txt',unpack=True)
    wwg,ttg = np.loadtxt('grism_eff.txt',unpack=True)
    
    tta = -2.5*(np.log10(tta1))
    fa = interp1d(wwa, tta, kind='cubic')
    fc = interp1d(wwc, ttc, kind='cubic')
    fe = interp1d(wwe, tte, kind='cubic')
    fg = interp1d(wwg, ttg, kind='cubic')
    
    eff_a = fa(wavelengths)/max(fa(wavelengths))
    eff_c = fc(wavelengths)
    eff_e = fe(wavelengths)
    eff_g = fg(wavelengths)
    final_eff_wvwise = eff_a*eff_e*eff_c*eff_g
    plt.plot(wavelengths,eff_a)
    plt.plot(wavelengths,eff_c)
    plt.plot(wavelengths,eff_e)
    plt.plot(wavelengths,eff_g)
    return final_eff_wvwise, fa, fc, fe, fg

def generate_conolved_spec(rawim,flat_image,nord,wave_file,high_opti_ord,blaze,inci,groove_sep,kernal_conv,final_eff_wvwise,fa, fc, fe, fg):
    points_to_fit = np.zeros((nord,flat_image.shape[1]))
    for j in range(flat_image.shape[1]):
        peaks_every_row = (flat_image[:,j]).nonzero()
        points_to_fit[:,j] = np.asarray(peaks_every_row)

    order_coeff = np.zeros((nord,5))
    pixelarray_new = fits.getdata('raw_im.fits')
    #kernel2 = Gaussian1DKernel(2.65)

    for k in range(0,nord):
        y_arr = points_to_fit[k,:]
        x_arr = np.linspace(0,flat_image.shape[1]-1,num=flat_image.shape[1]-0)
        x,y,coeffs=polynomialfit(x_arr, y_arr)
        inten, new_wave = blaze_intensity(x_arr, y_arr, k, wave_file, high_opti_ord, blaze, inci, groove_sep)
        
        #########Efficiency
        eff_a_ord = fa(new_wave)
        eff_c_ord = fc(new_wave)
        eff_e_ord = fe(new_wave)
        eff_g_ord = fg(new_wave)
        eff_final_ord = eff_a_ord*eff_c_ord*eff_e_ord*eff_g_ord
        #############
        #final_eff = fits.getdata('final_eff.fits')
    
        order_coeff[k,:] = coeffs 
        raw_spec = pixelarray_new[np.int64(y_arr),np.int64(x_arr)]
        raw_flat_spec = flat_image[np.int64(y_arr),np.int64(x_arr)]
        conv_spec = convolve(raw_spec, kernal_conv)
        conv_spec_flat = convolve(raw_flat_spec, kernal_conv)
        pixelarray_new[np.int64(y_arr),np.int64(x_arr)] = conv_spec*(inten/max(inten))*eff_final_ord/np.max(final_eff_wvwise)
        flat_image[np.int64(y_arr),np.int64(x_arr)] = conv_spec_flat*(inten/max(inten))*eff_final_ord/np.max(final_eff_wvwise)
        #xair = np.linspace(0,6143,6144)
        plt.plot(new_wave,conv_spec*(inten/max(inten))*eff_final_ord/np.max(final_eff_wvwise))
        plt.plot(new_wave, eff_final_ord/np.max(final_eff_wvwise))
        plt.show()
    


    np.savetxt('order_location.txt',order_coeff)   
    create_fits("conv_disp", pixelarray_new)

  
    a = pixelarray_new.nonzero()
    ro,co = a[0], a[1]

    for o in range(len(ro)):
        (pixelarray_new[ro[o],co[o]]) = (pixelarray_new[ro[o],co[o]])*50000.0
        pixelarray_new[ro[o]-10:ro[o]+11,co[o]] = (pixelarray_new[ro[o],co[o]])*(kernal_conv.array)
        (flat_image[ro[o],co[o]]) = (flat_image[ro[o],co[o]])*200000.0
        flat_image[ro[o]-10:ro[o]+11,co[o]] = (flat_image[ro[o],co[o]])*(kernal_conv.array)
    create_fits("conv_spatial", pixelarray_new)
    create_fits("flat_to_use", flat_image)
    return pixelarray_new


