#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:44:14 2020

@author: Rishikesh Sharma

This code generates the simulated CCD output spectra for a high-resolution spectrograph PARAS-2. 
"""


# %%
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from astropy.convolution import convolve, Gaussian1DKernel
import math
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

#%%
mat_file = "./CCD_array.mat"

def generate_flat(wave_file):
    theo_loc = loadmat(wave_file)['P']
    create_fits('wavelength_fitted',theo_loc)
    theo_loc = theo_loc.round(2)
    trgts = theo_loc.nonzero()
    rw,cl = trgts[0], trgts[1]
    for m in range(len(rw)):
        theo_loc[rw[m],cl[m]] = 1
    create_fits("flat_im", theo_loc)
    return theo_loc




def generate_rawim(wave_file):
    features_template = genfromtxt("sample_spectra.csv", delimiter=",")
    pixelarray = loadmat("./CCD_array.mat")['P']
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

rawim, wavelengths = generate_rawim(mat_file)
flatim = generate_flat(mat_file)
kernel1 = Gaussian1DKernel(2.65)


#%%
'''
#def blaze_intensity(xar, yar, ord_no):
    wave_array = loadmat("./CCD_array.mat")['P']
    new_wave = wave_array[np.int64(yar),np.int64(xar)]
    n = 162 - ord_no
    angle = np.radians(75.0)
    d = 316455.6962025
    beta = np.asin(((n*new_wave)/d)-np.sin(angle))
    part = (n*(np.pi)/2.0)*((np.sin(angle))+(np.cos(beta))*(1.0/np.tan(angle)))
    intensity_factor = ((np.sin(part))**2)/(part)**2
    return intensity_factor
'''
##############
wwa,tta1 = np.loadtxt('atm_trans.txt',unpack=True)
wwc,ttc = np.loadtxt('ccd_eff.txt',unpack=True)
wwe,tte = np.loadtxt('ech_eff.txt',unpack=True)
wwg,ttg = np.loadtxt('grism_eff.txt',unpack=True)

tta = -2.5*(np.log10(tta1))
fa = interp1d(wwa, tta, kind='cubic')
fc = interp1d(wwc, ttc, kind='cubic')
fe = interp1d(wwe, tte, kind='cubic')
fg = interp1d(wwg, ttg, kind='cubic')

eff_a = fa(wavelengths)
eff_c = fc(wavelengths)
eff_e = fe(wavelengths)
eff_g = fg(wavelengths)
final_eff_wvwise = eff_a*eff_e*eff_c*eff_g
plt.plot(wavelengths,eff_a)
plt.plot(wavelengths,eff_c)
plt.plot(wavelengths,eff_e)
plt.plot(wavelengths,eff_g)
###############
#%%
#def generate_conolved_spec(rawim,flatim,nord=73):
final_eff_all_ord=[]
final_eff_all_ord_wave=[]
nord=73
points_to_fit = np.zeros((nord,flatim.shape[1]))
#orders_points_to_fit = np.zeros((nord,flatim.shape[1],flatim.shape[1]))
for j in range(flatim.shape[1]):
    peaks_every_row = (flatim[:,j]).nonzero()
    points_to_fit[:,j] = np.asarray(peaks_every_row)

order_coeff = np.zeros((nord,5))
pixelarray_new = fits.getdata('raw_im.fits')
flat_image = fits.getdata('flat_im.fits')
kernel2 = Gaussian1DKernel(2.46)

for k in range(0,nord):
    y_arr = points_to_fit[k,:]
    x_arr = np.linspace(0,flatim.shape[1]-1,num=flatim.shape[1]-0)
    x,y,coeffs=polynomialfit(x_arr, y_arr)
    #inten = blaze_intensity(x_arr, y_arr, k)
    wave_array = loadmat("./CCD_array.mat")['P']
    new_wave = wave_array[np.int64(y_arr),np.int64(x_arr)]
    n = 162 - k
    
    angle = np.radians(75.3)
    alpha = np.radians(75.9)
    d = 316455.6962025
    beta1 = np.arcsin(((n*(new_wave)/d)-np.sin(angle)))
    beta = beta1-angle
    '''###############
    beta = np.zeros(len(beta1))
    
    for ris in range (0,len(beta1)):
        if np.rad2deg(beta1[ris])>76.0:
            beta[ris] = (beta1[ris]-angle)+angle
        else:
            beta[ris] = (angle-beta1[ris])+angle
    ####################'''
    #part = (n*(np.pi)/2.0)*((np.cos(angle))+(np.cos(beta)/np.tan(angle)))
    s = (np.cos(alpha)/np.cos(angle))*d
    part = (2.0*np.pi/new_wave)*(s*np.sin(beta/2.0)*(np.cos(angle-(beta/2.0))))
    inten = ((np.sin(part))**2)/(part)**2
    
    ###########EFF_FACTOR
    
    eff_a_ord = fa(new_wave)
    eff_c_ord = fc(new_wave)
    eff_e_ord = fe(new_wave)
    eff_g_ord = fg(new_wave)
    
    eff_final_ord = eff_a_ord*eff_c_ord*eff_e_ord*eff_g_ord
    ################
    
    '''
    angle = np.radians(75.2)
    alpha = np.radians(75.2)
    d = 316455.6962025
    beta = np.arcsin(((n*(new_wave)/d)-np.sin(angle)))
    eqn_k_comp = (np.cos(beta)/np.cos(alpha))*(np.cos(alpha-angle)/np.cos(beta-angle))
    eqn_k_comp[eqn_k_comp < 1.0] = 1.0
    rho = np.cos(alpha)/np.cos(alpha-angle)
    #inten = eqn_k_comp*(np.sinc(((np.pi*d*rho)/new_wave)*(np.sin(alpha-angle)+np.sin(beta-angle))))**2
    inten = eqn_k_comp*(np.sinc(((np.pi*n*rho))*(np.cos(angle)+(np.sin(angle)*(1/np.tan((alpha+beta)/2.0))))))**2
    '''
    final_eff = fits.getdata('final_eff.fits')
    order_coeff[k,:] = coeffs 
    #plt.plot(x_arr,y_arr)
    raw_spec = pixelarray_new[np.int64(y_arr),np.int64(x_arr)]
    raw_flat_spec = flat_image[np.int64(y_arr),np.int64(x_arr)]
    #plt.plot(rawim[np.int64(y_arr),np.int64(x_arr)]*10000.0,raw_spec)
    conv_spec = convolve(raw_spec, kernel2)
    conv_spec_flat = convolve(raw_flat_spec, kernel2)
    #plt.plot(rawim[np.int64(y_arr),np.int64(x_arr)]*10000.0,conv_spec)
    #print(max(conv_spec),max(inten))
    pixelarray_new[np.int64(y_arr),np.int64(x_arr)] = conv_spec*(inten/max(inten))*eff_final_ord/np.max(final_eff)
    flat_image[np.int64(y_arr),np.int64(x_arr)] = conv_spec_flat*(inten/max(inten))*eff_final_ord/np.max(final_eff)
    #plt.plot(np.rad2deg(beta))
   # final_eff_all_ord.append(eff_final_ord)
   # final_eff_all_ord_wave.append(new_wave)
    
    xair = np.linspace(0,6143,6144)
    plt.plot(new_wave,conv_spec*(inten/max(inten))*eff_final_ord/np.max(final_eff))
    plt.plot(new_wave, eff_final_ord/np.max(final_eff))
    
    #plt.plot(xair+(6144*k),(inten))
    plt.show()
    

#%%    
np.savetxt('order_location.txt',order_coeff)   
create_fits("conv_disp", pixelarray_new)

#pixelarray_new = fits.getdata('conv_disp.fits')
a = pixelarray_new.nonzero()
ro,co = a[0], a[1]
kernel1 = Gaussian1DKernel(2.46)

for o in range(len(ro)):
    (pixelarray_new[ro[o],co[o]]) = (pixelarray_new[ro[o],co[o]])*50000.0
    pixelarray_new[ro[o]-10:ro[o]+11,co[o]] = (pixelarray_new[ro[o],co[o]])*(kernel1.array)
    (flat_image[ro[o],co[o]]) = (flat_image[ro[o],co[o]])*200000.0
    flat_image[ro[o]-10:ro[o]+11,co[o]] = (flat_image[ro[o],co[o]])*(kernel1.array)
    #pixelarray_copy[row[i]-9:row[i]+9, col[i]] = pixelarray_copy[row[i],col[i]]*kernel1
create_fits("conv_spatial", pixelarray_new)
create_fits("flat_to_use", flat_image)
#return pixelarray_new

#conv_disp_spatial_im = generate_conolved_spec(rawim,flatim)


    
    
