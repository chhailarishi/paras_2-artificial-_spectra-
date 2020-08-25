# %%
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from astropy.io import fits
# %matplotlib
import scipy.stats as st
import functions as fn
from astropy.convolution import convolve, Gaussian1DKernel
def create_fits(filename,pixelarray,nan=False):
    if nan:
        pixelarray[pixelarray==0] = np.nan
    fits.writeto(filename+".fits", pixelarray, overwrite=True)

'''
def gauss_kernel(kernlen = kern ,nsig = sig):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    # kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    # kernel = kernel_raw/kernel_raw.sum()
    return kern1d
'''
#%%

features_template = genfromtxt("sample_spectra.csv", delimiter=",")

pixelarray = loadmat("./CCD_array.mat")['P']
#pixelarray = np.rot90(pixelarray, 3)
pixelarray = pixelarray.round(2)
pixelarray_1 = pixelarray.copy()
wavelengths = features_template[:,0]
#wavelengths = wavelengths/10000
wavelengths = wavelengths.round(2)

features = features_template[:,1]

spectra = dict( (wavelengths[i],features[i]) for i in  range(len(features)))

targets = pixelarray.nonzero()
row,col = targets[0], targets[1]

targets1 = pixelarray_1.nonzero()
row1,col1 = targets1[0], targets1[1]

spec_to_degrade = np.zeros(len(row))
wave_to_degrade = np.zeros(len(row))
#kernel1 = gauss_kernel(kernlen = 18.0 ,nsig = 4.0)
kernel1 = Gaussian1DKernel(3)
for i in range(len(row1)):
    pixelarray_1[row1[i],col1[i]] =  1

create_fits("flat", pixelarray_1)

pixelarray_1 = fits.getdata('flat.fits')
#%%
nord = 73
points_to_fit = np.zeros((nord,pixelarray_1.shape[1]))
orders_points_to_fit = np.zeros((nord,pixelarray_1.shape[1],pixelarray_1.shape[1]))
for j in range(pixelarray_1.shape[1]):
    #plt.plot(pixelarray_copy[:,j])
    peaks_every_row = (pixelarray_1[:,j]).nonzero()
    points_to_fit[:,j] = np.asarray(peaks_every_row)
    
plt.imshow(pixelarray_1)
order_coeff = np.zeros((nord,5))
for k in range(nord):
    y_arr = points_to_fit[k,:]
    x_arr = np.linspace(0,pixelarray.shape[1]-1,num=pixelarray.shape[1]-0)
    x,y,coeffs=fn.polynomialfit(x_arr, y_arr)
    order_coeff[k,:] = coeffs 
    plt.plot(x_arr,y_arr)
    
np.savetxt('order_location.txt',order_coeff)   
#%%

for l in range(len(row)):
    #wave_to_degrade[i] = pixelarray[row[i],col[i]]
    #spec_to_degrade[i] = spectra.get(pixelarray_copy[row[i],col[i]],0)
    pixelarray[row[l],col[l]] = spectra.get(pixelarray[row[l],col[l]],0)
    #pixelarray_copy[row[i]-9:row[i]+9, col[i]] = pixelarray_copy[row[i],col[i]]*kernel1
#################Convoving the spectra to match instrument resolution############
kernel2 = Gaussian1DKernel(2.55)
for m in range(0,1):
    poly=np.poly1d(order_coeff[m,:])
    x=np.linspace(0, 6143, 6144)
    y=poly(x)
    raw_spec = pixelarray[np.int64(y),np.int64(x)]
    conv_spec = convolve(raw_spec, kernel2)
    plt.plot(raw_spec)
    plt.plot(conv_spec)

    


