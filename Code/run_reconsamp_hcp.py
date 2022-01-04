#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:31:54 2020

@author: ktezcan
"""

import numpy as np

import pickle
import vaerecon6
import vaesampling

import scipy as sc 
import SimpleITK as sitk   

import argparse    
import sys  
import os  


###############################################################################
### This script runs on an example slice from the Human Connectome Datbase
### (HCP) dataset.  It runs
### 1. a reconstruction based on the deep density priors (DDP) method used to 
###    initialize the sampling.
### 2. the sampling method with the output of the VAE decoder as the samples,
### 3. the samples from the proper posterior
############################################################################### 



###############################################################################
### Get the directory of the script to access files later
###############################################################################
scriptdir = os.path.dirname(sys.argv[0])+'/'
print("KCT-info: running script from directory: " + scriptdir)
os.chdir(scriptdir)

#make the necessary folders if not existing
if not os.path.exists(os.getcwd() + '/../../results/hcp/reconstruction/'):
    os.mkdir(os.getcwd() + '/../../results/hcp/reconstruction/')
    
if not os.path.exists(os.getcwd() + '/../../results/hcp/samples/'):
    os.mkdir(os.getcwd() + '/../../results/hcp/samples/')
    
if not os.path.exists(os.getcwd() + '/../../results/hcp/samples/decoder_samples/'):
    os.mkdir(os.getcwd() + '/../../results/hcp/samples/decoder_samples/')





###############################################################################
### Some necessary functions
###############################################################################

def FT (x, normalize=False):
     #inp: [nx, ny]
     #out: [nx, ny]
     if normalize:
          return np.fft.fftshift(    np.fft.fft2(  x , axes=(0,1)  ),   axes=(0,1)    ) / np.sqrt(252*308)
     else:
          return np.fft.fftshift(    np.fft.fft2(  x , axes=(0,1)  ),   axes=(0,1)    ) 

def tFT (x, normalize=False):
     #inp: [nx, ny]
     #out: [nx, ny]
     if normalize:
          return np.fft.ifft2(  np.fft.ifftshift( x , axes=(0,1) ),  axes=(0,1)  )   * np.sqrt(252*308)
     else:
          return np.fft.ifft2(  np.fft.ifftshift( x , axes=(0,1) ),  axes=(0,1)  )   

def UFT(x, uspat, normalize=False):
     #inp: [nx, ny], [nx, ny]
     #out: [nx, ny]
     
     return uspat*FT(x, normalize)

def tUFT(x, uspat, normalize=False):
     #inp: [nx, ny], [nx, ny]
     #out: [nx, ny]
     return  tFT( uspat*x ,normalize)


def calc_rmse(rec,imorig):
     return 100*np.sqrt(np.sum(np.square(np.abs(rec) - np.abs(imorig))) / np.sum(np.square(np.abs(imorig))) )
 


def padslice_2d(array, padslice):
     # a function that does padding or slicing depending on the pad values
     tmp = array.copy()
     
     
     if tmp.ndim == 2:
          only2dimensional = True
          tmp = tmp[:,:,np.newaxis]
     elif tmp.ndim == 3:
          only2dimensional = False
     else:
          raise ValueError
          
     
     mode='constant'
     
     if padslice[0,0]>0:
          tmp = np.pad(tmp, [  [padslice[0,0], 0], [0 ,0], [0 ,0]  ] , mode=mode)          
     elif padslice[0,0]<0:
          tmp = tmp[-padslice[0,0]:, :,:]
     else:
          pass
     
     #print(tmp.shape)
     
     if padslice[0,1]>0:
          tmp = np.pad(tmp, [  [0, padslice[0,1]], [0 ,0], [0 ,0]  ], mode=mode)
     elif padslice[0,1]<0:
          tmp = tmp[:padslice[0,1], :,:]
     else:
          pass
     
     #print(tmp.shape)
     
     if padslice[1,0]>0:
          tmp = np.pad(tmp, [  [0 ,0], [padslice[1,0], 0], [0 ,0]  ], mode=mode)
     elif padslice[1,0]<0:
          tmp = tmp[:, -padslice[1,0]:, :]
     else:
          pass
     
     #print(tmp.shape)
     
     if padslice[1,1]>0:
          tmp = np.pad(tmp, [  [0 ,0],   [0, padslice[1,1]], [0 ,0]  ], mode=mode)
     elif padslice[1,1]<0:
          tmp = tmp[:, :padslice[1,1], :]
     else:
          pass
     
     print(tmp.shape)
     
     if only2dimensional:
          tmp = tmp[:,:,0]
     
     return tmp


def resizebatch(btch, factor=0.7):
     resized=[]
     for ix in range(btch.shape[2]):
          tmpmax=np.max(btch[:,:,ix])
          tmpmin=np.min(btch[:,:,ix])
          tmp=sc.misc.imresize(btch[:,:,ix], 0.7, interp='bilinear') 
          fct=(np.max(tmp)-np.min(tmp))/(tmpmax-tmpmin)
          tmp=tmp/fct
          tmp = tmp - (np.min(tmp) - tmpmin)
          resized.append( tmp )
     
     return np.transpose( np.array(resized) , [1,2,0] )         





###############################################################################
### Set parameters
###############################################################################

#get the undersampling factor as input
parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--usfact', type=float, default=5)
args=parser.parse_args()
usfact = args.usfact


print(usfact)
if np.floor(usfact) == usfact: # if it is already an integer
     usfact = int(usfact)
print(usfact)

# set the numbr of iterations for the matrix inversions in sampling
numinversioniters = 20




###############################################################################
### Load the input image and the us pattern
###############################################################################

orim = np.load('../example_data/hcp_image/sample_hcp_image.npy')
uspat = np.load('../example_data/hcp_image/sample_uspat_r'+str(usfact)+'.npy')


###############################################################################
### First do a deep density prior (DDP) reconstruction 
###############################################################################

# get the undersampled kspace and normalize it to 99th percentile
usksp = UFT(orim,uspat, normalize=False)/np.percentile( np.abs(tUFT(UFT(orim,uspat, normalize=False),uspat, normalize=False).flatten())  ,99)

# add a dummy axis for the coil dimension
usksp=usksp[:,:,np.newaxis]


# set some parameters of the reconstruction
regtype='reg2_dc'
reg=0.1
dcprojiter=10
chunks40=True
ndims=28
lat_dim=60
mode = 'MRIunproc'

numiter = 602

# reconstruct the image if it does not exist
if os.path.exists(os.getcwd() + '/../../results/hcp/reconstruction/rec_hcp_us'+str(usfact)):
    print("KCT-info: reconstruction already exists, loading it...")
    rec_vae = pickle.load(open('../../results/hcp/reconstruction/rec_hcp_us'+str(usfact),'rb'))
else:
    rec_vae = vaerecon6.vaerecon(usksp, sensmaps=np.ones_like(usksp), dcprojiter=dcprojiter, lat_dim=60, patchsize=28, contRec='' ,parfact=25, num_iter=numiter, regiter=10, reglmb=reg, regtype=regtype, half=True, mode=mode, chunks40=chunks40)
    rec_vae = rec_vae[0]
    #save the reconstruction
    pickle.dump(rec_vae, open('../../results/hcp/reconstruction/rec_hcp_us'+str(usfact),'wb')   )

lastiter = int((np.floor(rec_vae.shape[1]/13)-2)*13)
rec = rec_vae[:,lastiter].reshape([252,308])
    
r = calc_rmse(rec*np.linalg.norm(orim)/np.linalg.norm(rec), orim)
print("KCT-info: RMSE value of recon is: " +str(r))
 

###############################################################################
### Now do the sampling - get the decoder output samples x ~ p(x|z)
###############################################################################

# get the undersampled k-space again
if uspat.ndim == 3: uspat=uspat[:,:,0]
usksp = UFT(orim,uspat, normalize=True)/np.percentile( np.abs(tUFT(UFT(orim,uspat, normalize=True),uspat, normalize=True).flatten())  ,99)
usksp=usksp[:,:,np.newaxis]

# get the DDP reconstruction, i.e. the MAP estimate
lastiter = int((np.floor(rec_vae.shape[1]/13)-2)*13)
maprecon = rec_vae[:,lastiter].reshape([252, 308]) 


# get the values for padding later
imgsize=[252,308]
kspsize=[252,308,1] 

pad2edges00=int(np.ceil((kspsize[0]-imgsize[0])/2 ))
pad2edges01=int(np.floor((kspsize[0]-imgsize[0])/2 ))
pad2edges10=int(np.ceil((kspsize[1]-imgsize[1])/2 ))
pad2edges11=int(np.floor((kspsize[1]-imgsize[1])/2 ))

pads = -np.array( [ [pad2edges00,pad2edges01] , [pad2edges10,pad2edges11] ] )

empiricalPrior=True # False#True
lowres =  False
  

# estimate the bias field from the MAP estimate
inputImage = sitk.GetImageFromArray(np.abs(maprecon), isVector=False)
corrector = sitk.N4BiasFieldCorrectionImageFilter();
inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
output = corrector.Execute(inputImage)
N4biasfree_output = sitk.GetArrayFromImage(output)

n4biasfield = np.abs(maprecon)/(N4biasfree_output+1e-9)     

mapreconpad = padslice_2d( N4biasfree_output, pads ) # a4    

mapphase = np.exp(1j*np.angle(maprecon))   

# specify where to save the samples, samples will appear in this folder
dirname = '../../results/hcp/samples/decoder_samples'

numsamp = 10000
saveevery = 100

sampling  = vaesampling.vaesampling( usksp=usksp, sensmaps=np.ones_like(usksp), maprecon=mapreconpad, mapphase=np.exp(1j*np.angle(maprecon)) , directoryname=dirname, im_kspsize=[imgsize, kspsize], model_prec_val=50, numinversioniters=numinversioniters, empiricalPrior = empiricalPrior, lowres=lowres, BFcorr=True, biasfield = n4biasfield, numsamp = numsamp, saveevery = saveevery, nsksp_manual=-1)




###############################################################################
### Now do the proper sampling - x ~ p(x|y, z) using the decoder output from 
### the previous step
###############################################################################

# define the necessary functions
def cdp(a, b):
     # complex dot product
     return np.sum(a*np.conj(b))


def funmin_cg_ksp(mux, uspat, nsksp, nssx, y, numiter = 10):
    # function running the conjugate gradient matrix inversion
    y=y[:,:,0]
    
    def A(m):
        return FT(n4biasfield*n4biasfield*(1/nssx)*tFT(m, normalize=True), normalize=True)  + (1/nsksp)*uspat*m
           
    its = np.zeros([numiter+1,252,308], dtype=np.complex128)
    
    b =  (1/nssx)*FT(n4biasfield*n4biasfield*n4biasfield*mapphase*mux, normalize=True) + (1/nsksp)*uspat*y 
    its[0,:,:] = b.copy() 
    r = b - A(its[0,:,:])
    p = r.copy()

    errtmps = []
    alphas = []
    
    for ix in range(numiter): # the CG iterations
        rTr_curr = cdp(r,r) # np.sum(r*r)
        alpha = rTr_curr / cdp(p, A(p)) # np.sum(p*A(p, uspat))
        alphas.append(alpha)
     
        its[ix+1,:, :] = its[ix,:, :] + alpha* p
        
        errtmp = np.linalg.norm(A(its[ix,:,:]) - b) / np.linalg.norm(b)*100
        errtmps.append(errtmp)
     
        r = r - alpha*A(p)
        beta = cdp(r,r)/rTr_curr
        p = r +beta * p
     
    return its, errtmps, alphas, b

#now take the images as mean x from p(x|y,z)  
    
# estimate the noise in k-space coilwise
estim_ksp_ns = 1/usksp[0:20,int(np.floor(usksp.shape[1]/2))-5:int(np.floor(usksp.shape[1]/2))+5,0].var()/50
nssx=1


num_of_files = int(numsamp/saveevery)

for ix in range(num_of_files):
    print('reading and processing file '+str(ix+1)+'/'+str(num_of_files))
    aa = np.load(  '../../results/hcp/samples/decoder_samples/samples_'+str(ix+1)+'.npz'   )
   
    ims = aa['ims'] # the decoder output sample
    
    ress = np.zeros_like(ims)
    for ixim in range(ims.shape[0]):
        print('processing sample '+str(int(ixim)+1)+'/'+str(int(ims.shape[0])))
        res = funmin_cg_ksp(ims[ixim], uspat, 1/estim_ksp_ns, 1/nssx, usksp , numiter=10)[0][-1]
        res = np.abs(tFT(res, normalize=True))
        ress[ixim] = res
    
    np.save('../../results/hcp/samples/samples_'+str(ix+1), ress)
    
    

###############################################################################
### Now look at these samples
###############################################################################
num_of_files = int(numsamp/saveevery)


samps = []
for ix in range(num_of_files):
    samps.append(np.load('../../results/hcp/samples/samples_'+str(ix+1)+'.npy'))

samps = np.array(samps)
samps = np.reshape(samps, [-1, 252, 308])


mean_samples = samps.mean(axis = 0)
std_samples = samps.std(axis = 0)

plt.figure();
for ix in range(50):
    rr = np.random.randint(0,samps.shape[0])
    plt.imshow(np.abs(samps[rr]));plt.title('Sample no: ' + str(rr))
    plt.pause(0.1)
    
plt.figure();plt.imshow(np.abs(orim));plt.title('Original image')
plt.figure();plt.imshow(np.abs(rec_vae[:,0].reshape([252,308])));plt.title('Zero-filled image')
plt.figure();plt.imshow(np.abs(mean_samples));plt.title('Mean of samples')
plt.figure();plt.imshow(np.abs(std_samples));plt.title('Std of samples')
















