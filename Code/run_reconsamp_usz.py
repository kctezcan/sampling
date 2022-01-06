import numpy as np
import pickle
import vaerecon5
import vaesampling
import SimpleITK as sitk

import os, sys     
import argparse



###############################################################################
### This script runs on an example slice from the in-house measured
### dataset.  It runs
### 1. a reconstruction based on the deep density priors (DDP) method used to 
###    initialize the sampling.
### 2. the sampling method with the output of the VAE decoder as the samples,
### 3. the samples from the proper posterior
############################################################################### 

run_decoder_sampling = False
run_proper_sampling = True
visualize = True


###############################################################################
### Get the directory of the script to access files later
###############################################################################
scriptdir = os.path.dirname(sys.argv[0])+'/'
print("KCT-info: running script from directory: " + scriptdir)
os.chdir(scriptdir)

#make the necessary folders if not existing
if not os.path.exists(os.getcwd() + '/../../results/usz/reconstruction/'):
    os.mkdir(os.getcwd() + '../../results/usz/reconstruction/')
    
if not os.path.exists(os.getcwd() + '/../../results/usz/samples/'):
    os.mkdir(os.getcwd() + '../../results/usz/samples/')
    
if not os.path.exists(os.getcwd() + '/../../results/usz/samples/decoder_samples/'):
    os.mkdir(os.getcwd() + '../../results/usz/samples/decoder_samples/')



###############################################################################
### Some necessary functions
###############################################################################

def FT (x):
     #inp: [nx, ny]
     #out: [nx, ny, ns]
     return np.fft.fftshift(    np.fft.fft2( sensmaps*np.tile(x[:,:,np.newaxis],[1,1,sensmaps.shape[2]]) , axes=(0,1)  ),   axes=(0,1)    ) #  / np.sqrt(x.shape[0]*x.shape[1])

def tFT (x):
     #inp: [nx, ny, ns]
     #out: [nx, ny]
     
     temp = np.fft.ifft2(  np.fft.ifftshift( x , axes=(0,1) ),  axes=(0,1)  )
     return np.sum( temp*np.conjugate(sensmaps) , axis=2)  / np.sum(sensmaps*np.conjugate(sensmaps),axis=2) #  * np.sqrt(x.shape[0]*x.shape[1])


def UFT(x, uspat):
     #inp: [nx, ny], [nx, ny]
     #out: [nx, ny, ns]
     
     return np.tile(uspat[:,:,np.newaxis],[1,1,sensmaps.shape[2]])*FT(x)

def tUFT(x, uspat):
     #inp: [nx, ny], [nx, ny]
     #out: [nx, ny]
     
     tmp1 = np.tile(uspat[:,:,np.newaxis],[1,1,sensmaps.shape[2]])
#     print(x.shape)
#     print(tmp1.shape)
     
     return  tFT( tmp1*x )


def padslice_2d(array, padslice, rollval = 0):
          # a function that does padding or slicing depending on the pad values
          tmp = array.copy()
          
          
          if tmp.ndim == 2:
               only2dimensional = True
               tmp = tmp[:,:,np.newaxis]
          elif tmp.ndim == 3:
               only2dimensional = False
          else:
               raise ValueError
               
          if rollval!=0:
              if padslice[1,0]<0 and padslice[1,1]<0:
                  tmp = np.roll(tmp, rollval, axis=1)
                 
          
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
          
#          print(tmp.shape)
               
           
          if rollval!=0:
              if padslice[1,0]>0 and padslice[1,1]>0:
                  tmp = np.roll(tmp, -rollval, axis=1)
          
          if only2dimensional:
               tmp = tmp[:,:,0]
          
          return tmp

def calc_rmse(rec,imorig):
     return 100*np.sqrt(np.sum(np.square(np.abs(rec) - np.abs(imorig))) / np.sum(np.square(np.abs(imorig))) )


###############################################################################
### Set parameters
###############################################################################
parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--usfact', type=float, default=3) 

args=parser.parse_args()
print(args)     
 
ndims=28
lat_dim=60

mode = 'MRIunproc'#'Melanie_BFC'

usfact = args.usfact

if np.floor(usfact) == usfact: # if it is already an integer
     usfact = int(usfact)



###############################################################################
### Load the input image and the us pattern
###############################################################################

#the fully sampled k-space and the coils from ESPIRiT
dd = np.load(os.getcwd() + '/../example_data/usz_image/sample_usz_image.npy')
sensmaps = np.load(os.getcwd() + '/../example_data/usz_image/sample_usz_coils.npy')

#normalize the coils and the k-space
sensmaps=sensmaps/np.tile(np.sqrt(np.sum(sensmaps*np.conjugate(sensmaps),axis=2))[:,:,np.newaxis],[1, 1, sensmaps.shape[2]])

ddimc = tFT(dd)
dd=dd/np.percentile(  np.abs(ddimc).flatten()   ,99)

#ddimc containst the fully sampled coil combined inverse Fourier transform
ddimc = tFT(dd)

uspat = np.load(os.getcwd() + '/../example_data/usz_image/sample_uspat_r'+str(usfact)+'.npy')
print("KCT-info: Read from existing u.s. pattern file")

#get the undersampled k-space     
usksp = dd*np.tile(uspat[:,:,np.newaxis], [1, 1, dd.shape[2]])


###############################################################################
### do the prewhitening of the data and the coils
###############################################################################
prewhitening = True
if prewhitening:
    
     #calculate the coil covariance matrix from the outer parts of the k-space and take its Cholesky decomposition
     coil_cov = np.cov(usksp[0:20,int(np.floor(usksp.shape[1]/2))-5:int(np.floor(usksp.shape[1]/2))+5,:].reshape(-1,usksp.shape[2]).T)
     coil_cov_chol = np.linalg.cholesky(coil_cov)
     coil_cov_chol_inv = np.linalg.inv(coil_cov_chol)
     
     #transform the undersampled k-space wit hthe inverse Cholsky decomposition
     usksp_tmp = usksp.reshape([-1, dd.shape[2]]).T
     mlt = np.tensordot(coil_cov_chol_inv, usksp_tmp, axes=[1,0])
     usksp_prew = mlt.T.reshape([dd.shape[0], dd.shape[1], dd.shape[2]])
     
     #check if the transformed k-space has a identity covariance, otherwise abort
     coil_cov_usksp_prew = np.cov(usksp_prew[0:20,int(np.floor(usksp_prew.shape[1]/2))-5:int(np.floor(usksp_prew.shape[1]/2))+5,:].reshape(-1,usksp_prew.shape[2]).T)
     assert(np.isclose(coil_cov_usksp_prew, np.eye(coil_cov_usksp_prew.shape[0])).all())

     # transform the coil maps as well
     sensmaps_tmp = sensmaps.reshape([-1,  dd.shape[2]]).T
     sensmaps_mlt = np.tensordot(coil_cov_chol_inv, sensmaps_tmp, axes=[1,0])
     sensmaps_prew = sensmaps_mlt.T.reshape([dd.shape[0], dd.shape[1], dd.shape[2]])  
     
     # normalize again just in case
     sensmaps_prew=sensmaps_prew/np.tile(np.sqrt(np.sum(sensmaps_prew*np.conjugate(sensmaps_prew),axis=2))[:,:,np.newaxis],[1, 1, sensmaps_prew.shape[2]])
     sensmaps = sensmaps_prew.copy()
     
     # normalize the full k-space as well for comparison purposes
     dd_tmp = dd.reshape([-1,  dd.shape[2]]).T
     dd_mlt = np.tensordot(coil_cov_chol_inv, dd_tmp, axes=[1,0])
     dd_prew = dd_mlt.T.reshape([dd.shape[0], dd.shape[1], dd.shape[2]])  
     ddimc_prew = tFT(dd_prew)
     dd_prew=dd_prew/np.percentile(  np.abs(ddimc_prew).flatten()   ,99)
     ddimc_prew = tFT(dd_prew)
     
     # make the image from the prewhitened undersampled k-space
     usksp_prew_img = tFT(usksp_prew)
     usksp_prew=usksp_prew/np.percentile(  np.abs(usksp_prew_img).flatten()   ,99)
     usksp_prew_img  = tFT(usksp_prew)
     
     coil_cov_chol_inv = []



###############################################################################
### Set some reconstruction parameters and reconstruct if it does not exist
###############################################################################
num_iter = 202
     
regtype='reg2'
reg=0
dcprojiter=1
onlydciter=0
chunks40=True
mode = 'MRIunproc'


# reconstruct the image if it does not exist     
if os.path.exists(os.getcwd() + '/../../results/usz/reconstruction/rec_usz_us'+str(usfact)):
    print("KCT-info: reconstruction already exists, loading it...")
    rec_vae = pickle.load(open('../../results/usz/reconstruction/rec_usz_us'+str(usfact),'rb'))
else:    
    rec_vae = vaerecon5.vaerecon(usksp_prew, sensmaps=sensmaps_prew, dcprojiter=dcprojiter, onlydciter=onlydciter,lat_dim=lat_dim, patchsize=ndims, contRec='' ,parfact=25, num_iter=num_iter, regiter=10, reglmb=reg, regtype=regtype, half=True, mode=mode, chunks40=chunks40)
    pickle.dump(rec_vae[0], open(os.getcwd() + '/../../results/usz/reconstruction/rec_usz_us'+str(usfact) ,'wb')   )
    rec_vae = rec_vae[0]



###############################################################################
### Now do the sampling from the decoder output
###############################################################################
#get the MAP reonstruction first
recon_iter=101       
maprecon = rec_vae[:,recon_iter].reshape([dd.shape[0], dd.shape[1]])

#set some of the parameters 
imgsize=[182,210] # [252,308]#[182,210]
kspsize=[dd.shape[0], dd.shape[1], dd.shape[2]] # [238,266]

#calculate the padding values to use later
pad2edges00=int(np.ceil((kspsize[0]-imgsize[0])/2 ))
pad2edges01=int(np.floor((kspsize[0]-imgsize[0])/2 ))
pad2edges10=int(np.ceil((kspsize[1]-imgsize[1])/2 ))
pad2edges11=int(np.floor((kspsize[1]-imgsize[1])/2 ))
pads = -np.array( [ [pad2edges00,pad2edges01] , [pad2edges10,pad2edges11] ] )


#now we do a very coarse "registration" by simply making sure the center of tha brain is in the center of the FOV
#make a mask and find the roll value to center the brain in the FOV:
#the rollvalue is the difference of the center of the FOV to the center of the brain, found as the leftmost and rightmost
#points of the brain mask
import skimage.morphology
eroded_mask  = skimage.morphology.binary_erosion((np.abs(maprecon)>0.3), selem=np.ones([3,3]))
rollval = int((eroded_mask.shape[1] - np.where(eroded_mask==1)[1].max() - np.where(eroded_mask==1)[1].min())/2)

#obtain the padded and centered brain
mapreconpad = padslice_2d( maprecon, pads, rollval ) # a4

# estimate the biasfield from the MAP estimate
inputImage = sitk.GetImageFromArray(np.abs(maprecon), isVector=False)
corrector = sitk.N4BiasFieldCorrectionImageFilter();
inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
output = corrector.Execute(inputImage)
N4biasfree_output = sitk.GetArrayFromImage(output)
n4biasfield = np.abs(maprecon)/(N4biasfree_output+1e-9)

#obtain the padded and centered and biasfield corrected brain
mapreconpad = padslice_2d( N4biasfree_output, pads, rollval ) # a4

#set some parameters for the sampling
empiricalPrior=True
lowres = True
BFcorr = True

numinversioniters = 30
model_prec_val = 50
noisemodifier = 1
mask4sampling = True
numsamp = 10000
saveevery = 100

if mask4sampling:
    mapreconpad = mapreconpad*(np.abs(mapreconpad)>0.1)
   
#where to save the samples    
dir4decodersamples = os.getcwd() + '/../../results/usz/samples/decoder_samples/'

if run_decoder_sampling: 
    #run the sampling
    sampling  = vaesampling.vaesampling( usksp=usksp_prew, sensmaps=sensmaps_prew, maprecon=mapreconpad, mapphase=np.exp(1j*np.angle(maprecon)) , directoryname=dir4decodersamples, im_kspsize=[imgsize, kspsize], model_prec_val=model_prec_val, numinversioniters=numinversioniters, empiricalPrior = empiricalPrior, lowres=lowres, BFcorr=BFcorr, biasfield=n4biasfield, numsamp = numsamp, saveevery = saveevery, noisemodifier=noisemodifier, rollval = rollval)


###############################################################################
### Now do the proper sampling from p(x\y,z)
###############################################################################
dir4samples = os.getcwd() + '/../../results/usz/samples/'

#get the padded bias field  
n4biasfieldpad = padslice_2d(n4biasfield, pads, rollval)

#get the padded and centered brain
mapreconpad = padslice_2d(maprecon, pads, rollval)

# define the necessary functions
def cdp(a, b):
     #complex dot product
     return np.sum(a*np.conj(b))

#the Conjugate Gradient method for the matrix inversion
def funmin_cg_ksp(mux, uspat, nsksp, nssx, y, n4biasfield, numiter = 10):

    phs = np.exp(1j*np.angle(mapreconpad))
    n4biasfieldpad = padslice_2d(n4biasfield, pads, rollval)
    normfact = 1 #n2max/n1max
 
    
    def A(m):
        tmp1 = normfact*FT(n4biasfield*padslice_2d( (1/nssx)*padslice_2d( n4biasfield*tFT(normfact*m), -pads, rollval), pads, rollval))
        
        tmp2 = (1/nsksp)*np.tile(uspat[:,:,np.newaxis],y.shape[2])*m
        
        return tmp1 + tmp2
    
    its = np.zeros([numiter+1,y.shape[0],y.shape[1], y.shape[2]], dtype=np.complex128)
    
    b1 =   FT((1/nssx)*padslice_2d(n4biasfieldpad*n4biasfieldpad*n4biasfieldpad*phs*mux, -pads, rollval)) *normfact
    b2 =  (1/nsksp)*np.tile(uspat[:,:,np.newaxis],y.shape[2])*y 
    b= b1+b2
    r = b - A(its[0,:,:])
    p = r.copy()
    
    errtmps = []
    alphas = []
    
    for ix in range(numiter):
     
        rTr_curr = cdp(r,r) # np.sum(r*r)
        alpha = rTr_curr / cdp(p, A(p)) # np.sum(p*A(p, uspat))
        alphas.append(alpha)
     
        its[ix+1,:, :] = its[ix,:, :] + alpha* p
        
        errtmp = np.linalg.norm(A(its[ix,:,:]) - b) / np.linalg.norm(b)*100
        errtmps.append(errtmp)
     
        r = r - alpha*A(p)
     
        beta = cdp(r,r)/rTr_curr
     
        p = r + beta * p        
     
    return its, errtmps, alphas


if run_proper_sampling:

    #now estimate the k-space noise again 
    estim_ksp_ns = np.array([1/usksp_prew[0:20,int(np.floor(usksp.shape[1]/2))-5:int(np.floor(usksp.shape[1]/2))+5,ix].var()/50 for ix in range(usksp.shape[2])])
    nssx=1
     
    numfiles = int(numsamp/saveevery)
    
    for ix in range(numfiles):
        print('reading and processing file '+str(ix+1)+'/'+str(numfiles))
        print('--------------------------------------------')
        aa = np.load(  dir4decodersamples + 'samples_'+str(ix+1)+'.npz'   )
       
        ims = aa['ims'] #the decoder output sample
        
        ress = []
        for ixim in range(0,ims.shape[0],10): # take only one tenth of the decoder smaples to save time
            print('processing sample '+str(int(ixim/10)+1)+'/'+str(int(ims.shape[0]/10)))
            res = funmin_cg_ksp(np.abs(ims[ixim]), uspat, estim_ksp_ns, nssx, usksp_prew , n4biasfield, numiter=10)[0][-1]
            res = np.abs(tFT(res))
            ress.append(res)
        
        ress = np.array(ress)
        np.save(dir4samples + '/samples_'+str(ix+1), ress)
        print('saved samples!')




###############################################################################
### Now look at these samples
###############################################################################
if visualize:
    
    numfiles = int(numsamp/saveevery)
    
    samps = []
    for ix in range(numfiles):
        samps.append(np.load(dir4samples + 'samples_'+str(ix+1)+'.npy'))
    
    samps = np.array(samps)
    samps = np.reshape(samps, [-1, samps.shape[2], samps.shape[3]])
    
    
    mean_samples = samps.mean(axis = 0)
    std_samples = samps.std(axis = 0)
    
    plt.figure();
    for ix in range(50):
        rr = np.random.randint(0,samps.shape[0])
        plt.imshow(np.abs(samps[rr]), vmin = 0, vmax = 1.2);plt.title('Sample no: ' + str(rr))
        plt.pause(0.1)
        
    plt.figure();plt.imshow(np.abs(tFT(dd_prew)), vmin = 0, vmax = 1.2);plt.title('Original image')
    plt.figure();plt.imshow(np.abs(rec_vae[:,0].reshape([dd.shape[0], dd.shape[1]])), vmin = 0, vmax = 1.2);plt.title('Zero-filled image')
    plt.figure();plt.imshow(np.abs(mean_samples), vmin = 0, vmax = 1.2);plt.title('Mean of samples')
    plt.figure();plt.imshow(np.abs(std_samples));plt.title('Std of samples')


 


     
