import numpy as np
import pickle
import vaerecon5

import os, sys

from US_pattern import US_pattern
     
     
import argparse



###############################################################################
### This script runs on an example slice from the in-house measured
### dataset.  It runs
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
if not os.path.exists(os.getcwd() + '/../../results/reconstruction/'):
    os.mkdir(os.getcwd() + '/../../results/reconstruction/')
    
if not os.path.exists(os.getcwd() + '/../../results/sampling/'):
    os.mkdir(os.getcwd() + '/../../results/sampling/')
    
if not os.path.exists(os.getcwd() + '/../../results/sampling/decoder_samples/'):
    os.mkdir(os.getcwd() + '/../../results/sampling/decoder_samples/')


#bases = [ "sess_23_05_2018/EK/",
#"sess_26_07_2018/VA/",
#"sess_26_07_2018/KE/",
#"sess_02_07_2018/CK/",
#"sess_02_07_2018/TC/",
#"sess_04_06_2018/KT/",
#"sess_27_06_2018/FP/"]

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--sli', type=int, default=3) 
parser.add_argument('--base', default="sess_26_07_2018/VA/")  #    #sess_23_05_2018/EK/ # sess_02_07_2018/CK/
parser.add_argument('--usfact', type=float, default=2) 
parser.add_argument('--contrun', type=int, default=0) 
parser.add_argument('--skiprecon', type=int, default=1) 
parser.add_argument('--runsampling', type=int, default=0) 
parser.add_argument('--runsampling_step2', type=int, default=1) 
parser.add_argument('--prewhitening', type=int, default=1) 
parser.add_argument('--dcprojiter', type=int, default=1)  
parser.add_argument('--onlydciter', type=int, default=0)  
parser.add_argument('--mask4sampling', type=int, default=1)  
parser.add_argument('--numinversioniters', type=int, default=30) 
parser.add_argument('--modelprecval', type=float, default=50) 
parser.add_argument('--noisemodifier', type=float, default=1) 




args=parser.parse_args()


print(args)


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




#############################


#
#
#def FT (x):
#     #inp: [nx, ny]
#     #out: [nx, ny]
#     return np.fft.fftshift(    np.fft.fft2(  x , axes=(0,1)  ),   axes=(0,1)    ) / np.sqrt(252*308)
#
#def tFT (x):
#     #inp: [nx, ny]
#     #out: [nx, ny]
#     return np.fft.ifft2(  np.fft.ifftshift( x , axes=(0,1) ),  axes=(0,1)  )   * np.sqrt(252*308)
#
#def UFT(x, uspat):
#     #inp: [nx, ny], [nx, ny]
#     #out: [nx, ny]
#     
#     return uspat*FT(x)
#
#def tUFT(x, uspat):
#     #inp: [nx, ny], [nx, ny]
#     #out: [nx, ny]
#     return  tFT( uspat*x )


def calc_rmse(rec,imorig):
     return 100*np.sqrt(np.sum(np.square(np.abs(rec) - np.abs(imorig))) / np.sum(np.square(np.abs(imorig))) )

ndims=28
lat_dim=60

mode = 'MRIunproc'#'Melanie_BFC'

USp = US_pattern()

usfact = args.usfact

print(usfact)
if np.floor(usfact) == usfact: # if it is already an integer
     usfact = int(usfact)
print(usfact)



###################
###### RECON ######
###################

#dirbase = '/usr/bmicnas01/data-biwi-01/ktezcan/measured_data/measured2/the_h5_files/'+args.base
#sli= args.sli
#
#import h5py
#f = h5py.File(dirbase+'ddr_sl'+str(sli)+'.h5', 'r')
#ddr = np.array((f['DS1']))
#f = h5py.File(dirbase+'ddi_sl'+str(sli)+'.h5', 'r')
#ddi = np.array((f['DS1']))
#dd= ddr+1j*ddi
#dd=np.transpose(dd)
##          dd=np.transpose(dd,axes=[1,0,2])
#dd=np.rot90(np.rot90(np.rot90(dd,3),3),3)
#
#dd = np.fft.ifftn(np.fft.fftshift(np.fft.fftn(dd,axes=[0,1]),axes=[0,1]),axes=[0,1])
#
#
#f = h5py.File(dirbase+'espsi_sl'+str(sli)+'.h5', 'r')
#espsi = np.array((f['DS1']))
#f = h5py.File(dirbase+'espsr_sl'+str(sli)+'.h5', 'r')
#espsr = np.array((f['DS1']))
#esps= espsr+1j*espsi
#esps = np.transpose(esps)
##               esps=np.transpose(esps,axes=[1,0,2])
#esps=np.rot90(np.rot90(np.rot90(esps,3),3),3)
#esps=np.fft.fftshift(esps,axes=[0,1])
#sensmaps = esps.copy()
#sensmaps = np.fft.fftshift(sensmaps,axes=[0,1])
#          
#
#R=usfact
#
#ddimc = tFT(dd)
#dd=dd/np.percentile(  np.abs(ddimc).flatten()   ,99)

#################################################################################

dirbase = '/usr/bmicnas01/data-biwi-01/ktezcan/measured_data/measured2/the_h5_files/'+args.base
sli= args.sli

import h5py
f = h5py.File(dirbase+'ddr_sl'+str(sli)+'.h5', 'r')
ddr = np.array((f['DS1']))
f = h5py.File(dirbase+'ddi_sl'+str(sli)+'.h5', 'r')
ddi = np.array((f['DS1']))

dd= ddr+1j*ddi
dd=np.transpose(dd)
#          dd=np.transpose(dd,axes=[1,0,2])
dd=np.rot90(dd,3)

dd = np.fft.ifftn(np.fft.fftshift(np.fft.fftn(dd,axes=[0,1]),axes=[0,1]),axes=[0,1])



f = h5py.File(dirbase+'espsi_sl'+str(sli)+'.h5', 'r')
espsi = np.array((f['DS1']))
f = h5py.File(dirbase+'espsr_sl'+str(sli)+'.h5', 'r')
espsr = np.array((f['DS1']))

esps= espsr+1j*espsi
esps = np.transpose(esps)
#               esps=np.transpose(esps,axes=[1,0,2])
esps=np.rot90(esps,3)
esps=np.fft.fftshift(esps,axes=[0,1])
sensmaps = esps.copy()
sensmaps = np.fft.fftshift(sensmaps,axes=[0,1])

sensmaps=np.rot90(np.rot90(sensmaps))
dd=np.rot90(np.rot90(dd))
#          sensmaps=np.swapaxes(sensmaps,0,1)
#          dd=np.swapaxes(dd,0,1)



#dd = dd[:,:,6:12]
#sensmaps=sensmaps[:,:,6:12]

ddim=np.fft.ifft2(dd,axes=[0,1])


sensmaps=sensmaps/np.tile(np.sqrt(np.sum(sensmaps*np.conjugate(sensmaps),axis=2))[:,:,np.newaxis],[1, 1, sensmaps.shape[2]])

ddimc = tFT(dd)
dd=dd/np.percentile(  np.abs(ddimc).flatten()   ,99)
ddimc = tFT(dd)

          


###################################################

recon_base = '/usr/bmicnas01/data-biwi-01/ktezcan/reconsampling/MAPestimation/usz_images/rolled/'+args.base


if not os.path.exists(recon_base):
    os.makedirs(recon_base)

R=usfact

try:
     uspat = np.load(recon_base + 'uspat_us'+str(R)+'_sli'+str(sli)+'.npy')
     print("Read from existing u.s. pattern file")
except:
     uspat = USp.generate_opt_US_pattern_1D(dd.shape[0:2], R=R, max_iter=100, no_of_training_profs=15)
     np.save(recon_base + 'uspat_us'+str(R)+'_sli'+str(sli), uspat)
     print("Generated a new u.s. pattern file")
     
     
usksp = dd*np.tile(uspat[:,:,np.newaxis], [1, 1, dd.shape[2]])


if args.prewhitening:
     coil_cov = np.cov(usksp[0:20,int(np.floor(usksp.shape[1]/2))-5:int(np.floor(usksp.shape[1]/2))+5,:].reshape(-1,usksp.shape[2]).T)
#     coil_cov = np.diag(np.diag(coil_cov))
     coil_cov_chol = np.linalg.cholesky(coil_cov)
     
     #np.isclose(coil_cov, np.dot(coil_cov_chol, coil_cov_chol.T.conj())).all()
     
     coil_cov_chol_inv = np.linalg.inv(coil_cov_chol)
     
#     coil_cov_chol_inv = np.eye(13)
     
     
     
     
     usksp_tmp = usksp.reshape([-1, dd.shape[2]]).T
     mlt = np.tensordot(coil_cov_chol_inv, usksp_tmp, axes=[1,0])
     usksp_prew = mlt.T.reshape([dd.shape[0], dd.shape[1], dd.shape[2]])
     
     


     coil_cov_usksp_prew = np.cov(usksp_prew[0:20,int(np.floor(usksp_prew.shape[1]/2))-5:int(np.floor(usksp_prew.shape[1]/2))+5,:].reshape(-1,usksp_prew.shape[2]).T)
     
     #otherwise there is a problem
     assert(np.isclose(coil_cov_usksp_prew, np.eye(coil_cov_usksp_prew.shape[0])).all())
#     
#     usksp_prew_tmp = usksp_prew.reshape([-1, 13]).T
#     mlt_prew = np.tensordot(coil_cov_chol_inv.T.conj(), usksp_prew_tmp, axes=[1,0])
#     usksp_prew_prew = mlt_prew.T.reshape([237, 256, 13])
#     

     
     sensmaps_tmp = sensmaps.reshape([-1,  dd.shape[2]]).T
     sensmaps_mlt = np.tensordot(coil_cov_chol_inv, sensmaps_tmp, axes=[1,0])
     sensmaps_prew = sensmaps_mlt.T.reshape([dd.shape[0], dd.shape[1], dd.shape[2]])  
     
     sensmaps_prew=sensmaps_prew/np.tile(np.sqrt(np.sum(sensmaps_prew*np.conjugate(sensmaps_prew),axis=2))[:,:,np.newaxis],[1, 1, sensmaps_prew.shape[2]])
     sensmaps = sensmaps_prew.copy()
     
     dd_tmp = dd.reshape([-1,  dd.shape[2]]).T
     dd_mlt = np.tensordot(coil_cov_chol_inv, dd_tmp, axes=[1,0])
     dd_prew = dd_mlt.T.reshape([dd.shape[0], dd.shape[1], dd.shape[2]])  
     ddimc_prew = tFT(dd_prew)
     dd_prew=dd_prew/np.percentile(  np.abs(ddimc_prew).flatten()   ,99)
     ddimc_prew = tFT(dd_prew)

     usksp_prew_img = tFT(usksp_prew)
     usksp_prew=usksp_prew/np.percentile(  np.abs(usksp_prew_img).flatten()   ,99)
     usksp_prew_img  = tFT(usksp_prew)
     
     coil_cov_chol_inv = []

#     plt.figure();plt.imshow((np.abs(coil_cov_usksp_prew)))
else:
     coil_cov_chol_inv = []
     usksp_prew = usksp.copy()
     sensmaps_prew=sensmaps.copy()

if R<=3:
     num_iter = 602 # 302
else:
     num_iter = 602 # 602
     
num_iter = 202
     
regtype='reg2'
reg=0
dcprojiter=args.dcprojiter
onlydciter=args.onlydciter
chunks40=True
mode = 'MRIunproc'

     
if not args.skiprecon:
    
     
     #rec_vae = vaerecon5.vaerecon(usksp, sensmaps=sensmaps, dcprojiter=10, onlydciter=onlydciter, lat_dim=lat_dim, patchsize=ndims, parfact=20, num_iter=302, rescaled = rescaled, half=half,regiter=15, reglmb=reg, regtype=regtype)
               
     rec_vae = vaerecon5.vaerecon(         usksp_prew, sensmaps=sensmaps_prew, dcprojiter=dcprojiter,   onlydciter=onlydciter,                     lat_dim=lat_dim, patchsize=ndims, contRec='' ,parfact=25, num_iter=num_iter,                                  regiter=10, reglmb=reg, regtype=regtype, half=True, mode=mode, chunks40=chunks40)
     pickle.dump(rec_vae[0], open(recon_base + 'rec_us'+str(R)+'_sli'+str(sli)+'_dcprojiter_'+str(dcprojiter)+'_onlydciter_'+str(onlydciter)+'_prew'+str(args.prewhitening) ,'wb')   )
     rec_vae = rec_vae[0]

else:
     rec_vae = pickle.load( open(recon_base +  'rec_us'+str(R)+'_sli'+str(sli)+'_dcprojiter_'+str(dcprojiter)+'_onlydciter_'+str(onlydciter)+'_prew'+str(args.prewhitening) ,'rb'))



######################
###### SAMPLING ######

if args.runsampling==1:
     
     import vaesampling
     import os     
     
     # Now redefine the FFT functions to make the Ft unitary:
#     del FT
#     del tFT
#     del UFT
#     del tUFT
          
#     def FT (x):
#          #inp: [nx, ny]
#          #out: [nx, ny, ns]
#          return np.fft.fftshift(    np.fft.fft2( sensmaps*np.tile(x[:,:,np.newaxis],[1,1,sensmaps.shape[2]]) , axes=(0,1)  ),   axes=(0,1)    )  / np.sqrt(x.shape[0]*x.shape[1])
#     
#     def tFT (x):
#          #inp: [nx, ny, ns]
#          #out: [nx, ny]
#          
#          temp = np.fft.ifft2(  np.fft.ifftshift( x , axes=(0,1) ),  axes=(0,1)  )
#          return np.sum( temp*np.conjugate(sensmaps) , axis=2) /  np.sum(sensmaps*np.conjugate(sensmaps),axis=2)   * np.sqrt(x.shape[0]*x.shape[1])
#     
#     
#     def UFT(x, uspat):
#          #inp: [nx, ny], [nx, ny]
#          #out: [nx, ny, ns]
#          
#          return np.tile(uspat[:,:,np.newaxis],[1,1,sensmaps.shape[2]])*FT(x)
#     
#     def tUFT(x, uspat):
#          #inp: [nx, ny], [nx, ny]
#          #out: [nx, ny]
#          
#          tmp1 = np.tile(uspat[:,:,np.newaxis],[1,1,sensmaps.shape[2]])
#     #     print(x.shape)
#     #     print(tmp1.shape)
#          
#          return  tFT( tmp1*x )
     


     sampling_base = '/usr/bmicnas01/data-biwi-01/ktezcan/reconsampling/samples/usz_images/bfc/'+args.base+'R'+str(R)+'_sli'+str(args.sli)+'_prew'+str(args.prewhitening)+'_masking'+str(args.mask4sampling)
     if not os.path.exists(sampling_base):
         os.makedirs(sampling_base)
          
          
          
     
     
     
     
     recon_iter=101       
     
#     print("FOR DEBUG")
#     print(dd.shape)
#     print(dd.shape[0])
#     print(dd.shape[1])
#     print([dd.shape[0], dd.shape[1]])
#     print("----------------")
#     
#     print(rec_vae.shape)
#     print(dd.shape[0] * dd.shape[0])
     
     
#     try:
     maprecon = rec_vae[:,recon_iter].reshape([dd.shape[0], dd.shape[1]])
#     except:
#          ddsz1= dd.shape[0]
#          ddsz2= dd.shape[1]
#          tmp=rec_vae[:,recon_iter]
#          maprecon = np.reshape(tmp, (ddsz1, ddsz2))
     
          
          
     
     imgsize=[182,210] # [252,308]#[182,210]
     kspsize=[dd.shape[0], dd.shape[1], dd.shape[2]] # [238,266]
     
     
     kshval1 =  int(np.floor(kspsize[0]/2))
     kshval2 =  int(np.floor(kspsize[1]/2))
     
     pad2edges00=int(np.ceil((kspsize[0]-imgsize[0])/2 ))
     pad2edges01=int(np.floor((kspsize[0]-imgsize[0])/2 ))
     pad2edges10=int(np.ceil((kspsize[1]-imgsize[1])/2 ))
     pad2edges11=int(np.floor((kspsize[1]-imgsize[1])/2 ))
     
     pads = -np.array( [ [pad2edges00,pad2edges01] , [pad2edges10,pad2edges11] ] )
     
     
     #make a mask and find the roll value to center the brain in the FOV:
     import skimage.morphology
     eroded_mask  = skimage.morphology.binary_erosion((np.abs(maprecon)>0.3), selem=np.ones([3,3]))
     rollval = int((eroded_mask.shape[1] - np.where(eroded_mask==1)[1].max() - np.where(eroded_mask==1)[1].min())/2)
     
     mapreconpad = padslice_2d( maprecon, pads, rollval ) # a4
     
     # bias field stuff:
     import SimpleITK as sitk
     
     ddimcabs = np.abs(maprecon)
     
     inputImage = sitk.GetImageFromArray(ddimcabs, isVector=False)
     corrector = sitk.N4BiasFieldCorrectionImageFilter();
     inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
     output = corrector.Execute(inputImage)
     N4biasfree_output = sitk.GetArrayFromImage(output)
     
     n4biasfield = np.abs(maprecon)/(N4biasfree_output+1e-9)
     
     mapreconpad = padslice_2d( N4biasfree_output, pads, rollval ) # a4
     
     empiricalPrior=True
     lowres = True
     BFcorr = True
     
     numinversioniters = args.numinversioniters
     model_prec_val = args.modelprecval
     noisemodifier = args.noisemodifier
     
     #n4biasfield = np.ones(kspsize[0:2])
     
     if args.mask4sampling==1:
          mapreconpad = mapreconpad*(np.abs(mapreconpad)>0.1)
     
     
     #sampling  = vaesampling.vaesampling( usksp=usksp, sensmaps=sensmaps, maprecon=mapreconpadbfc, mapphase=np.exp(1j*np.angle(maprecon)) , directoryname=dirname, model_prec_val=50, empiricalPrior = empiricalPrior, lowres=lowres, BFcorr=True, biasfield = n4biasfield, numsamp = 10000, saveevery = 100)
     sampling  = vaesampling.vaesampling( usksp=usksp_prew, sensmaps=sensmaps_prew, maprecon=mapreconpad, mapphase=np.exp(1j*np.angle(maprecon)) , directoryname=sampling_base, im_kspsize=[imgsize, kspsize], model_prec_val=model_prec_val, numinversioniters=numinversioniters, empiricalPrior = empiricalPrior, lowres=lowres, BFcorr=BFcorr, biasfield=n4biasfield, numsamp = 10000, saveevery = 100,noisemodifier=noisemodifier, rollval = rollval)
     


if args.runsampling_step2:
    
    import os
          
    sampling_base = '/usr/bmicnas01/data-biwi-01/ktezcan/reconsampling/samples/usz_images/bfc_rolled/'+args.base+'R'+str(R)+'_sli'+str(args.sli)+'_prew'+str(args.prewhitening)+'_masking'+str(args.mask4sampling)
     
#    dirname=sampling_base+'samples/vol'+str(vol)+'_sli'+str(sli)+'_us'+str(R)+'_kspns'+str(args.kspnoisemultip)
    
    try:
         os.mkdir(dirname)
    except:
         pass # probably the directory already exists
         
    lastiter = 101 # int((np.floor(rec_vae.shape[1]/13)-2)*13)
    maprecon = rec_vae[:,lastiter].reshape([usksp_prew.shape[0], usksp_prew.shape[1]])
    imgsize=[182,210] # [252,308]# [280,336]#[182,210]
    kspsize=[usksp_prew.shape[0], usksp_prew.shape[1],1] # [238,266]
    pad2edges00=int(np.ceil((kspsize[0]-imgsize[0])/2 ))
    pad2edges01=int(np.floor((kspsize[0]-imgsize[0])/2 ))
    pad2edges10=int(np.ceil((kspsize[1]-imgsize[1])/2 ))
    pad2edges11=int(np.floor((kspsize[1]-imgsize[1])/2 ))
    
    pads = -np.array( [ [pad2edges00,pad2edges01] , [pad2edges10,pad2edges11] ] )
    
    
    import skimage.morphology
    eroded_mask  = skimage.morphology.binary_erosion((np.abs(maprecon)>0.3), selem=np.ones([3,3]))
    rollval = int((eroded_mask.shape[1] - np.where(eroded_mask==1)[1].max() - np.where(eroded_mask==1)[1].min())/2)

      
    empiricalPrior=True # False#True
    lowres =  False
      
    # bias field stuff:
    import SimpleITK as sitk
    
    ddimcabs = np.abs(maprecon)
    
    inputImage = sitk.GetImageFromArray(ddimcabs, isVector=False)
    corrector = sitk.N4BiasFieldCorrectionImageFilter();
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
    output = corrector.Execute(inputImage)
    N4biasfree_output = sitk.GetArrayFromImage(output)
    
    n4biasfield = ddimcabs/(N4biasfree_output+1e-9)   
    n4biasfieldpad = padslice_2d(n4biasfield, pads, rollval)
    
    mapreconpad = padslice_2d(maprecon, pads, rollval)

    # define the necessary functions
    def cdp(a, b):
         return np.sum(a*np.conj(b))
    
    
    def funmin_cg_ksp(mux, uspat, nsksp, nssx, y, n4biasfield, numiter = 10):
#        y=y[:,:,:]
        
#        ddimcpad = padslice_2d(ddimc_prew, pads)
#        phs = np.exp(1j*np.angle(ddimcpad))
        phs = np.exp(1j*np.angle(mapreconpad))
#        print(phs.shape)
        
        
        n4biasfieldpad = padslice_2d(n4biasfield, pads, rollval)
#        n4biasfieldpad = 1
        
        normfact = 1 #n2max/n1max
        
#        mux = normfact*mux
        
        
        def A(m):
            tmp1 = normfact*FT(n4biasfield*padslice_2d( (1/nssx)*padslice_2d( n4biasfield*tFT(normfact*m), -pads, rollval), pads, rollval))
            
            tmp2 = (1/nsksp)*np.tile(uspat[:,:,np.newaxis],y.shape[2])*m
            
            return tmp1 + tmp2
            
               
        its = np.zeros([numiter+1,y.shape[0],y.shape[1], y.shape[2]], dtype=np.complex128)
        
        b1 =   FT((1/nssx)*padslice_2d(n4biasfieldpad*n4biasfieldpad*n4biasfieldpad*phs*mux, -pads, rollval)) *normfact
        b2 =  (1/nsksp)*np.tile(uspat[:,:,np.newaxis],y.shape[2])*y 
    
        
        b= b1+b2
        
        
#        its[0,:,:] = b.copy()
         
        r = b - A(its[0,:,:])
        
    #    print("norm of starting r: " + str(np.linalg.norm(r)))
        
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
            
#        print(errtmps)
         
        return its, errtmps, alphas
    
    
    def convexsum(mux, y, uspat, nssx, nsksp):
        
#        y=y[:,:,0]
        tmp1_1 = FT(padslice_2d(n4biasfieldpad*np.exp(1j*np.angle(mapreconpad))*mux, -pads, rollval))
        tmp1_2 = y
        
#        normfact = np.array( [  np.linalg.norm(np.abs(tmp1_2[:,:,ix]))/np.linalg.norm(np.abs(tmp1_1[:,:,ix])) for ix in range(tmp1_1.shape[2]) ])
        normfact =  np.max(np.abs(tmp1_2[:,:,ix]))/np.max(np.abs(tmp1_1[:,:,ix])) 
        
        
        tmp1 = normfact*(1/nssx)*tmp1_1  + (1/nsksp)*tmp1_2
        
        tmp2 = (1/nssx)*np.ones_like(np.tile(uspat[:,:,np.newaxis],[1,1,y.shape[2]])) + (1/nsksp)*np.tile(uspat[:,:,np.newaxis],[1,1,y.shape[2]])
        
        return tmp1/tmp2
    
    #now take the images as mean f p(x|y,z)    
#    estim_ksp_ns = np.array([1/usksp[0:20,int(np.floor(usksp.shape[1]/2))-5:int(np.floor(usksp.shape[1]/2))+5,ix].var()/50 for ix in range(usksp.shape[2])])
    estim_ksp_ns = np.array([1/usksp_prew[0:20,int(np.floor(usksp.shape[1]/2))-5:int(np.floor(usksp.shape[1]/2))+5,ix].var()/50 for ix in range(usksp.shape[2])])

    nssx=1
    
    ddimcabs = np.abs(maprecon)
    inputImage = sitk.GetImageFromArray(ddimcabs, isVector=False)
    corrector = sitk.N4BiasFieldCorrectionImageFilter();
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
    output = corrector.Execute(inputImage)
    N4biasfree_output = sitk.GetArrayFromImage(output)
#        
    n4biasfield = ddimcabs/(N4biasfree_output+1e-9)   
    
    if not os.path.exists(sampling_base + '/samples_x_given_yz/'):
        print('folder to save samples does not exist...')
        os.makedirs(sampling_base + '/samples_x_given_yz/')
        print('... created one!')
 
    for ix in range(0, 100):
        print('reading and processing file '+str(ix+1)+'/100')
        print('--------------------------------------------')
        aa = np.load(  sampling_base + '/arrays_model_precision_value50_withscl_r35_scl50_sx1_empPriorTrue_step0.001_samp'+str(ix+1)+'.npz'   )
       
        ims = aa['ims']
        
        ress = []
        for ixim in range(0,ims.shape[0],10):
            print('processing sample '+str(int(ixim/10)+1)+'/'+str(int(ims.shape[0]/10)))
            res = funmin_cg_ksp(np.abs(ims[ixim]), uspat, estim_ksp_ns, nssx, usksp_prew , n4biasfield, numiter=10)[0][-1]
            res = np.abs(tFT(res))
            ress.append(res)
        
        ress = np.array(ress)
        np.save(sampling_base + '/samples_x_given_yz/samps_nsksp_'+str(ix+1), ress)
        print('saved samples!')
    
     


####################################################
####################################################
#     
#ns=10
#sx=1
#        
#def EHE(im, uspat):
#         
#     return ns*tUFT(UFT(im,uspat),uspat) 
#     
#def AHA (im, uspat):
#    
#    tmp1 = im*sx*sx
#    tmp2 = sx*EHE(im,uspat)
#    tmp3 = sx*EHE(im,uspat)
#    tmp4 = EHE(EHE(im, uspat),uspat)         
#    return tmp1+tmp2+tmp3+tmp4         
#     
#def A(im, uspat):
#    return im*sx + EHE(im,uspat)  
#
#def grad(im, mu, uspat):
#    return AHA(im,uspat) - A(sx*mu, uspat)
#    
#def error(im, mu, uspat):
#     return np.linalg.norm(A(im, uspat) - sx*mu)
#    
#     
#def findgamma(mu, numiter = 10, alpha = 1e-4):  
#               
#     its = np.zeros([numiter+1,237,256], dtype=np.complex128)
#
#     mu = np.abs(mu)
#     
#     
#     its[0,:,:] = mu.copy()
#     
#     
#     errtmps = []
#     
#     for ix in range(numiter):
#          grdtmp = grad(its[ix,:,:], mu, uspat)
#          its[ix+1,:,:] =  its[ix,:,:] - alpha*grdtmp      
#          errtmp = np.linalg.norm(A(its[ix,:,:], uspat) - sx*mu) / np.linalg.norm(mu)*100
#          errtmps.append( errtmp )
#          print("iter: "+str(ix+1)+" norm grad: " + str(np.linalg.norm(grdtmp)) + " error (%): " +str(errtmp))
#     
#     return its[-1,:,:], errtmps
#
#
#alphaval=0.0045
#it, err = findgamma(ddimc, numiter = 750, alpha = alphaval)
##errs = []
##for alpix, alphaval in enumerate(np.linspace(0.016,0.018, 15)):
##     print(alpix)
##     it, err = findgamma(ddimc, numiter = 50, alpha = alphaval)
##     errs.append((alphaval, err[-1]))











#plot rmses
     
####################################################
####################################################
     
####################################################
####################################################
     
####################################################
####################################################

analyse = False  
if analyse:
         
     usfact = 3
     R=3
     
     
     base = 'sess_23_05_2018/EK/'
     recon_base = '/usr/bmicnas01/data-biwi-01/ktezcan/reconsampling/MAPestimation/usz_images/'+base
     
     
     dirbase = '/usr/bmicnas01/data-biwi-01/ktezcan/measured_data/measured2/the_h5_files/'+base
     sli= args.sli
     
     import h5py
     f = h5py.File(dirbase+'ddr_sl'+str(sli)+'.h5', 'r')
     ddr = np.array((f['DS1']))
     f = h5py.File(dirbase+'ddi_sl'+str(sli)+'.h5', 'r')
     ddi = np.array((f['DS1']))
     
     dd= ddr+1j*ddi
     dd=np.transpose(dd)
     #          dd=np.transpose(dd,axes=[1,0,2])
     dd=np.rot90(dd,3)
     
     dd = np.fft.ifftn(np.fft.fftshift(np.fft.fftn(dd,axes=[0,1]),axes=[0,1]),axes=[0,1])
     
     
     
     f = h5py.File(dirbase+'espsi_sl'+str(sli)+'.h5', 'r')
     espsi = np.array((f['DS1']))
     f = h5py.File(dirbase+'espsr_sl'+str(sli)+'.h5', 'r')
     espsr = np.array((f['DS1']))
     
     esps= espsr+1j*espsi
     esps = np.transpose(esps)
     #               esps=np.transpose(esps,axes=[1,0,2])
     esps=np.rot90(esps,3)
     esps=np.fft.fftshift(esps,axes=[0,1])
     sensmaps = esps.copy()
     sensmaps = np.fft.fftshift(sensmaps,axes=[0,1])
     
     sensmaps=np.rot90(np.rot90(sensmaps))
     dd=np.rot90(np.rot90(dd))
     #          sensmaps=np.swapaxes(sensmaps,0,1)
     #          dd=np.swapaxes(dd,0,1)
     
     
     
     #dd = dd[:,:,6:12]
     #sensmaps=sensmaps[:,:,6:12]
     
     ddim=np.fft.ifft2(dd,axes=[0,1])
     
     
     sensmaps=sensmaps/np.tile(np.sqrt(np.sum(sensmaps*np.conjugate(sensmaps),axis=2))[:,:,np.newaxis],[1, 1, sensmaps.shape[2]])
     
     ddimc = tFT(dd)
     dd=dd/np.percentile(  np.abs(ddimc).flatten()   ,99)
     ddimc = tFT(dd)
     
     
     
     try:
          uspat = np.load(recon_base + 'uspat_us'+str(R)+'_sli'+str(sli)+'.npy')
          print("Read from existing u.s. pattern file")
     except:
          uspat = USp.generate_opt_US_pattern_1D(dd.shape[0:2], R=R, max_iter=100, no_of_training_profs=15)
          np.save(recon_base + 'uspat_us'+str(R)+'_sli'+str(sli), uspat)
          print("Generated a new u.s. pattern file")
          
          
     usksp = dd*np.tile(uspat[:,:,np.newaxis], [1, 1, dd.shape[2]])
     
     
     if True:
          coil_cov = np.cov(usksp[0:20,int(np.floor(usksp.shape[1]/2))-5:int(np.floor(usksp.shape[1]/2))+5,:].reshape(-1,usksp.shape[2]).T)
     #     coil_cov = np.diag(np.diag(coil_cov))
          coil_cov_chol = np.linalg.cholesky(coil_cov)
          
          #np.isclose(coil_cov, np.dot(coil_cov_chol, coil_cov_chol.T.conj())).all()
          
          coil_cov_chol_inv = np.linalg.inv(coil_cov_chol)
          
     #     coil_cov_chol_inv = np.eye(13)
          
          
          
          
          usksp_tmp = usksp.reshape([-1, dd.shape[2]]).T
          mlt = np.tensordot(coil_cov_chol_inv, usksp_tmp, axes=[1,0])
          usksp_prew = mlt.T.reshape([dd.shape[0], dd.shape[1], dd.shape[2]])
          
          
     
     
          coil_cov_usksp_prew = np.cov(usksp_prew[0:20,int(np.floor(usksp_prew.shape[1]/2))-5:int(np.floor(usksp_prew.shape[1]/2))+5,:].reshape(-1,usksp_prew.shape[2]).T)
          
          #otherwise there is a problem
          assert(np.isclose(coil_cov_usksp_prew, np.eye(coil_cov_usksp_prew.shape[0])).all())
     #     
     #     usksp_prew_tmp = usksp_prew.reshape([-1, 13]).T
     #     mlt_prew = np.tensordot(coil_cov_chol_inv.T.conj(), usksp_prew_tmp, axes=[1,0])
     #     usksp_prew_prew = mlt_prew.T.reshape([237, 256, 13])
     #     
     
          
          sensmaps_tmp = sensmaps.reshape([-1,  dd.shape[2]]).T
          sensmaps_mlt = np.tensordot(coil_cov_chol_inv, sensmaps_tmp, axes=[1,0])
          sensmaps_prew = sensmaps_mlt.T.reshape([dd.shape[0], dd.shape[1], dd.shape[2]])  
          
          sensmaps_prew=sensmaps_prew/np.tile(np.sqrt(np.sum(sensmaps_prew*np.conjugate(sensmaps_prew),axis=2))[:,:,np.newaxis],[1, 1, sensmaps_prew.shape[2]])
          sensmaps = sensmaps_prew.copy()
          
          dd_tmp = dd.reshape([-1,  dd.shape[2]]).T
          dd_mlt = np.tensordot(coil_cov_chol_inv, dd_tmp, axes=[1,0])
          dd_prew = dd_mlt.T.reshape([dd.shape[0], dd.shape[1], dd.shape[2]])  
          ddimc_prew = tFT(dd_prew)
          dd_prew=dd_prew/np.percentile(  np.abs(ddimc_prew).flatten()   ,99)
          ddimc_prew = tFT(dd_prew)
     
          usksp_prew_img = tFT(usksp_prew)
          usksp_prew=usksp_prew/np.percentile(  np.abs(usksp_prew_img).flatten()   ,99)
          usksp_prew_img  = tFT(usksp_prew)
          
          coil_cov_chol_inv = []
     
     #     plt.figure();plt.imshow((np.abs(coil_cov_usksp_prew)))
     else:
          coil_cov_chol_inv = []
          usksp_prew = usksp.copy()
          sensmaps_prew=sensmaps.copy()
                                   
     
        
     num_iter = 802
          
     regtype='reg2'
     reg=0
     chunks40=True
     mode = 'MRIunproc'
     
     
     
     rmses=np.zeros([2,2,2,num_iter])
     for ix1, dcprojiter in enumerate([10, 1]):
          for ix2, onlydciter in enumerate([0, 10]):
               for ix3, prew in enumerate([0, 1]):
                                   
                    
                   
                                   
                    if prew ==0:
                         im2comp = ddimc
                    else:
                         im2comp = ddimc_prew
                    
                    
                    try:
                         print(dcprojiter, onlydciter, prew)
                         rec_vae = pickle.load( open(recon_base +  'rec_us'+str(R)+'_sli'+str(sli)+'_dcprojiter_'+str(dcprojiter)+'_onlydciter_'+str(onlydciter)+'_prew'+str(prew) ,'rb'))
                    except:
                         print("could not load the file")
                    rec_vae = rec_vae.reshape([ddimc.shape[0], ddimc.shape[1], num_iter])
                    
                    mask = np.abs(ddimc_prew)>0.1
                    
                    for ix in range(num_iter):
                         rmses[ix1, ix2, ix3, ix] = calc_rmse(np.abs(mask*rec_vae[:,:,ix])*np.linalg.norm(mask*np.abs(im2comp))/np.linalg.norm(mask*np.abs(rec_vae[:,:,ix])), mask*np.abs(im2comp))
          
     plt.figure();
     for ix1, dcprojiter in enumerate([10, 1]):
          for ix2, onlydciter in enumerate([0, 10]):
               for ix3, prew in enumerate([0, 1]):
                    plt.plot(rmses[ix1,ix2,ix3,:]);
     plt.legend(['dc_proj: 10 only_dc: 0 pre-whit: 0', 
     'dc_proj: 10 only_dc: 0 pre-whit: 1',  
     'dc_proj: 10 only_dc: 10 pre-whit: 0',  
     'dc_proj: 10 only_dc: 10 pre-whit: 1',  
     'dc_proj: 1 only_dc: 0 pre-whit: 0',  
     'dc_proj: 1 only_dc: 0 pre-whit: 1',  
     'dc_proj: 1 only_dc: 10 pre-whit: 0',  
     'dc_proj: 1 only_dc: 10 pre-whit: 1' ], prop={'size': 26})                    
     
     
     for ix1, dcprojiter in enumerate([10, 1]):
          for ix2, onlydciter in enumerate([0, 10]):
               for ix3, prew in enumerate([0, 1]):
                    print("'dc_proj: "+str(dcprojiter)+" only_dc: "+str(onlydciter)+" pre-whit: "+str(prew)+"',\ ")
     
     
     
     
