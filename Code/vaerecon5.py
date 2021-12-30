# -*- coding: utf-8 -*-

# Simple VAE to see if VAEs can learn sparsity inducing distributions
# Kerem Tezcan, CVL
# initial: 28.05.2017
# last mod: 30.05.2017 

from __future__ import division
from __future__ import print_function
import numpy as np

from Patcher import Patcher
from definevae2 import definevae2

import scipy.io

import scipy.optimize as sop


import SimpleITK as sitk


import os
import subprocess
import sys

def vaerecon(us_ksp_r2, sensmaps, dcprojiter, onlydciter=0, lat_dim=60, patchsize=28, contRec='', parfact=10, num_iter=302, rescaled=False, half=False, regiter=15, reglmb=0.05, regtype='TV', usemeth=1, stepsize=1e-4, optScale=False, mode=[], chunks40=False, Melmodels='', N4BFcorr=False, z_multip=1.0, coil_cov_chol_inv=[]):
     
     
     print("KCT-info: Reg value is: " + str(reglmb))

     
     # set parameters
     #==============================================================================
     np.random.seed(seed=1)
     
     imsizer=us_ksp_r2.shape[0] #252#256#252
     imrizec=us_ksp_r2.shape[1] #308#256#308
     
     nsampl=50#0
          
     
     #make a network and a patcher to use later
     #==============================================================================
     
#     x_rec, x_inp, funop, grd0, sess, grd_p_x_z0, grd_p_z0, grd_q_z_x0, grd20, y_out, y_out_prec, z_std_multip = definevae2(lat_dim=lat_dim, patchsize=patchsize, batchsize=parfact*nsampl, rescaled=rescaled, half=half)

     x_rec, x_inp, funop, grd0, sess, grd_p_x_z0, grd_p_z0, grd_q_z_x0, grd20, y_out, y_out_prec, z_std_multip, op_q_z_x, mu, std, grd_q_zpl_x_az0, op_q_zpl_x, z_pl, z = definevae2(lat_dim=lat_dim, patchsize=patchsize, batchsize=parfact*nsampl, rescaled=rescaled, half=half, mode=mode, chunks40=chunks40,Melmodels=Melmodels)


     Ptchr=Patcher(imsize=[imsizer,imrizec],patchsize=patchsize,step=int(patchsize/2), nopartials=True, contatedges=True)
     
     
     nopatches=len(Ptchr.genpatchsizes)
     print("KCT-INFO: there will be in total " + str(nopatches) + " patches.")
     
     
     #define the necessary functions
     #==============================================================================
     
     def FT (x):
          #inp: [nx, ny]
          #out: [nx, ny, ns]
          
          return np.fft.fftshift(    np.fft.fft2( sensmaps*np.tile(x[:,:,np.newaxis],[1,1,sensmaps.shape[2]]) , axes=(0,1)  ),   axes=(0,1)    )


#          
#          if coil_cov_chol_inv == []:
#               return tmp
#          else:
#               #do the pre-whitening
#               tmp_orig_shape = tmp.shape
#               tmp = tmp.reshape([-1, sensmaps.shape[2]]).T
#               mlt_prew = np.tensordot(coil_cov_chol_inv, tmp, axes=[1,0])
#               mlt_prew = mlt_prew.T.reshape(tmp_orig_shape)
#               return mlt_prew

          
     
     def tFT (x):
          #inp: [nx, ny, ns]
          #out: [nx, ny]
          
         
            temp = np.fft.ifft2(  np.fft.ifftshift( x , axes=(0,1) ),  axes=(0,1)  )
            return np.sum( temp*np.conjugate(sensmaps) , axis=2)  / np.sum(sensmaps*np.conjugate(sensmaps),axis=2)


#           if coil_cov_chol_inv == []:
#          else:
#               #do the pre-whitening
#               tmp_orig_shape = x.shape
#               tmp = x.reshape([-1, sensmaps.shape[2]]).T
#               mlt_prew = np.tensordot(coil_cov_chol_inv.T.conj(), tmp, axes=[1,0])
#               x = mlt_prew.T.reshape(tmp_orig_shape)
#               
#               temp = np.fft.ifft2(  np.fft.ifftshift( x , axes=(0,1) ),  axes=(0,1)  )
#               return np.sum( temp*np.conjugate(sensmaps) , axis=2)  / np.sum(sensmaps*np.conjugate(sensmaps),axis=2)
               
               
          
          
     
     

#     def FT (x):
#          #inp: [nx, ny]
#          #out: [nx, ny, ns]
#          
#          return np.fft.fftshift(    np.fft.fft2( sensmaps*np.tile(x[:,:,np.newaxis],[1,1,sensmaps.shape[2]]) , axes=(0,1)  ),   axes=(0,1)    )
#     
#     def tFT (x):
#          #inp: [nx, ny, ns]
#          #out: [nx, ny]
#          
#          temp = np.fft.ifft2(  np.fft.ifftshift( x , axes=(0,1) ),  axes=(0,1)  )
#          return np.sum( temp*np.conjugate(sensmaps) , axis=2)  / np.sum(sensmaps*np.conjugate(sensmaps),axis=2)
     
     
     def UFT(x, uspat):
          #inp: [nx, ny], [nx, ny]
          #out: [nx, ny, ns]
          
          return np.tile(uspat[:,:,np.newaxis],[1,1,sensmaps.shape[2]])*FT(x)
     
     def tUFT(x, uspat):
          #inp: [nx, ny], [nx, ny]
          #out: [nx, ny]
          
          tmp1 = np.tile(uspat[:,:,np.newaxis],[1,1,sensmaps.shape[2]])
          
          return  tFT( tmp1*x )
     
     def dconst(us):
          #inp: [nx, ny]
          #out: [nx, ny]
          
          return np.linalg.norm( UFT(us, uspat) - data ) **2
     
     def dconst_grad(us):
          #inp: [nx, ny]
          #out: [nx, ny]
          return 2*tUFT(UFT(us, uspat) - data, uspat)
     
     def likelihood(us):
          #inp: [parfact,ps*ps]
          #out: parfact
          
          us=np.abs(us)
          funeval = funop.eval(feed_dict={x_rec: np.tile(us,(nsampl,1)), z_std_multip: z_multip }) # ,x_inp: np.tile(us,(nsampl,1))
          #funeval: [500x1]
          funeval=np.array(np.split(funeval,nsampl,axis=0))# [nsampl x parfact x 1]
          return np.mean(funeval,axis=0).astype(np.float64)
     
     def likelihood_grad(us):
          #inp: [parfact, ps*ps]
          #out: [parfact, ps*ps]
          
          usc=us.copy()
          usabs=np.abs(us)
          
          
          grd0eval = grd0.eval(feed_dict={x_rec: np.tile(usabs,(nsampl,1)), z_std_multip: z_multip }) # ,x_inp: np.tile(usabs,(nsampl,1))
          
          #grd0eval: [500x784]
          grd0eval=np.array(np.split(grd0eval,nsampl,axis=0))# [nsampl x parfact x 784]
          grd0m=np.mean(grd0eval,axis=0) #[parfact,784]

          grd0m = usc/np.abs(usc)*grd0m
                            

          return grd0m #.astype(np.float64)
     
     def likelihood_grad_meth3(us):
          #inp: [parfact, ps*ps]
          #out: [parfact, ps*ps]
          
          usc=us.copy()
          usabs=np.abs(us)
          
          
          mueval = mu.eval(feed_dict={x_rec: np.tile(usabs,(nsampl,1)) }) # ,x_inp: np.tile(usabs,(nsampl,1))
          
          stdeval = std.eval(feed_dict={x_rec: np.tile(usabs,(nsampl,1)) }) # ,x_inp: np.tile(usabs,(nsampl,1))
          
          zvals = mueval + np.random.rand(mueval.shape[0],mueval.shape[1])*stdeval
          
          y_outeval = y_out.eval( feed_dict={ z : zvals } )
          y_out_preceval = y_out_prec.eval( feed_dict={ z : zvals } )
          
          tmp = np.tile(usabs,(nsampl,1)) - y_outeval
          tmp =  (-1) * tmp * y_out_preceval
          
          
          
          #grd0eval: [500x784]
          grd0eval=np.array(np.split(tmp,nsampl,axis=0))# [nsampl x parfact x 784]
          grd0m=np.mean(grd0eval,axis=0) #[parfact,784]

          grd0m = usc/np.abs(usc)*grd0m
                            

          return grd0m #.astype(np.float64)
     
     def likelihood_grad_patches(ptchs):
          #inp: [np, ps, ps] 
          #out: [np, ps, ps] 
          #takes set of patches as input and returns a set of their grad.s 
          #both grads are in the positive direction
          
          shape_orig=ptchs.shape
          
          ptchs = np.reshape(ptchs, [ptchs.shape[0], -1] )
          
          grds=np.zeros([int(np.ceil(ptchs.shape[0]/parfact)*parfact), np.prod(ptchs.shape[1:])], dtype=np.complex64)
          
          extraind=int(np.ceil(ptchs.shape[0]/parfact)*parfact) - ptchs.shape[0]
          ptchs=np.pad(ptchs,( (0,extraind),(0,0)  ), mode='edge' )
          
          
          for ix in range(int(np.ceil(ptchs.shape[0]/parfact))):
               if usemeth==1:
                    grds[parfact*ix:parfact*ix+parfact,:]=likelihood_grad(ptchs[parfact*ix:parfact*ix+parfact,:]) 
               elif usemeth==3:
                    grds[parfact*ix:parfact*ix+parfact,:]=likelihood_grad_meth3(ptchs[parfact*ix:parfact*ix+parfact,:]) 
               else:
                    assert(1==0)
               
                  
          grds=grds[0:shape_orig[0],:]

          
          return np.reshape(grds, shape_orig)
     
     def likelihood_patches(ptchs):
          #inp: [np, ps, ps] 
          #out: 1
          
          fvls=np.zeros([int(np.ceil(ptchs.shape[0]/parfact)*parfact) ])
          
          extraind=int(np.ceil(ptchs.shape[0]/parfact)*parfact) - ptchs.shape[0]
          ptchs=np.pad(ptchs,[ (0,extraind),(0,0), (0,0)  ],mode='edge' )
          
          for ix in range(int(np.ceil(ptchs.shape[0]/parfact))):
               fvls[parfact*ix:parfact*ix+parfact] = likelihood(np.reshape(ptchs[parfact*ix:parfact*ix+parfact,:,:],[parfact,-1]) )
               
          fvls=fvls[0:ptchs.shape[0]]
               
          return np.mean(fvls)
     
     
     def full_gradient(image):
          #inp: [nx*nx, 1]
          #out: [nx, ny], [nx, ny]
          
          #returns both gradients in the respective positive direction.
          #i.e. must 
          
          ptchs = Ptchr.im2patches(np.reshape(image, [imsizer,imrizec]))
          ptchs=np.array(ptchs)
          
          
          grd_lik = likelihood_grad_patches(ptchs)
          grd_lik = (-1)* Ptchr.patches2im(grd_lik)
          
          grd_dconst = dconst_grad(np.reshape(image, [imsizer,imrizec]))
          
          return grd_lik + grd_dconst, grd_lik, grd_dconst
     
     
     def full_funceval(image):
          #inp: [nx*nx, 1]
          #out: [1], [1], [1]
          
          tmpimg = np.reshape(image, [imsizer,imrizec])
          
          dc = dconst(tmpimg)
     
          ptchs = Ptchr.im2patches(np.reshape(image, [imsizer,imrizec]))
          ptchs=np.array(ptchs)
          
          lik = (-1)*likelihood_patches(np.abs(ptchs))
         
          
          return lik + dc, lik, dc    
     
     def tv_proj(phs,mu=0.125,lmb=2,IT=225):
          
          phs = fb_tv_proj(phs,mu=mu,lmb=lmb,IT=IT)
          
          return phs
     
     def fgrad(im):
          imr_x = np.roll(im,shift=-1,axis=0)
          imr_y = np.roll(im,shift=-1,axis=1)
          grd_x = imr_x - im
          grd_y = imr_y - im
          
          return np.array((grd_x, grd_y))
     
     def fdivg(im):
          imr_x = np.roll(np.squeeze(im[0,:,:]),shift=1,axis=0)
          imr_y = np.roll(np.squeeze(im[1,:,:]),shift=1,axis=1)
          grd_x = np.squeeze(im[0,:,:]) - imr_x
          grd_y = np.squeeze(im[1,:,:]) - imr_y
          
          return grd_x + grd_y
     
     
     
     
     def f_st(u,lmb):
          
          uabs = np.squeeze(np.sqrt(np.sum(u*np.conjugate(u),axis=0)))
          
          tmp=1-lmb/uabs
          tmp[np.abs(tmp)<0]=0
             
          uu = u*np.tile(tmp[np.newaxis,:,:],[u.shape[0],1,1])
          
          return uu
     
     
     def fb_tv_proj(im, u0=0, mu=0.125, lmb=1, IT=15):
          
          sz = im.shape
          us=np.zeros((2,sz[0],sz[1],IT))
          us[:,:,:,0] = u0
          
          for it in range(IT-1):
               
               #grad descent step:
               tmp1 = im - fdivg(us[:,:,:,it])
               tmp2 = mu*fgrad(tmp1)
               
               tmp3 = us[:,:,:,it] - tmp2
                 
               #thresholding step:
               us[:,:,:,it+1] = tmp3 - f_st(tmp3, lmb=lmb)     
               
          #endfor     

          return im - fdivg(us[:,:,:,it+1])
     
     
     def low_pass(im):
          import scipy.ndimage as sndi
          
          filtered = sndi.gaussian_filter(im,15)
          
          return filtered
     
     def tikh_proj(usph, niter=100, alpha=0.05):
          
          
          ims = np.zeros((imsizer,imrizec,niter))
          ims[:,:,0]=usph.copy()
          for ix in range(niter-1):
              ims[:,:,ix+1] = ims[:,:,ix] + alpha*2*fdivg(fgrad(ims[:,:,ix]))
              
          return ims[:,:,-1]
     
     def reg2_proj(usph, niter=100, alpha=0.05):
          #from  Separate Magnitude and Phase Regularization via Compressed Sensing,  Feng Zhao
          
          usph=usph+np.pi
          
          ims = np.zeros((imsizer,imrizec,niter))
          ims[:,:,0]=usph.copy()
          regval = reg2eval(ims[:,:,0].flatten())
          print(regval)
          for ix in range(niter-1):
              ims[:,:,ix+1] = ims[:,:,ix] +alpha*reg2grd(ims[:,:,ix].flatten()).reshape([252,308]) # *alpha*np.real(1j*np.exp(-1j*ims[:,:,ix])*    fdivg(fgrad(np.exp(  1j* ims[:,:,ix]    )))     )
              regval = reg2eval(ims[:,:,ix+1].flatten())
#              print(regval)
          
          return ims[:,:,-1]-np.pi     
     
     def reg2eval(im):
          #takes in 1d, returns scalar
          im=im.reshape([252,308])
          phs = np.exp(1j*im)
          return np.linalg.norm(fgrad(phs).flatten())
     
     def reg2grd(im):
          #takes in 1d, returns 1d
          im=im.reshape([252,308])
          return -2*np.real(1j*np.exp(-1j*im) *  fdivg(fgrad(np.exp(  1j* im    )))     ).flatten()
     
     def reg2_dcgrd(phim, magim, bfestim):
          #takes in 1d, returns 1d
          phim=phim.reshape([252,308])
          magim=magim.reshape([252,308])
          
          return -2*np.real(1j*np.exp(-1j*phim)*magim *  bfestim*tUFT(  (UFT(bfestim*np.exp(1j*phim)*magim, uspat)-data ), uspat)     ).flatten()
     
     def reg2_dcproj(usph, magim, bfestim, niter=100, alpha_reg=0.05, alpha_dc=0.05):
          #from  Separate Magnitude and Phase Regularization via Compressed Sensing,  Feng Zhao
          
          #usph=usph+np.pi
          
          ims = np.zeros((imsizer,imrizec,niter))
          grds_reg = np.zeros((imsizer,imrizec,niter))
          grds_dc = np.zeros((imsizer,imrizec,niter))
          ims[:,:,0]=usph.copy()
          regval = reg2eval(ims[:,:,0].flatten())
          print(regval)
          for ix in range(niter-1):
               
              grd_reg = reg2grd(ims[:,:,ix].flatten()).reshape([252,308])  # *alpha*np.real(1j*np.exp(-1j*ims[:,:,ix])*    fdivg(fgrad(np.exp(  1j* ims[:,:,ix]    )))     )
              grds_reg[:,:,ix]  = grd_reg
              grd_dc = reg2_dcgrd(ims[:,:,ix].flatten() , magim, bfestim).reshape([252,308])
              grds_dc[:,:,ix]  = grd_dc
              
              ims[:,:,ix+1] = ims[:,:,ix] + alpha_reg*grd_reg  - alpha_dc*grd_dc
              regval = reg2eval(ims[:,:,ix+1].flatten())
              f_dc = dconst(magim*np.exp(1j*ims[:,:,ix+1])*bfestim)
              
              print("KCT-dbg: norm grad reg: " + str(np.linalg.norm(grd_reg)))
              print("KCT-dbg: norm grad dc: " + str(np.linalg.norm(grd_dc)) )
              
              print("KCT-dbg: regval: " + str(regval))
              print("KCT-dbg: fdc: (*1e9) {0:.6f}".format(f_dc/1e9))
          
#          np.save('/home/ktezcan/unnecessary_stuff/phase', ims)
#          np.save('/home/ktezcan/unnecessary_stuff/grds_reg', grds_reg)
#          np.save('/home/ktezcan/unnecessary_stuff/grds_dc', grds_dc)
#          print("SAVED!!!!!!")
          return ims[:,:,-1]#-np.pi    
     
     def reg2_proj_ls(usph, niter=100):
          
          # from  Separate Magnitude and Phase Regularization via Compressed Sensing,  Feng Zhao
          # with line search
         
          usph=usph+np.pi
          
          ims = np.zeros((imsizer,imrizec,niter))
          ims[:,:,0]=usph.copy()
          regval = reg2eval(ims[:,:,0].flatten())
          print(regval)
          for ix in range(niter-1):
               
              currgrd = reg2grd(ims[:,:,ix].flatten())     
              
              res = sop.minimize_scalar(lambda alpha: reg2eval(ims[:,:,ix].flatten() + alpha * currgrd   ), method='Golden'    )
              alphaopt = res.x
              print("KCT-dbg: optimal alpha: " + str(alphaopt) )
               
              ims[:,:,ix+1] = ims[:,:,ix] + alphaopt*currgrd.reshape([252,308]) # *alpha*np.real(1j*np.exp(-1j*ims[:,:,ix])*    fdivg(fgrad(np.exp(  1j* ims[:,:,ix]    )))     )
              regval = reg2eval(ims[:,:,ix+1].flatten())
              print("KCT-dbg: regval: " + str(regval) )
             
          return ims[:,:,-1]-np.pi 


     def N4corrf(im):
          
          phasetmp = np.angle(im)
          ddimcabs = np.abs(im)
     
          inputImage = sitk.GetImageFromArray(ddimcabs, isVector=False)
          corrector = sitk.N4BiasFieldCorrectionImageFilter();
          inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
          output = corrector.Execute(inputImage)
          N4biasfree_output = sitk.GetArrayFromImage(output)
          
          n4biasfield = ddimcabs/(N4biasfree_output+1e-9)
          
          if np.isreal(im).all():
               return n4biasfield, N4biasfree_output 
          else:
               return n4biasfield, N4biasfree_output*np.exp(1j*phasetmp)
     
     
     rmses=np.zeros((1,1,4))
     
     #make the data
     #===============================
     
     uspat=np.abs(us_ksp_r2)>0
     uspat=uspat[:,:,0]
     
     
     data=us_ksp_r2
     
     
          
#     print(uspat)
     
     
          
          
     import pickle
#     lrphase = np.angle( tUFT(data*trpat[:,:,np.newaxis],uspat) )
#     lrphase = pickle.load(open('/home/ktezcan/unnecessary_stuff/lowresphase','rb'))
#     truephase = pickle.load(open('/home/ktezcan/unnecessary_stuff/truephase','rb'))
#     lrphase = pickle.load(open('/home/ktezcan/unnecessary_stuff/usphase','rb')) 
#     lrphase = pickle.load(open('/home/ktezcan/unnecessary_stuff/lrusphase','rb'))
#     lrphase = pickle.load(open('/home/ktezcan/unnecessary_stuff/lrmaskphase','rb'))
     
     
     #make the functions for POCS
     #===================================== 
     numiter=num_iter
     
     multip = 0 #0.1
     
     alphas=stepsize*np.ones(numiter) # np.logspace(-4,-4,numiter)                 
#     alphas=np.ones_like(np.logspace(-4,-4,numiter))*5e-3                                   
     
     def feval(im):
          return full_funceval(im)
     
     def geval(im):
          t1, t2, t3 = full_gradient(im)
          return np.reshape(t1,[-1]), np.reshape(t2,[-1]), np.reshape(t3,[-1])
     
     # initialize data
     recs=np.zeros((imsizer*imrizec,numiter), dtype=complex) 
     
#     recs[:,0] = np.abs(tUFT(data, uspat).flatten().copy()) #kct
     recs[:,0] = tUFT(data, uspat).flatten().copy() 
     
     if N4BFcorr:
          phasetmp = np.reshape(np.angle(recs[:,0]),[imsizer,imrizec])
          n4bf, N4bf_image = N4corrf( np.reshape(recs[:,0],[imsizer,imrizec]) )
          recs[:,0] = N4bf_image.flatten()
          
          
#     recs[:,0] = np.abs(tUFT(data, uspat).flatten().copy() )*np.exp(1j*lrphase).flatten()
     
     
     phaseregvals = []
   
     
     pickle.dump(recs[:,0],open('/home/ktezcan/unnecessary_stuff/init','wb'))

     print('KCT-info: contRec is ' + contRec)
     if contRec != '':
          try:
               print('KCT-INFO: reading from a previous pickle file '+contRec)
               import pickle
               rr=pickle.load(open(contRec,'rb'))
               recs[:,0]=rr[:,-1]
               print('KCT-INFO: initialized to the previous recon from pickle: ' + contRec)
          except:
               print('KCT-INFO: reading from a previous numpy file '+contRec)
               rr=np.load(contRec)
               recs[:,0]=rr[:,-1]
               print('KCT-INFO: initialized to the previous recon from numpy: ' + contRec)

     printed=False
     
     sclval=1
     
     vs=[]
     
     n4biasfields=[]
     
     for it in range(numiter-1):
          
          
          alpha=alphas[it]
     
          if it >onlydciter:
                ftot, f_lik, f_dc = feval(recs[:,it])
                gtot, g_lik, g_dc = geval(recs[:,it])
               
                print("it no: " + str(it) + " f_tot= " + str(ftot) + " f_lik= " + str(f_lik) + " f_dc (1e6)= " + str(f_dc/1e6) + " |g_lik|= " + str(np.linalg.norm(g_lik)) + " |g_dc|= " + str(np.linalg.norm(g_dc)) )
               
                recs[:,it+1]=recs[:,it] - alpha * g_lik
                
                if printed==False:
                     pickle.dump(g_lik,open('/home/ktezcan/unnecessary_stuff/grad_meth3','wb'))
                     printed=True
                     
                tmpa=np.abs(np.reshape(recs[:,it+1],[imsizer,imrizec]))
                tmpp=np.angle(np.reshape(recs[:,it+1],[imsizer,imrizec]))
                
#                tmpatv=tv_proj(tmpa, mu=0.125,lmb=0.1,IT=15).flatten()
                tmpatv = tmpa.copy().flatten()
                
                if reglmb == 0:
                     print("KCT-info: skipping phase proj")
                     tmpptv=tmpp.copy().flatten()
                else:
                     if regtype=='TV':
                          tmpptv=tv_proj(tmpp, mu=0.125,lmb=reglmb,IT=regiter).flatten() #0.1, 15
                     elif regtype=='reg2':
                          tmpptv=reg2_proj(tmpp, alpha=reglmb,niter=regiter).flatten() #0.1, 15
                          regval=reg2eval(tmpp)
                          phaseregvals.append(regval)
                          print("KCT-dbg: pahse reg value is " + str(regval))
                     elif regtype=='reg2_dc':
                          tmpptv=reg2_dcproj(tmpp, tmpa, n4bf, alpha_reg=reglmb, alpha_dc=reglmb, niter=100).flatten()
                          #regval=reg2_dceval(tmpp, tmpa)
                          #phaseregvals.append(regval)
                          #print("KCT-dbg: reg2+DC pahse reg value is " + str(regval))
                     elif regtype=='abs':
                          tmpptv=np.zeros_like(tmpp).flatten()
                     elif regtype=='reg2_ls':
                          tmpptv=reg2_proj_ls(tmpp, niter=regiter).flatten() #0.1, 15
                          regval=reg2eval(tmpp)
                          phaseregvals.append(regval)
                          print("KCT-dbg: pahse reg value is " + str(regval))
                     else:
                          print("hey mistake!!!!!!!!!!")
                          raise ValueError
                
#                tmpatv=tikh_proj(tmpa, niter=100, alpha=0.05).flatten()
#                tmpptv=tikh_proj(tmpp, niter=100, alpha=0.05).flatten()
                
                recs[:,it+1] = tmpatv*np.exp(1j*tmpptv)
                
                if optScale:
                   #try different scale values:
                    v1 = np.linalg.norm(data - UFT(np.reshape(0.99*sclval*recs[:,it+1],[imsizer,imrizec]), uspat  )   )
                    v2 = np.linalg.norm(data - UFT(np.reshape(1*sclval*recs[:,it+1],[imsizer,imrizec]), uspat  )   )
                    v3 = np.linalg.norm(data - UFT(np.reshape(1.01*sclval*recs[:,it+1],[imsizer,imrizec]), uspat  )   )
                   
                    print(v1, v2, v3)
                    if v1<v2 and v1<v3:
                         sclval = sclval*0.99
                         recs[:,it+1] = 0.99*sclval*recs[:,it+1]
                         print("chose 1, val: " +str(sclval))
                    elif v2<v1 and v2<v3:
                         sclval = sclval*1
                         recs[:,it+1] = 1*sclval*recs[:,it+1]
                         print("chose 2, val: " +str(sclval))
                    elif v3<v2 and v3<v1:
                         sclval = sclval*1.01
                         recs[:,it+1] = 1.01*sclval*recs[:,it+1]
                         print("chose 3, val: " +str(sclval))
                         
                vs.append(sclval)
                   
#               print("doing a 2nd reg phase projection")
#               tmp=recs[:,it+1]
#               tmp=np.reshape(tmp,[imsizer,imrizec])
#               tmpp = np.angle(tmp)
##               tmpptv=tv_proj(tmpp,mu=0.125,lmb=1,IT=15)
##               tmpptv=low_pass(tmpp)
##               tmpptv=tikh_proj(tmpp,niter=100,alpha=0.1)
#               tmpptv=reg2_proj(tmpp,niter=20,alpha=0.1)
#               tmp=np.abs(tmp)*np.exp(1j*tmpptv) # *(np.abs(tmp)>0.1)
#               recs[:,it+1]=tmp.flatten().copy()

#               print("doing a ABS phase projection")
#               recs[:,it+1]=np.abs(recs[:,it+1])
                   
          else:   
               print('KCT-info: skipping prior proj for the first onlydciters iter.s, doing only phase proj (then maybe DC proj as well) !!!')
               recs[:,it+1]=recs[:,it].copy()
               
               tmpa=np.abs(np.reshape(recs[:,it+1],[imsizer,imrizec]))
               tmpp=np.angle(np.reshape(recs[:,it+1],[imsizer,imrizec]))
                
#               tmpatv=tv_proj(tmpa, mu=0.125,lmb=0.1,IT=15).flatten()
               tmpatv = tmpa.copy().flatten()
                
               if reglmb == 0:
                     print("KCT-info: skipping phase proj")
                     tmpptv=tmpp.copy().flatten()
                     
               else:
                    if regtype=='TV':
                          tmpptv=tv_proj(tmpp, mu=0.125,lmb=reglmb,IT=regiter).flatten() #0.1, 15
                    elif regtype=='reg2':
                          tmpptv=reg2_proj(tmpp, alpha=reglmb,niter=regiter).flatten() #0.1, 15
                          regval=reg2eval(tmpp)
                          phaseregvals.append(regval)
                          print("KCT-dbg: pahse reg value is " + str(regval))
                    elif regtype=='abs':
                          tmpptv=np.zeros_like(tmpp).flatten()
                    elif regtype=='reg2_ls':
                          tmpptv=reg2_proj_ls(tmpp, niter=regiter).flatten() #0.1, 15
                          regval=reg2eval(tmpp)
                          phaseregvals.append(regval)
                          print("KCT-dbg: pahse reg value is " + str(regval))
                    else:
                          print("hey mistake!!!!!!!!!!")
                          raise ValueError
                
#                tmpatv=tikh_proj(tmpa, niter=100, alpha=0.05).flatten()
#                tmpptv=tikh_proj(tmpp, niter=100, alpha=0.05).flatten()
               
               recs[:,it+1] = tmpatv*np.exp(1j*tmpptv)
               

          #do the DC projection every N iterations    
          if  it < onlydciter+1 or it % dcprojiter == 0: # 
               
               
               if not N4BFcorr:  
                    tmp1 = UFT(np.reshape(recs[:,it+1],[imsizer,imrizec]), (1-uspat)  )
                    tmp2 = UFT(np.reshape(recs[:,it+1],[imsizer,imrizec]), (uspat)  )
                    tmp3= data*uspat[:,:,np.newaxis]
                    
                    tmp=tmp1 + multip*tmp2 + (1-multip)*tmp3
                    recs[:,it+1] = tFT(tmp).flatten()
                    
                    #dbg:
                    tmp2 = UFT(tFT(tmp), uspat)
#                    np.savez('/home/ktezcan/unnecessary_stuff/tmps', tmp=tmp, tmp2=tmp2, uspat=uspat)
                    
                    ftot, f_lik, f_dc = feval(recs[:,it+1])
                    print('f_dc (1e6): ' + str(f_dc/1e6) + '  perc: ' + str(100*f_dc/np.linalg.norm(data)**2))
                    
               elif N4BFcorr:
                    
                    n4bf_prev = n4bf.copy()
                    imgtmp = np.reshape(recs[:,it+1],[imsizer,imrizec]) # biasfree
                    imgtmp_bf = imgtmp*n4bf_prev # img with bf
                    
                    n4bf, N4bf_image = N4corrf( imgtmp_bf ) # correct the bf, this correction is supposed to be better now.
                    
                    imgtmp_new = imgtmp*n4bf
                    
                    
                    n4biasfields.append(n4bf)
                    
                    
                    
                    tmp1 = UFT(imgtmp_new, (1-uspat)  )
                    tmp2 = UFT(np.reshape(recs[:,it+1],[imsizer,imrizec]), (uspat)  )
                    tmp3= data*uspat[:,:,np.newaxis]
                    
                    tmp=tmp1 + multip*tmp2 + (1-multip)*tmp3 # multip=0 by default
                    recs[:,it+1] = (tFT(tmp)/n4bf).flatten()
                    
                    ftot, f_lik, f_dc = feval(recs[:,it+1])
                    print('f_dc (1e6): ' + str(f_dc/1e6) + '  perc: ' + str(100*f_dc/np.linalg.norm(data)**2))
                    
#               elif N4BFcorr:
#                    
#                    n4bf_prev = n4bf.copy()
##                    imgtmp = np.reshape(recs[:,it+1],[imsizer,imrizec]) # biasfree
##                    imgtmp_bf = imgtmp*n4bf_prev # img with bf
##                    
##                    n4bf, N4bf_image = N4corrf( imgtmp_bf ) # correct the bf, this correction is supposed to be better now.
##                    
##                    imgtmp_new = imgtmp*n4bf
#                    
#                    
#                    n4biasfields.append(n4bf)
#                    
#                    
#                    
#                    tmp1 = UFT(np.reshape(recs[:,it+1],[imsizer,imrizec])*n4bf_prev, (1-uspat)  )
#                    tmp2 = UFT(np.reshape(recs[:,it+1],[imsizer,imrizec]), (uspat)  )
#                    tmp3= data*uspat[:,:,np.newaxis]
#                    
#                    tmp=tmp1 + multip*tmp2 + (1-multip)*tmp3 # multip=0 by default
#                    
#                    tmp = tFT(tmp)
#                    n4bf_new, N4bf_image = N4corrf( tmp ) # correct the bf, this correction is supposed to be better now.
#                    
#                    
#                    recs[:,it+1] = (tmp/n4bf_new).flatten()
#                    
#                    ftot, f_lik, f_dc = feval(recs[:,it+1])
#                    print('f_dc (1e6): ' + str(f_dc/1e6) + '  perc: ' + str(100*f_dc/np.linalg.norm(data)**2))
                    
               else:
                    pass  # n4bf, N4bf_image = N4corrf( np.reshape(recs[:,it+1],[imsizer,imrizec]) )
     
#               
#          pickle.dump(recs,open('/home/ktezcan/unnecessary_stuff/recs','wb'))  
#            
          #endif     
     
          
          
     return recs, vs, phaseregvals, n4biasfields



     


