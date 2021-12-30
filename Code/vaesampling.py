from __future__ import division
from __future__ import print_function
import numpy as np

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import time

scriptdir = os.path.dirname(sys.argv[0])+'/'
#print("KCT-info: running script from directory: " + scriptdir)
os.chdir(scriptdir)



def vaesampling(usksp, sensmaps, maprecon, mapphase, directoryname, step = 1e-3, im_kspsize=[[252,308], [252,308,1]], model_prec_val=50, numinversioniters=35, empiricalPrior=True, lowres=False, BFcorr=False, biasfield = [0], numsamp = 100000, saveevery = 200, nsksp_manual=-1, noisemodifier=1.0, rollval = 0):
     
     
     empiricalPrior=empiricalPrior # True
     
     lowres = lowres
     
          
     # set only the VAE precision value here:
     # ----------------------
     model_precision_value = model_prec_val # 50
     #I'll define this to be 1, and rescale everything accordingly.
     sx=1
     #1)so the measured noise precision value should be divided by 50.
     #2) in the MALA acceptance probaility, I need to adjust the values as well.
     # this is done by setting the scaling value=modelprecision value:
     scl = model_precision_value
     
     
     # get the necessary image/kspace related stuff
     mapim = np.abs(maprecon)
     mapphase = mapphase#  np.exp(1j*np.angle(maprecon)) # 
     
     #uspat=pickle.load(open('/usr/bmicnas01/data-biwi-01/ktezcan/old_recon_results/recon_results/uspat_latd'+str(60)+'_MRIunproc_us'+str(4)+'_im'+str(9),'rb'))
            
     sensmaps = sensmaps # np.ones([252,308,1])
     
     # usksp = usksp
     uspat = (np.abs(usksp)>0)
     
#     print(biasfield)
     if biasfield.all()==-1:
          biasfield = np.ones([usksp.shape[0], usksp.shape[1]])
          
     if (np.imag(biasfield)!=0).any():
          print("KCT-error: biasfield cannot be complex!")
          assert ValueError
          
     print("KCT-info: biasfield shape: " + str(biasfield.shape))
     
     if nsksp_manual==-1:
          #calculate the ns value here. For this you need to measure the k-space noise
          #precision and divide this by the value of model precision.  
          nsksp = np.zeros([1, 1, usksp.shape[2]])
          for ix in range(usksp.shape[2]):
               nsprec = 1/usksp[0:20,int(np.floor(usksp.shape[1]/2))-5:int(np.floor(usksp.shape[1]/2))+5,ix].var() * noisemodifier
               nsksp[0,0,ix] = nsprec/model_precision_value
          print("KCT-info: noise value is " + str(nsksp) )
     else:
          nsksp = np.array([nsksp_manual])
          print("KCT-WARN: manually set the noise value!!!!")
          print("KCT-info: noise value is " + str(nsksp) )
     
     
#     # input the noise value here by hand:
#     nsprec = 1/0.13
#     ns = nsprec/model_precision_value
#     #################################
     
     
     
     # Load the model
     errtmpfvs = []
     chctr=0
     
     
     
     from definevae_2d_v1_mri_nocov_fullim_conz_homodyn_varsize_f2 import definevae_2d_v1_mri_nocov_fullim_conz_homodyn_varsize_f2
     
     
     def log_q_xp_x(xp, x, p_x):
          tmp = -1/(4*step) * np.square(np.linalg.norm(xp - x - step*p_x))
          return tmp
     
     #-----------------------------------------------
     # HCP:
     # SX1:    ns1=1e-1, ns21 = 1e-3, ns81 = 1e-4, ns540=2e-6, ns2000=2e-7, ns20000=4e-9, ns54000=1.92e-10
     #-----------------------------------------------
     # real image:
     # SX1::   ns540:2e-6
     #-----------------------------------------------
          
     sclfct_cur = np.array( [1+0*1j])
     sclfct_curs = []
          
     #for ixa, alphaval in enumerate(np.logspace(-6,-5,15)): 
     alphaval = 4e-6
     
     
     #alphaval=1e-12
       
#     del definevae_2d_v1_mri_nocov_fullim_conz_homodyn_f   
#     
#     from definevae_2d_v1_mri_nocov_fullim_conz_homodyn_f import definevae_2d_v1_mri_nocov_fullim_conz_homodyn_f
     
     
     print("KCT-info: rollval value in vaesampling: " + str(rollval))
     
     
     starttime1 = time.time() 

#               else:
     dec_mu, _, x_inp, grd0, funop, grd_p_x_z0, grd_p_z0, grd_q_z_x0, \
     grd_op_p_x_z_cs0, grd_cs0, enc_mu, enc_std, z_samples, \
     z_samplesr, dec_mur, sess, _, uspattff, sensmapsplf, itsf, \
     errtmpf, _, mupSmup, grd_mupSmup, yy, full1f, full2, errtmp0f, \
     grd1f, dec_muf, gammaf, full3f, p_y_xf, phaseimf, sclfactorf, alphaiterpl, \
     dbgimpl, ufttff, bb, biasfieldtf, optimalscalef, optimalscale_tmp1f, \
     optimalscale_tmp2f, ttt11f, ttt12f, mupost_zy, mupost_tmp1f, mupost_tmp2f  \
     = definevae_2d_v1_mri_nocov_fullim_conz_homodyn_varsize_f2(batch_size=5, im_kspsize=im_kspsize , lowres=lowres, BFcorr = BFcorr, nskspval=nsksp, sxval=sx, numinversioniters=numinversioniters, rollval = rollval)

     
     
     
     
     print("KCT-info: noise values are:" + str(nsksp))
     
#     # first do a parameter search for the proper alpha value for the matrix inversion:
#     print("KCT-info: Doing a parameter search for the proper alpha value for the matrix inversion")
#     enc_muv = enc_mu.eval(feed_dict={x_inp: np.tile(mapim[np.newaxis,:,:],[5,1,1])})
#     zsc = enc_muv[0,:,:,:].copy()
#     alphaval = 1
#     while(True): 
#          errtmpfv, errtmp0fv = sess.run([errtmpf, errtmp0f], \
#                                         feed_dict={x_inp: np.zeros([5,mapim.shape[0],mapim.shape[1]]), z_samples: np.tile((zsc)[np.newaxis,:,:],[5,1,1,1]), uspattff: uspat[:,:,0], sensmapsplf:sensmaps, yy:usksp, phaseimf:mapphase, sclfactorf:sclfct_cur, alphaiterpl:alphaval })
#          print("KCT-info: alphaval: {0:.8f}, error val: {1:.8f}".format(alphaval, errtmpfv))
#          if np.isnan(errtmpfv) or errtmpfv>100:
#               alphaval=alphaval/2
#          elif errtmpfv<100 and errtmpfv>1:
#                alphaval=alphaval/1.5            
#          elif errtmpfv<1e-3:
#               break
#          elif errtmpfv<1:
#               alphaval=alphaval*1.5
#     print("KCT-info: Found!! alphaval: {0:.8f}, error val: {1:.8f}".format(alphaval, errtmpfv))
     
     
#     # first do a parameter search for the proper alpha value for the matrix inversion:
#     print("KCT-info: Doing a parameter search for the proper alpha value for the matrix inversion")
#     enc_muv = enc_mu.eval(feed_dict={x_inp: np.tile(mapim[np.newaxis,:,:],[5,1,1])})
#     zsc = enc_muv[0,:,:,:].copy()
#     alphaval = 1
#     errtmpfv, _ = sess.run([errtmpf, errtmp0f], feed_dict={x_inp: np.zeros([5,mapim.shape[0],mapim.shape[1]]), z_samples: np.tile((zsc)[np.newaxis,:,:],[5,1,1,1]), uspattff: uspat[:,:,0], sensmapsplf:sensmaps, yy:usksp, phaseimf:mapphase, sclfactorf:sclfct_cur, alphaiterpl:alphaval })
#          
#     while(np.isnan(errtmpfv)): 
#          alphaval=alphaval/2
#          errtmpfv, errtmp0fv = sess.run([errtmpf, errtmp0f], feed_dict={x_inp: np.zeros([5,mapim.shape[0],mapim.shape[1]]), z_samples: np.tile((zsc)[np.newaxis,:,:],[5,1,1,1]), uspattff: uspat[:,:,0], sensmapsplf:sensmaps, yy:usksp, phaseimf:mapphase, sclfactorf:sclfct_cur, alphaiterpl:alphaval })
#               
#     print("KCT-info: Non-Nan found!! alphaval: {0:.8f}, error val: {1:.8f}".format(alphaval, errtmpfv))
#     print("KCT-info: Now look for coarse good value...")
#     
#     while(True):
#          alphavalnew=alphaval/2
#          errtmpfv, _ = sess.run([errtmpf, errtmp0f], feed_dict={x_inp: np.zeros([5,mapim.shape[0],mapim.shape[1]]), z_samples: np.tile((zsc)[np.newaxis,:,:],[5,1,1,1]), uspattff: uspat[:,:,0], sensmapsplf:sensmaps, yy:usksp, phaseimf:mapphase, sclfactorf:sclfct_cur, alphaiterpl:alphaval })
#          errtmpfvnew, _ = sess.run([errtmpf, errtmp0f], feed_dict={x_inp: np.zeros([5,mapim.shape[0],mapim.shape[1]]), z_samples: np.tile((zsc)[np.newaxis,:,:],[5,1,1,1]), uspattff: uspat[:,:,0], sensmapsplf:sensmaps, yy:usksp, phaseimf:mapphase, sclfactorf:sclfct_cur, alphaiterpl:alphavalnew })
#          if errtmpfvnew < errtmpfv:
#               alphaval = alphavalnew
#          elif errtmpfvnew > errtmpfv:
#               break
#     print("KCT-info:Coarse value found!! alphaval: {0:.8f}, error val: {1:.8f}".format(alphaval, errtmpfv))
#      
#     print("KCT-info: Now look for fine value...")
#     factor = 1.1
#     while(True):
#           alphavalnewlow=alphaval/factor
#           alphavalnewhigh=alphaval*factor
#           
#           errtmpfv, _ = sess.run([errtmpf, errtmp0f], feed_dict={x_inp: np.zeros([5,mapim.shape[0],mapim.shape[1]]), z_samples: np.tile((zsc)[np.newaxis,:,:],[5,1,1,1]), uspattff: uspat[:,:,0], sensmapsplf:sensmaps, yy:usksp, phaseimf:mapphase, sclfactorf:sclfct_cur, alphaiterpl:alphaval })
#           errtmpfvnewlow, _ = sess.run([errtmpf, errtmp0f], feed_dict={x_inp: np.zeros([5,mapim.shape[0],mapim.shape[1]]), z_samples: np.tile((zsc)[np.newaxis,:,:],[5,1,1,1]), uspattff: uspat[:,:,0], sensmapsplf:sensmaps, yy:usksp, phaseimf:mapphase, sclfactorf:sclfct_cur, alphaiterpl:alphavalnewlow })
#           errtmpfvnewhigh, _ = sess.run([errtmpf, errtmp0f], feed_dict={x_inp: np.zeros([5,mapim.shape[0],mapim.shape[1]]), z_samples: np.tile((zsc)[np.newaxis,:,:],[5,1,1,1]), uspattff: uspat[:,:,0], sensmapsplf:sensmaps, yy:usksp, phaseimf:mapphase, sclfactorf:sclfct_cur, alphaiterpl:alphavalnewhigh })
#
#           print("current, low high: " + str(errtmpfv) + ", " + str(errtmpfvnewlow) + ", " + str(errtmpfvnewhigh) )
#           print("max diff: " + str(max(np.abs(errtmpfvnewlow - errtmpfv), np.abs(errtmpfvnewhigh - errtmpfv) )))
#           
#           if errtmpfvnewlow < errtmpfv:
#               alphaval = alphavalnewlow
#               print("took: lower")
#               
#           elif errtmpfvnewhigh < errtmpfv:
#               alphaval = alphavalnewhigh
#               print("took: higher")
#               
#           elif errtmpfvnewlow > errtmpfv and errtmpfvnewhigh > errtmpfv:
#               fctdif = factor-1
#               factor = 1+(fctdif/2)
#               print("factor: " + str(factor))
#          
##           if np.abs(errtmpfvnewlow - errtmpfv) < 1e-7 or np.abs(errtmpfvnewhigh - errtmpfv) < 1e-7:
#           if factor < 1.00005:
#               break
          
     
     print("KCT-info: starting testing inversion")
     
     enc_muv = enc_mu.eval(feed_dict={x_inp: np.tile(mapim[np.newaxis,:,:],[5,1,1])})
     zsc = enc_muv[0,:,:,:].copy()
     itsfv, errtmpfv, errtmp0fv = sess.run([itsf, errtmpf, errtmp0f], feed_dict={x_inp: np.zeros([5,mapim.shape[0],mapim.shape[1]]), z_samples: np.tile((zsc)[np.newaxis,:,:],[5,1,1,1]), uspattff: uspat[:,:,0], sensmapsplf:sensmaps, yy:usksp, phaseimf:mapphase, sclfactorf:sclfct_cur, alphaiterpl:alphaval, biasfieldtf: biasfield })
     
     print("KCT-info: Inversion: final: {0:.5f}, initial: {1:.5f}".format(errtmpfv, errtmp0fv))
          
     
     print("KCT-DBG: doing checking/debug stuff here:")
#     dbgimpl, ufttff, bb
      
     dec_muv, ufttffv, bbv = sess.run([dec_mu, ufttff, bb], feed_dict={dbgimpl:mapim,  x_inp: np.tile(mapim[np.newaxis,:,:],[5,1,1]), z_samples: np.tile((zsc)[np.newaxis,:,:],[5,1,1,1]), uspattff: uspat[:,:,0], sensmapsplf:sensmaps, yy:usksp, phaseimf:mapphase, sclfactorf:sclfct_cur, alphaiterpl:alphaval, biasfieldtf: biasfield })
#     np.savez('/home/ktezcan/unnecessary_stuff/dbgstuff', mapim=mapim,  ufttffv=ufttffv,  bbv=bbv, dec_muv=dec_muv, itsfv=itsfv, usksp=usksp)
     
     not_overlapping = []
     for ix in range(im_kspsize[1][2]):
          max_tf = np.argmax(np.abs(ufttffv[:,:,ix]).flatten())
          max_np = np.argmax(np.abs(usksp[:,:,ix]).flatten())
          print("max tf, max_np:")
          print(np.unravel_index(max_tf, usksp.shape[0:2]), np.unravel_index(max_np, usksp.shape[0:2]))
          if max_tf !=max_np:
               not_overlapping.append(ix)
               
     if not_overlapping: # list not empty
          print("KCT-WARNING: The k-spaces from the calling function and the Tensorflow are not ovrelapping at indices:")
          print(not_overlapping)
#          assert(1==0)
     elif not not_overlapping:
          print("The k-spaces are overlapping, all good.")
               
     
     print("KCT-DBG: debug stuff finished:")
#     alphaval=0.017
#     print("KCT-WARN: alphaval set manually to: {0:.8f}".format(alphaval))
     
     
     
     
#     # first do a parameter search for the proper alpha value for the matrix inversion:
#     print("KCT-info: Doing a parameter search for the proper alpha value for the matrix inversion")
#     enc_muv = enc_mu.eval(feed_dict={x_inp: np.tile(mapim[np.newaxis,:,:],[5,1,1])})
#     zsc = enc_muv[0,:,:,:].copy()
#     alphaval = 1
#     multip = 0.5
#     ctr=0
#     while(True): 
#          print("======== ctr: "+ str(ctr))
#          alphaval_prop = alphaval*multip
#          errtmpfv_alphaval, _ = sess.run([errtmpf, errtmp0f], \
#                                         feed_dict={x_inp: np.zeros([5,mapim.shape[0],mapim.shape[1]]), z_samples: np.tile((zsc)[np.newaxis,:,:],[5,1,1,1]), uspattff: uspat[:,:,0], sensmapsplf:sensmaps, yy:usksp, phaseimf:mapphase, sclfactorf:sclfct_cur, alphaiterpl:alphaval })
#          errtmpfv_alphavalprop, _ = sess.run([errtmpf, errtmp0f], \
#                                         feed_dict={x_inp: np.zeros([5,mapim.shape[0],mapim.shape[1]]), z_samples: np.tile((zsc)[np.newaxis,:,:],[5,1,1,1]), uspattff: uspat[:,:,0], sensmapsplf:sensmaps, yy:usksp, phaseimf:mapphase, sclfactorf:sclfct_cur, alphaiterpl:alphaval_prop })
#         
#          print("KCT-info: alphaval: {0:.8f}, error val: {1:.8f}".format(alphaval, errtmpfv_alphaval))
#          print("KCT-info: alphavalprop: {0:.8f}, error val prop: {1:.8f}".format(alphaval_prop, errtmpfv_alphavalprop))
#          if np.isnan(errtmpfv_alphaval) or np.isnan(errtmpfv_alphavalprop):
#               alphaval=alphaval/2
#          elif errtmpfv_alphavalprop < errtmpfv_alphaval:
#                alphaval=alphaval_prop
#                cnt=True
#                while(cnt ):
#                     alphaval_prop2=alphaval*multip
#                     errtmpfv_alphaval, _ = sess.run([errtmpf, errtmp0f], \
#                                         feed_dict={x_inp: np.zeros([5,mapim.shape[0],mapim.shape[1]]), z_samples: np.tile((zsc)[np.newaxis,:,:],[5,1,1,1]), uspattff: uspat[:,:,0], sensmapsplf:sensmaps, yy:usksp, phaseimf:mapphase, sclfactorf:sclfct_cur, alphaiterpl:alphaval })
#                     errtmpfv_alphavalprop2, _ = sess.run([errtmpf, errtmp0f], \
#                                         feed_dict={x_inp: np.zeros([5,mapim.shape[0],mapim.shape[1]]), z_samples: np.tile((zsc)[np.newaxis,:,:],[5,1,1,1]), uspattff: uspat[:,:,0], sensmapsplf:sensmaps, yy:usksp, phaseimf:mapphase, sclfactorf:sclfct_cur, alphaiterpl:alphaval_prop2 })
#                     print(" inner loop: alphaval: {0:.8f}, error val: {1:.8f}".format(alphaval, errtmpfv_alphaval))
#                     print(" inner loop: alphavalprop2: {0:.8f}, error val prop: {1:.8f}".format(alphaval_prop2, errtmpfv_alphavalprop2))
#                     
#                     if errtmpfv_alphavalprop2 < errtmpfv_alphaval:
#                          alphaval = alphaval_prop2
#                     elif errtmpfv_alphavalprop2 > errtmpfv_alphaval:
#                          print("broke inner")
#                          cnt=False
#          elif errtmpfv_alphaval > errtmpfv_alphavalprop:
#               multip = multip*1.5
#               
#          if multip > 1:# or np.abs(errtmpfv_alphaval - errtmpfv_alphavalprop) < 1e-3:
#               print(multip)
#               print(np.abs(errtmpfv_alphaval - errtmpfv_alphavalprop))               
#               print("broke outer")
#               break
#               
#          ctr=ctr+1
#                          
#              
#     print("KCT-info: Found!! alphaval: {0:.8f}, error val: {1:.8f}".format(alphaval, errtmpfv_alphaval))

     if empiricalPrior:
          
          #load the necessary stuff here:
          if lowres==False and BFcorr==True:
               stuff = np.load(os.getcwd() + '/../../trained_models/covariances_emp_prior/covariancestuff_legacymodel_bfc.npz')
               lssize=[18,22]
               print("KCT-info: loaded the empirical prior: " + os.getcwd() + '/../../trained_models/covariances_emp_prior/covariancestuff_legacymodel_bfc.npz')
          elif lowres==True and BFcorr==True:
               stuff = np.load(os.getcwd() + '/../../trained_models/covariances_emp_prior/covariancestuff_legacymodel_bfc_lowres.npz')
               lssize=[13,15]
               print("KCT-info: loaded the empirical prior: " + os.getcwd() + '/../../trained_models/covariances_emp_prior/covariancestuff_legacymodel_bfc_lowres.npz')
#               step=2e-5
               
               
          covs=stuff['covs']
          covs_invs=stuff['covs_invs']
          means=stuff['means']
          nginx=stuff['nginx']
          covng=stuff['covng']
          covng_inv=stuff['covng_inv']
          meanng=stuff['meanng']
          
          
          
          def Ci_vec_prod(vec):
               # z: [18x22x60]# mean_comb = meanng
               # do all in the [60,18,22] order, then convert back
               
               vec = np.transpose(vec, [2,0,1]) # [60,18,22]
               
               #first do the fully correlated ones
               ng_zs = vec[nginx,:,:].flatten()
               ng_zs = np.matmul( covng_inv, ng_zs - meanng)
               ng_zs = np.reshape(ng_zs, [nginx.shape[0], lssize[0], lssize[1]])
               
               
               #now do the channel-uncorrelated ones
               tmp=np.zeros([60,lssize[0]*lssize[1]])
               for ix in range(60):
                    tmp[ix,:] = np.matmul( covs_invs[ix,:,:], (vec[ix,:,:].flatten() - means[ix,:])  )
               tmp=np.reshape(tmp,[60,lssize[0], lssize[1]])
               
               tmp[nginx,:,:] = ng_zs.copy()
               
               prod = np.transpose(tmp, [1,2,0]) # back to [18,22,11]
               
               #finally just get the z-mu:
               diff = np.transpose(  vec - means.reshape(60,lssize[0], lssize[1]) , [1,2,0]  )
               
               return prod, diff  
          
          def grad_pz(z):
               prod, _ = Ci_vec_prod(z)
               return (-1)*prod
          
          def norm_pz(z):
               prod, diff = Ci_vec_prod(z)
               return (-0.5)*np.sum(diff*prod)
               
               
               
          # get the cpvariance structures of p(z) here - DONE
          #############################################
          ##############################################
     
     print("KCT-info: step used is " + str(step))
     
     acceptanceratios = []
     
     pzddv = 1
     
     #-----------------------------------------------    
     # HCP:     
     # using 4e-4 for ns=21, using 5e-4 for ns=420, using 4e-4 for ns=540, using 4e-23 for ns=54000 
     #-----------------------------------------------
     # measured:
     # around 2e-3...
     #-----------------------------------------------
     
#     print(time.time()- starttime1)    
     
     
          
     enc_muv = enc_mu.eval(feed_dict={x_inp: np.tile(mapim[np.newaxis,:,:],[5,1,1])})
#          zs = np.zeros([numsamp+2,18,22,60])
     zsc = enc_muv[0,:,:,:].copy()
     
     
#     print(time.time()- starttime1)
     
     acceptance = []
     p1s = []
     q1s = []
     ims = []
     zscs = []
     mupSmups = []
     pzs = []
     full1s=[]
     full2s=[]
     full3s=[]
     p_y_xs = []
   
     grd_pz_norms = []
     grd_pyz_pz_norms = []
     gammavs = []
     
     mupost_zys = []
     mupost_tmp1s = []
     mupost_tmp2s = []
     
     acceptctr = 0
          
     
#     np.savez(directoryname+'/arrays_model_precision_value'+str(model_precision_value)+'_withscl_r35_scl'+str(scl)+'_sx'+str(sx)+'_empPrior'+str(empiricalPrior)+'_step'+str(step)+'_samp_init', \
#             zsc=zsc) # , gammavs=gammavs
     
     
     for ix in range(1, numsamp+1):
          
          
         #first do a scale estimation and use this for the sampling 
         optimalscalev, optimalscale_tmp1v, optimalscale_tmp2v, ttt11v, ttt12v = sess.run([ optimalscalef, optimalscale_tmp1f, optimalscale_tmp2f, ttt11f, ttt12f], feed_dict={x_inp: np.zeros([5,mapim.shape[0],mapim.shape[1]]), z_samples: np.tile((zsc)[np.newaxis,:,:],[5,1,1,1]), uspattff: uspat[:,:,0], sensmapsplf:sensmaps, yy:usksp, phaseimf:mapphase, sclfactorf:[1+0*1j], biasfieldtf: biasfield})
         print("KCT-info: optimal scale value is: :")   
         print(optimalscalev)
#         print("tmp scale value 1 is: :")   
#         print(optimalscale_tmp1v)
#         print("tmp scale value 2 is: :")   
#         print(optimalscale_tmp2v)
         
         print("KCT-WARN: switched ON scale factor correction")    
         sclfct_cur = [np.real(optimalscalev)] # np.array( [1+0*1j]) #  
          
          
         grd_mupSmupv, mupSmupv, errtmpfv, errtmp0fv,  dec_mufv, full2v, gammav, \
         full1v, full3v, p_y_xv_curr,  mupost_zy_currv, mupost_tmp1_currv, mupost_tmp2_currv = \
         sess.run([grd_mupSmup[0], mupSmup, errtmpf, errtmp0f, dec_muf, full2, gammaf, full1f, full3f, p_y_xf,  mupost_zy, mupost_tmp1f, mupost_tmp2f], \
                  feed_dict={x_inp: np.zeros([5,mapim.shape[0],mapim.shape[1]]), z_samples: np.tile((zsc)[np.newaxis,:,:],[5,1,1,1]), uspattff: uspat[:,:,0], sensmapsplf:sensmaps, yy:usksp, phaseimf:mapphase, sclfactorf:sclfct_cur, alphaiterpl:alphaval, biasfieldtf: biasfield })
         
         mupSmupv_curr = mupSmupv.copy()
         grd_mupSmupv_curr = grd_mupSmupv.copy()
         
         if empiricalPrior:
              grd_pz_curr =  (1/scl)*grad_pz(zsc)
              pz_curr =  (1/scl)*norm_pz(zsc)
         else:
              grd_pz_curr = - (1/scl)*zsc*pzddv
              pz_curr = - (1/scl)*(1/2)*np.square(np.linalg.norm(zsc*pzddv)) # np.square
         
         
         
         
#              print(">>>>> alphavalue is : " + str(alphaval))
#              print(">>>>> noise value is : " + str(nsksp))
         print("")
         print("ix: {0}, mupSmup: {1:.2f}, errtmpf: {2:.4f}, errtmpf0: {3:.2f}, |grd_mupSmupv|: {4:.2f}, |dec_mu|: {5:.2f}".format(ix, (mupSmupv), (errtmpfv), (errtmp0fv), np.abs(np.linalg.norm(grd_mupSmupv[0,:,:,:])), np.abs(np.linalg.norm(dec_mufv)) )   )
         print("p(z): {0:.5f}, grd p(z): {1:.5f}, grd p(y|z)p(z): {2:.2f}, p(y|z)p(z): {3:.5f}".format(pz_curr, np.linalg.norm(grd_pz_curr), np.linalg.norm(grd_mupSmupv[0,:,:,:] + grd_pz_curr), ( (mupSmupv) + pz_curr      ) ))
         print("full1: " + str((full1v)) + " | full2: " + str((full2v)) + " | full3: " + str((full3v)) + " log p(y|x): " + str((p_y_xv_curr))   )
         print("sx and scale values are " + str(sx) +", "+str(scl))
#         print("full2 value: " + str(full2v))
         
              
         zs_prop = zsc +  step * (grd_mupSmupv[0,:,:,:] + grd_pz_curr  ) + np.sqrt(2*step/scl)*np.sqrt(pzddv)*np.random.randn(zsc.shape[0],zsc.shape[1],zsc.shape[2]) # - grd_muxSmuxf[0,:,:,:]
         
         grd_mupSmupv_prop, mupSmupv_prop, errtmpfv, errtmp0fv,  \
         dec_mufv, p_y_xv_prop,  mupost_zy_propv, mupost_tmp1_propv, mupost_tmp2_propv = \
         sess.run([grd_mupSmup[0], mupSmup, errtmpf, errtmp0f, dec_muf, p_y_xf,  mupost_zy, mupost_tmp1f, mupost_tmp2f], \
                  feed_dict={x_inp: np.zeros([5,mapim.shape[0],mapim.shape[1]]), z_samples: np.tile(zs_prop[np.newaxis,:,:],[5,1,1,1]), uspattff: uspat[:,:,0], sensmapsplf:sensmaps, yy:usksp, phaseimf:mapphase, sclfactorf:sclfct_cur, alphaiterpl:alphaval, biasfieldtf: biasfield})
         

         if empiricalPrior:
              grd_pz_prop =  (1/scl)*grad_pz(zs_prop)
              pz_prop =  (1/scl)*norm_pz(zs_prop)
         else:
              grd_pz_prop = - (1/scl)* zs_prop*pzddv
              pz_prop = - (1/scl)*(1/2)*np.square(np.linalg.norm(zs_prop*pzddv)) # np.square
         
         p1 = np.real(mupSmupv_prop + pz_prop) - np.real(mupSmupv_curr + pz_curr) # in log space
         
         q1 = (1/scl)*( log_q_xp_x(zsc, zs_prop, grd_mupSmupv_prop+grd_pz_prop) - log_q_xp_x(zs_prop, zsc, grd_mupSmupv_curr+grd_pz_curr)  )
#              
         def log_q_xp_x(xp, x, p_x):
               tmp = -1/(4*step) * np.square(np.linalg.norm(xp - x - step*p_x))
               return tmp

         
#              def log_q_xp_x(xp, x, p_x):
#                    tmp = -1/(4*step) * np.square(np.linalg.norm(xp - x - step*p_x))
#                    return tmp+
         
         alpha = p1+q1
         
#         print("p1")
#         print(p1)
#         print("q1")
#         print(q1)
         
#         print("p1 + q1")
#         print(alpha)
         
         p1s.append(p1)
         q1s.append(q1)
         
         if alpha>0:
              alpha=0
              
#         alpha = np.exp(alpha)


         sclfct_curs.append(sclfct_cur)
              
         #get a random val u
         u = (1/scl)*np.log(np.random.rand()) # do this also in log space
         
         if u < alpha:
              zsc = zs_prop
              acceptctr = acceptctr + 1
              acceptance.append(1)
              
              mupSmups.append(mupSmupv_prop)
              pzs.append(pz_prop)
              full1s.append(full1v)
              full2s.append(full2v)
              full3s.append(full3v)
              p_y_xs.append(p_y_xv_prop)
              
              grd_pz_norms = np.linalg.norm(grd_pz_prop)     
              grd_pyz_pz_norms = np.linalg.norm(grd_mupSmupv[0,:,:,:] + grd_pz_prop)
              
              gammavs.append(gammav)
              
              mupost_zys.append(mupost_zy_propv)
              mupost_tmp1s.append(mupost_tmp1_propv)
              mupost_tmp2s.append(mupost_tmp2_propv)
              
         else:
              zsc = zsc.copy()
              acceptance.append(0)
              
              mupSmups.append(mupSmupv_curr)
              pzs.append(pz_curr)
              full1s.append(full1v)
              full2s.append(full2v)
              full3s.append(full3v)
              p_y_xs.append(p_y_xv_curr)
              
              grd_pz_norms =  np.linalg.norm(grd_pz_curr)
              grd_pyz_pz_norms = np.linalg.norm(grd_mupSmupv[0,:,:,:] + grd_pz_curr)
              
              gammavs.append(gammav)
              
              mupost_zys.append(mupost_zy_currv)
              mupost_tmp1s.append(mupost_tmp1_currv)
              mupost_tmp2s.append(mupost_tmp2_currv)
              
         zscs.append(zsc)
                
         
              
         
         print("u: "+str(u)+", alpha: "+str(alpha)+", accept ratio: " + str(acceptctr/(ix+1)) + " scale value: " + str(np.real(sclfct_cur)))
         
         if np.abs(errtmpfv)>0.1:
              print("KCT-warn: WARNING!!! High inversion error") 
         
         if ix % 200==0:
              print(acceptance)
              
         dec_muv = dec_mu.eval(feed_dict={x_inp: np.zeros([5,mapim.shape[0],mapim.shape[1]]), z_samples: np.tile((zsc)[np.newaxis,:,:],[5,1,1,1])})
         ims.append(dec_muv[0,:,:,0] )
         
              
         print("-----------------------------------------------------")
         
         
         
#              print("NOT SAVING!!!!!!!!")               
         
         if ix % saveevery == 0: 
              np.savez(directoryname+'/samples_'+str(int(np.floor(ix/saveevery))), \
                       ims=ims, p1s=p1s, q1s=q1s, acceptance=acceptance,\
                       mupSmups=mupSmups, pzs=pzs, full1s=full1s, full2s=full2s,\
                       full3s=full3s, p_y_xs=p_y_xs, grd_pz_norms=grd_pz_norms, \
                       grd_pyz_pz_norms=grd_pyz_pz_norms, zscs=zscs, sclfct_curs=sclfct_curs, nsksp = nsksp) # , gammavs=gammavs
              print("KCT-info: SAVED!")
              
              del ims
              ims=[]
              del zscs
              zscs=[]
#              if ix%10==0:    
#                   np.savez('/scratch_net/bmicdl02/modelsampling/arraysfolder/arrays', ims=ims, p1s=p1s, q1s=q1s, acceptance=acceptance, mupSmups=mupSmups, pzs=pzs, full1s=full1s, full2s=full2s, full3s=full3s, p_y_xs=p_y_xs)
#         print("str : " + str(time.time()-stt))


#          print("alphaval: {0:5f}, errtmpfv: {1:.4f}".format(alphaval, errtmpfv))            
     errtmpfvs.append(errtmpfv)
     print(errtmpfvs)
    
     acceptanceratios.append(   np.sum(np.array(acceptance))/len(acceptance)   )
     print("KCT-info: >>> step: "+str(step) +", accept ratio: " + str(np.sum(np.array(acceptance))/len(acceptance)))
     
     print("KCT-info: acceptance ratios are until now: ")
     print(acceptanceratios)
               
     return 0


     #     
          
     #def next_pow_two(n):
     #    i = 1
     #    while i < n:
     #        i = i << 1
     #    return i   
     #     
     #def autocorr_func_1d(x, norm=True):
     #    x = np.atleast_1d(x)
     #    if len(x.shape) != 1:
     #        raise ValueError("invalid dimensions for 1D autocorrelation function")
     #    n = next_pow_two(len(x))
     #
     #    # Compute the FFT and then (from that) the auto-correlation function
     #    f = np.fft.fft(x - np.mean(x), n=2*n)
     #    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
     #    acf /= 4*n
     #
     #    # Optionally normalize
     #    if norm:
     #        acf /= acf[0]
     #
     #    return acf     
     #     
     #def ac(x, lag):
     #    tmp1 = 0
     #    for ix in range(x.shape[0]-lag):
     #        tmp1 = tmp1 + x[ix]*x[ix+lag]
     #
     #    return tmp1
     #
     #def acc(x):
     #    tmp=[]
     #    for ix in range(x.shape[0]):
     #        tmp.append(ac(x,lag=ix))
     #
     #    tmp = np.array(tmp)
     #
     #    return tmp/tmp[0]
     #     
     #     
     #     