import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True 
import numpy as np
import os
import time as tm
import sys
import pickle


scriptdir = os.path.dirname(sys.argv[0])+'/'
#print("KCT-info: running script from directory: " + scriptdir)
os.chdir(scriptdir)




#def definevae_2d_v1_mri_nocov_fullim_conz_homodyn_varsize_f(batch_size=5, imsize=[-1, -1],  lowres=False, highprecmode=False, usemodelwithprecval=0, denoising=False, denoisingnoiseprec=400, heterosced=0, contleg=True, empiricalPrior=False, nginx=[], nskspval=1, sxval=1): 
def definevae_2d_v1_mri_nocov_fullim_conz_homodyn_varsize_f2(batch_size=5, im_kspsize=[[252,308], [252,308]],  lowres=False, BFcorr = False, heterosced=0,  empiricalPrior=False, nginx=[], nskspval=1, sxval=1, numinversioniters=3, rollval = 0): 
     
     scriptdir = os.path.dirname(sys.argv[0])+'/'
     #print("KCT-info: running script from directory: " + scriptdir)
     os.chdir(scriptdir)
    
     lat_dim=60
     
     nsksp=nskspval
     sx=sxval
     
     print("KCT-info: in the VAE function: empirical prior is " + str(empiricalPrior))
     
     str1=14 #4
     str2=14 #2
     
     ks1=19 #3
     ks2=19 #3
     
     no_filt = 64
     
     con_kernel_size=3#15
     
     denoising = False
     
     print("KCT-INFO: lat_dim value: "+str(lat_dim))
     
     print("KCT-INFO: rollval value: "+str(rollval))
          
#     os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
#     from tensorflow.python.client import device_lib
#     print (device_lib.list_local_devices())
#     print( os.environ['SGE_GPU'])
         
     tf.reset_default_graph()         
     
          
     
     nocoils=im_kspsize[1][2]
     
     
     imgsize=im_kspsize[0] # [252,308]#[182,210]
     kspsize=im_kspsize[1] # [237, 256] # [238,266]
#     imgsize=[237,256] # -> k-space
#     kspsize=[238,266] # image space
     
#     imgsize=[182,210]
#     kspsize=[237, 256] # [238,266]
     
#     imgsize=[252,308]#[182,210]
#     kspsize=[237, 256] # [238,266]
     
     kshval1 =  int(np.floor(kspsize[0]/2))
     kshval2 =  int(np.floor(kspsize[1]/2))
     
     
     #pad2edges0=int(np.floor((imgsizeM[0]-imgsize[0])/2 ))
     #pad2edges1=int(np.floor((imgsizeM[1]-imgsize[1])/2 ))
     
     pad2edges00=int(np.ceil((kspsize[0]-imgsize[0])/2 ))
     pad2edges01=int(np.floor((kspsize[0]-imgsize[0])/2 ))
     pad2edges10=int(np.ceil((kspsize[1]-imgsize[1])/2 ))
     pad2edges11=int(np.floor((kspsize[1]-imgsize[1])/2 ))
     
     pads = -np.array([ [pad2edges00,pad2edges01] , [pad2edges10,pad2edges11] ])
     
     
     
     uspat = tf.placeholder(tf.complex64, shape=[kspsize[0],kspsize[1]], name="pl2")
     sensmaps = tf.placeholder(tf.complex64, shape=[kspsize[0],kspsize[1],nocoils], name="pl1")  
     biasfield = tf.placeholder(tf.complex64, shape=[kspsize[0],kspsize[1]], name="biasfield")  
     
     
     #assert((imgsizeM[0]-imgsize[0]) == (imgsizeM[1]-imgsize[1]))
     
     #global sx
     #global ns
     
     def padslice_2d(array, padslice, rollval = 0):
          # a function that does padding or slicing depending on the pad values
          tmp = array #.copy()
          
          
          if rollval!=0:
              if padslice[1,0]<0 and padslice[1,1]<0:
                  tmp = tf.manip.roll(tmp, rollval, axis=1)
          
          mode='constant'
          
#          tf.slice(tmp1, [pad2edges00, pad2edges10, 0], [imgsize[0], imgsize[1], tf.shape(sensmaps)[2]])
          
          if padslice[0,0]>0:
               tmp = tf.pad(tmp, [  [padslice[0,0], 0], [0 ,0]  ], mode=mode)
          elif padslice[0,0]<0:
               tmp = tmp[-padslice[0,0]:, :]
          else:
               pass
          
          
          if padslice[0,1]>0:
               tmp = tf.pad(tmp, [  [0, padslice[0,1]], [0 ,0] ], mode=mode)
          elif padslice[0,1]<0:
               tmp = tmp[:padslice[0,1], :]
          else:
               pass
          
          
          if padslice[1,0]>0:
               tmp = tf.pad(tmp, [  [0 ,0], [padslice[1,0], 0]  ], mode=mode)
          elif padslice[1,0]<0:
               tmp = tmp[:, -padslice[1,0]:]
          else:
               pass
          
          
          if padslice[1,1]>0:
               tmp = tf.pad(tmp, [  [0 ,0],   [0, padslice[1,1]]  ], mode=mode)
          elif padslice[1,1]<0:
               tmp = tmp[:, :padslice[1,1]]
          else:
               pass
          
#          print(tmp.shape)
               
          if rollval!=0:
             if padslice[1,0]>0 and padslice[1,1]>0:
                 tmp = tf.manip.roll(tmp, -rollval, axis=1)
          
          return tmp
     
     def padslice_3d(array, padslice):
          # a function that does padding or slicing depending on the pad values
          # there is a 3rd dimension but it does not change in this function
          tmp = array.copy()
          
          mode='constant'
          
#          tf.slice(tmp1, [pad2edges00, pad2edges10, 0], [imgsize[0], imgsize[1], tf.shape(sensmaps)[2]])
          
          if padslice[0,0]>0:
               tmp = tf.pad(tmp, [  [padslice[0,0], 0], [0 ,0], [0 ,0]  ], mode=mode)
          elif padslice[0,0]<0:
               tmp = tmp[-padslice[0,0]:, :,:]
          else:
               pass
          
          
          if padslice[0,1]>0:
               tmp = tf.pad(tmp, [  [0, padslice[0,1]], [0 ,0], [0 ,0]  ], mode=mode)
          elif padslice[0,1]<0:
               tmp = tmp[:padslice[0,1], :,:]
          else:
               pass
          
          
          if padslice[1,0]>0:
               tmp = tf.pad(tmp, [  [0 ,0], [padslice[1,0], 0], [0 ,0]  ], mode=mode)
          elif padslice[1,0]<0:
               tmp = tmp[:, -padslice[1,0]:, :]
          else:
               pass
          
          
          if padslice[1,1]>0:
               tmp = tf.pad(tmp, [  [0 ,0],   [0, padslice[1,1]], [0 ,0]  ], mode=mode)
          elif padslice[1,1]<0:
               tmp = tmp[:, :padslice[1,1], :]
          else:
               pass
          
          
          return tmp
     
     
     def fftsh(im): 
          #innermost dims are shifted # in kspace
     #     tmp = im
          tmp = tf.manip.roll(im, [kshval1, kshval2] , axis=[1, 2])
#          return im
          return tmp
     
     
     
     def ifftsh(im):
          #innermost dims are shifted # in image space
     #     tmp =im
          tmp = tf.manip.roll(im, [-kshval1, -kshval2] , axis=[1, 2])
#          return im
          return tmp
     
     def FT_tf (x)   :
          #inp: [nx, ny]
          #out: [nx, ny, ns]
          
          
          
          xp = padslice_2d(x, -pads, rollval)
          
          
          if BFcorr:
               x2=xp*biasfield*phaseim*sclfactor # 
          else:
               x2=xp* phaseim*sclfactor # *biasfield
              
          tmp1 = sensmaps*tf.tile(x2[:,:,tf.newaxis],[1,1,sensmaps.shape[2]]) # [x,y,1]
          
#          tmp2 = tf.slice(tmp1, [pad2edges00, pad2edges10, 0], [imgsize[0], imgsize[1], tf.shape(sensmaps)[2]])
          #tmp2 = padslice_2d(tmp1, -pads) # slicing
          
          tmp3 = tf.transpose(tmp1, [2,0,1]) # [1,x,y]
                    
          tmp4 = (1/np.sqrt(kspsize[0]*kspsize[1]))* fftsh(tf.fft2d(tmp3)) #  (1/np.sqrt(imgsize[0]*imgsize[1]))
          
#          tmp4_1 = tf.reshape( tf.tensordot(tf.eye(nocoils), tf.reshape(tmp4, [nocoils, kspsize[0]*kspsize[1]]), axes=[0,0]  ), [nocoils, kspsize[0], kspsize[1]] )
#          
#          tmp5 = tf.transpose(tmp4_1, [1,2,0])
          
          tmp5 = tf.transpose(tmp4, [1,2,0])
          
              
          return tmp5
     
     def tFT_tf (x) :
          #inp: [nx, ny, ns]
          #out: [nx, ny]
          
          x = tf.transpose(x, [2,0,1]) # now [1,x,y] # a1
          
          temp = np.sqrt(kspsize[0]*kspsize[1]) * tf.ifft2d(  ifftsh( x ) ) # a2 # np.sqrt(imgsize[0]*imgsize[1])
          
          temp = tf.transpose(temp, [1,2,0]) # a3
          
#          temp = tf.pad(temp, [ [pad2edges00,pad2edges01] , [pad2edges10,pad2edges11], [0 ,0] ] , "CONSTANT") # a4
          
          
          aa1=tf.reduce_sum( temp*tf.cast(tf.conj(sensmaps),dtype=tf.complex64) , axis=2)  # a5
          aa2=tf.cast(   tf.reduce_sum(sensmaps*tf.conj(sensmaps),axis=2)    ,dtype=tf.complex64    )
          aa=aa1 # /aa2
          
          if BFcorr:
               qq = aa*biasfield/phaseim # 
          else:
               qq = aa/phaseim # *biasfield
          
          cc = padslice_2d(qq, pads, rollval ) # padding
          
     #     print(sclfactor)
          
          return cc*sclfactor      
     
     def UFT_tf(x   ):
          #inp: [nx, ny], [nx, ny]
          #out: [nx, ny, ns]
          
#          uspat = tf.cast(uspat, dtype=tf.complex64) #.astype(np.complex64)
          
          tmp=x
          
          tmp2 = tf.tile(uspat[:,:,tf.newaxis],[1,1,sensmaps.shape[2]])
          
          tmp3 = FT_tf(tmp)
          
          return tmp2*tmp3
     
     def tUFT_tf(x    ):
          #inp: [nx, ny], [nx, ny]
          #out: [nx, ny]
          
     #     print(uspat.shape)
          
          tmp1 = tf.tile(uspat[:,:,tf.newaxis],[1,1,sensmaps.shape[2]])
          tmp1 = tf.cast(tmp1, dtype=tf.complex64)
          
          return  tFT_tf( tmp1*x)
     
     def EHE_tf(im   ):
          
          #nskspt = nsksp[np.newaxis,np.newaxis, :]
     
         return tUFT_tf(nsksp*UFT_tf(im)) 
     
     def AHA_tf(im  ):
     
         tmp1 = im*sx*sx
         tmp2 = sx*EHE_tf(im)
         tmp3 = sx*EHE_tf(im)
         tmp4 = EHE_tf(EHE_tf(im))         
         return tmp1+tmp2+tmp3+tmp4         
     
     def A_tf(im):
             
         return im*sx + EHE_tf(im)  
     
     def grad_tf(im, m):
       
         return AHA_tf(im) - A_tf(m)
         
     def error_tf(im, m):
          return tf.linalg.norm(A_tf(im ) - m)  
     
     def cdp_tf(a,b):
          return tf.reduce_sum(  tf.multiply( a, tf.conj(b) )  )
     
     def findgamma_cg2(mm, numiters=5 ):  
          
          print("KCT-info: no of inversion iterations: " + str(numiters))
              
                      
          
          b = tf.cast(tf.abs(mm), tf.complex64)
          
          #init:
     #     its = b    
          its = tf.cast(tf.zeros_like(b), tf.complex64)
          
          errtmp0 = tf.linalg.norm(A_tf(its) - b) / tf.linalg.norm(b)*100
          
          r = b - A_tf(its)   
          p = r # .copy()
             
          for ix in range(numiters):
               
               rTr_curr = cdp_tf(r,r) # np.sum(r*r)
               alpha = rTr_curr / cdp_tf(p, A_tf(p)) # np.sum(p*A(p, uspat))
               its = its+ alpha* p
               
               r = r - alpha*A_tf(p)
               
               beta = cdp_tf(r,r)/rTr_curr
               
               p = r +beta * p
               
          errtmp = tf.linalg.norm(A_tf(its) - b) / tf.linalg.norm(b)*100
               
          return  its, errtmp, errtmp0
     
#     def findgamma_tf6(mm, phaseim, sclfactor, numiters=5, alpha = 1e-4):  
#          #initialize with dec_mu
#          
#          # this function returns in "its" the term:
#          # 
#          
#     #     uspattf = tf.Variable(name='uspattf'+str(np.random.randint(0,1000)), initial_value=uspat)
#          uspattf = tf.placeholder(tf.complex64, shape=[imgsize[0],imgsize[1]], name="pl2")
#          sensmapspl = tf.placeholder(tf.complex64, shape=[imgsizeM[0],imgsizeM[1],nocoils], name="pl1")
#     #     mpl = tf.placeholder(tf.complex64, shape=[252, 308])
#          
#          # mm = sx*tf.cast(dec_mu[0,:,:,0],dtype=tf.complex64)
#          mpl=mm
#          
#          a1 = A_tf(mpl, uspattf, sensmapspl, phaseim, sclfactor)
#          a2=tf.linalg.norm(a1 - mpl)
#          a3 = tf.linalg.norm(mpl)
#          
#          
#          errtmp0 = a2 /a3 *100  
#                    
#          grd1 = grad_tf(mpl, mpl, uspattf, sensmapspl, phaseim, sclfactor)
#          its1 =  mpl - alpha*grd1     
#          
#          its = its1                                                                
#          
#          
#          
#          for ix in range(numiters-1):
#               grd = grad_tf(its, mpl, uspattf, sensmapspl, phaseim, sclfactor)
#               itsnew =  its - alpha*grd
#               its = itsnew
#                       
#          errtmp = tf.linalg.norm(A_tf(its, uspattf, sensmapspl, phaseim, sclfactor) - mpl) / tf.linalg.norm(mpl)*100   
#     
#     #     gg = tf.gradients(errtmp, mpl, grad_ys=tf.ones([1], dtype=tf.complex64))
#          
#          return uspattf, sensmapspl, its, errtmp, errtmp0,grd1, a1,a2,a3#, gg[0]
          
     
     
     
     
     

     #define the input place holder                                                 
     x_inp = tf.placeholder("float", shape=[None, imgsize[0], imgsize[1]], name="pl3")
     
     
     # a. build the encoder layers
     
     x_inp_ = tf.reshape(x_inp, [tf.shape(x_inp)[0], tf.shape(x_inp)[1],  tf.shape(x_inp)[2], 1])
     
     ### Encoder
     conv1 = tf.layers.conv2d(inputs=x_inp_, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu) # 
     conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu) # 
     conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu) # 
     conv4 = tf.layers.conv2d(inputs=conv3, filters=no_filt, kernel_size=(3,3), padding='same', activation=tf.nn.relu) # 
     enc_mu = tf.layers.conv2d(inputs = conv4, filters=lat_dim, kernel_size=(ks1,ks2), strides=(str1,str2), padding='same', activation=None)
     
     enc_logstd = tf.layers.conv2d(inputs = conv3, filters=lat_dim, kernel_size=(ks1,ks2), strides=(str1,str2), padding='same', activation=None)
     enc_std = tf.exp(0.5*enc_logstd)  
     
     eps = tf.random_normal(tf.shape(enc_std))           
     
     z_samples = enc_mu  + tf.multiply(eps, enc_std)
     
     z_samplesr = tf.reshape(z_samples,[tf.shape(x_inp)[0], -1]) # to force the 1d representation in between
#     z_samplesrr = tf.reshape(z_samplesr,[tf.shape(x_inp)[0],z_samples.get_shape()[1], z_samples.get_shape()[2], z_samples.get_shape()[3]])
                                      
     # decoder 
     dec1_0 = tf.layers.conv2d(inputs=z_samples, filters=no_filt*str1*1, kernel_size=(con_kernel_size,con_kernel_size), padding='same', activation=None )
     dec1_1 = tf.transpose(dec1_0,perm=[0,1,3,2])
     dec1_2 = tf.reshape(dec1_1, [tf.shape(x_inp)[0], tf.shape(x_inp)[1],  no_filt, tf.cast((tf.shape(x_inp)[2]/str2), dtype=tf.int32)])
     dec1_3=tf.transpose(dec1_2, perm=[0,1,3,2])
     
     dec1_4 = tf.layers.conv2d(inputs=dec1_3, filters=no_filt*1*str2, kernel_size=(con_kernel_size,con_kernel_size), padding='same', activation=None )
     dec1_5 = tf.reshape(dec1_4, [tf.shape(x_inp)[0], tf.shape(x_inp)[1], tf.shape(x_inp)[2], no_filt])
     
     dec1 = tf.nn.relu(dec1_5)
     
     if heterosced==0:
     
          dec2 = tf.layers.conv2d(inputs=dec1, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
          dec3 = tf.layers.conv2d(inputs=dec2, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
          dec4 = tf.layers.conv2d(inputs=dec3, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
          
          dec5 = tf.layers.conv2d(inputs=dec4, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
          dec6 = tf.layers.conv2d(inputs=dec5, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
          dec7 = tf.layers.conv2d(inputs=dec6, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
          
          dec_mu = tf.layers.conv2d(inputs=dec7, filters=1, kernel_size=(3,3), padding='same', activation=None) # Now 28x28x32
          dec_mur = tf.reshape(dec_mu, [tf.shape(x_inp)[0], tf.shape(x_inp)[1]*tf.shape(x_inp)[2]])
          dec_murr = tf.reshape(dec_mur, [tf.shape(x_inp)[0], tf.shape(x_inp)[1], tf.shape(x_inp)[2], 1 ])
               
          ########
          
          y_out = tf.contrib.layers.flatten(dec_murr)
          
          
          if denoising==True:
               x_noiseless = tf.placeholder("float", shape=[None, None, None])
     
     elif heterosced==1:
          dec2 = tf.layers.conv2d(inputs=dec1, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
          dec3 = tf.layers.conv2d(inputs=dec2, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
          dec4 = tf.layers.conv2d(inputs=dec3, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
          
          dec5 = tf.layers.conv2d(inputs=dec4, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
          dec6 = tf.layers.conv2d(inputs=dec5, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
          dec7 = tf.layers.conv2d(inputs=dec6, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
          
          dec5c = tf.layers.conv2d(inputs=dec4, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
          dec6c = tf.layers.conv2d(inputs=dec5c, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
          dec7c = tf.layers.conv2d(inputs=dec6c, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
          
          dec_mu = tf.layers.conv2d(inputs=dec7, filters=1, kernel_size=(3,3), padding='same', activation=None) # Now 28x28x32
          
          dec_mur = tf.reshape(dec_mu, [tf.shape(x_inp)[0], tf.shape(x_inp)[1]*tf.shape(x_inp)[2]])
          dec_murr = tf.reshape(dec_mur, [tf.shape(x_inp)[0], tf.shape(x_inp)[1], tf.shape(x_inp)[2], 1 ])
          
          dec_logprec = tf.layers.conv2d(inputs=dec7c, filters=1, kernel_size=(3,3), padding='same', activation=None) # Now 28x28x32
          dec_prec  = tf.exp(dec_logprec)
          
          ########
          
          y_out = tf.contrib.layers.flatten(dec_mu)
          
          y_out_prec = tf.contrib.layers.flatten(dec_prec)
          
     elif heterosced==2:
          dec2 = tf.layers.conv2d(inputs=dec1, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
          dec3 = tf.layers.conv2d(inputs=dec2, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
          dec4 = tf.layers.conv2d(inputs=dec3, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
          
          dec5 = tf.layers.conv2d(inputs=dec4, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
          dec6 = tf.layers.conv2d(inputs=dec5, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
          dec7 = tf.layers.conv2d(inputs=dec6, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
          
          dec7c = tf.layers.conv2d(inputs=dec6, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
          
          dec_mu = tf.layers.conv2d(inputs=dec7, filters=1, kernel_size=(3,3), padding='same', activation=None) # Now 28x28x32
          
          dec_mur = tf.reshape(dec_mu, [tf.shape(x_inp)[0], tf.shape(x_inp)[1]*tf.shape(x_inp)[2]])
          dec_murr = tf.reshape(dec_mur, [tf.shape(x_inp)[0], tf.shape(x_inp)[1], tf.shape(x_inp)[2], 1 ])
          
          dec_logprec = tf.layers.conv2d(inputs=dec7c, filters=1, kernel_size=(3,3), padding='same', activation=None) # Now 28x28x32
          dec_prec  = tf.exp(dec_logprec)
          
          y_out = tf.contrib.layers.flatten(dec_mu)

          y_out_prec = tf.contrib.layers.flatten(dec_prec)
     
     else:
          assert(1==0)
          
     
     
     
     # build the loss functions and the optimizer
     #==============================================================================   
     #==============================================================================   
     
     
     ##KLD loss
     #dets = tf.linalg.logdet(enc_sigma) # only on GPU
     #dets = tf.reduce_sum(dets,axis=1)
     #dets=tf.cast(dets,tf.float32)
     #
     #trc_tmp = tf.matrix_diag_part(enc_sigma)
     #trc = tf.reduce_sum(trc_tmp, axis=[1,2])
     #
     #muTmu = tf.reduce_sum( tf.square(enc_mu), axis= [1,2,3])
     #
     #KLD = 0.5* (-dets + trc + muTmu) # 
     
     enc_logstdf = tf.contrib.layers.flatten(enc_logstd)
     enc_muf = tf.contrib.layers.flatten(enc_mu)
     KLD = -0.5 * tf.reduce_sum(1 + enc_logstdf - tf.pow(enc_muf, 2) - tf.exp(enc_logstdf), reduction_indices=1)
                          
     # L2 loss of the batch
#     if denoising:
#          x_inp_rsp = tf.reshape(x_noiseless, [tf.shape(x_inp)[0], tf.shape(x_inp)[1]*tf.shape(x_inp)[2] ])
#     else:
     x_inp_rsp = tf.reshape(x_inp, [tf.shape(x_inp)[0], tf.shape(x_inp)[1]*tf.shape(x_inp)[2] ])
     
     
     if heterosced==0:
          l2_loss_ = 0.5*tf.reduce_sum(tf.pow((y_out - x_inp_rsp),2)*50,axis=1)
     elif heterosced==1 or heterosced==2:
          # to copy from:
          l2_loss_1 = tf.reduce_sum(tf.multiply(tf.pow((y_out - x_inp_rsp),2), y_out_prec),axis=1)
          l2_loss_2 = tf.reduce_sum(tf.log(y_out_prec), axis=1) #tf.reduce_sum(tf.log(y_out_cov),axis=1)
          l2_loss_ = 0.5*(l2_loss_1 - l2_loss_2)
     else:
          assert(1==0)
          
     
     
          
     # take the total mean loss of this batch
     loss_tot = tf.reduce_mean(   KLD + l2_loss_) # #KCT KLD +
     
     
     
     sess = tf.InteractiveSession()
     
     
     sess.run(tf.global_variables_initializer())
     
     
     
     
     # do the training
     #============================================================================== 
     #============================================================================== 
     
     
     saver = tf.train.Saver(max_to_keep=0)
     
     
     
     
     print("KCT-info: restoring model")
     
     print("KCT-info: current directory: " + os.getcwd())

     if BFcorr == True and lowres == False:
          step=1900000
          modelname = 'cvae2d_mri_s14k19_fullconv_nocov_fullim_conz33_homodyn_60ulargedec_252x308_xb5_lat60_varsize_bigDS_aug_new_bfc_noiseprec50_l2prec50_denFalse_step'+str(step)
          saver.restore(sess, os.getcwd() + '/../../trained_models/'+modelname)# 
          print("KCT-info:  Loading legecy model, with Bf corr, high res: " + modelname)
     elif BFcorr == True and lowres == True:
          step=1750000
          modelname = 'cvae2d_mri_s14k19_fullconv_nocov_fullim_conz33_homodyn_60ulargedec_252x308_xb5_lat60_varsize_bigDS_aug_new_lowres_bfc_noiseprec50_l2prec50_denFalse_step'+str(step)
          saver.restore(sess, os.getcwd() + '/../../trained_models/'+modelname)# 
          print("KCT-info:  Loading legecy model, with Bf corr, low res: " + modelname)
     else:
          print("KCT-error: I don't have a trined model for settings, except BFCorr=True and lowres=True/False")  
          raise ValueError
     
     
     
     print("KCT-info: restored")
     
     
     phaseim = tf.placeholder("complex64", shape=[kspsize[0],kspsize[1]], name="phaseimpl") # pl4

     #define the necessary operations and gradients
     op_p_x_z = tf.reduce_mean(l2_loss_)
     
     op_p_x_z_cs= -(1/0.05)*0.5*tf.reduce_sum(tf.pow((y_out - x_inp_rsp),2),axis=1)
     
     
     op_q_z_x_ = (- 0.5 * tf.reduce_sum(tf.multiply(tf.pow((z_samples - enc_mu),2), tf.reciprocal(enc_std)),axis=[1,2,3]) \
                       - 0.5 * tf.reduce_sum(tf.log(enc_std), axis=[1,2,3]))
     op_q_z_x = tf.reduce_mean(op_q_z_x_)
     
     op_p_z_ = (- 0.5 * tf.reduce_sum(tf.multiply(tf.pow((z_samples - tf.zeros_like(enc_mu)),2), tf.reciprocal(tf.ones_like(enc_std))),axis=[1,2,3]) \
                       - 0.5 * tf.reduce_sum(tf.log(tf.ones_like(enc_std)), axis=[1,2,3]) )
     op_p_z = tf.reduce_mean(op_p_z_)
     
     funop = op_p_x_z + op_p_z - op_q_z_x
     grd0 = tf.gradients(op_p_x_z + op_p_z - op_q_z_x, x_inp)[0]
     grd_p_x_z0 = tf.gradients(op_p_x_z, x_inp)[0]
     grd_p_z0 = tf.gradients(op_p_z, x_inp)[0]
     grd_q_z_x0 = tf.gradients(op_q_z_x, x_inp)[0]
     grd_op_p_x_z_cs0 = tf.gradients(op_p_x_z_cs, x_inp)[0]
     
     #     funop_cs = op_p_x_z_cs + op_p_z - op_q_z_x
     
     grd_cs0 = tf.gradients(op_p_x_z_cs + op_p_z - op_q_z_x, x_inp)[0]
     
     oplt = KLD + l2_loss_
     grd_lt0 = tf.gradients(KLD + l2_loss_ , x_inp)[0]
     
     
     
     ##### the stuff for the empirical p(z)
     
     if empiricalPrior:
     
          nginx = tf.placeholder(tf.int32, shape=[None], name="nginx") 
          covng_inv = tf.placeholder(tf.float64, shape=[None,  None], name="covng_inv") 
          meanng = tf.placeholder(tf.float64, shape=[None], name="meanng") 
          covs_invs = tf.placeholder(tf.float64, shape=[None, None, None], name="covs_invs") 
          means = tf.placeholder(tf.float64, shape=[None, None], name="means") 
          
          def Ci_vec_prod(vec):
               # z: [18x22x60]# mean_comb = meanng
               # do all in the [60,18,22] order, then convert back
               
               
     #          vec = np.transpose(vec, [2,0,1]) # [60,18,22]
               vec = tf.transpose(vec[0,:,:,:], [2,0,1])
               
               vec=tf.cast(vec, dtype=tf.float64)
               
               # first do the fully correlated ones
               ng_zs = tf.reshape(tf.gather(vec, nginx), [-1]) # vec[nginx,:,:].flatten()
               dtemp1 = tf.cast(ng_zs, tf.float64)
               dtemp2 = meanng # tf.constant(meanng,dtype=tf.float64)
               diff = dtemp1 - dtemp2
               ng_zs1 = tf.matmul(covng_inv, diff[:,tf.newaxis])
               ng_zs2 = tf.reshape(ng_zs1[:,tf.newaxis], [10, 18, 22]) #KCT: TODO: 10 <- nginx.shape[0]
               
               
               # now do the channel-uncorrelated ones
               tmp=[]
               ctr=0
               for ix in range(60):
                    if ix in nginx:
                         tmp.append(ng_zs2[ctr,:,:])
                         ctr=ctr+1
                    else:
                         tmp.append( tf.reshape(tf.matmul( covs_invs[ix,:,:] , (tf.reshape(vec[ix,:,:],[-1]) - means[ix,:])[:,tf.newaxis]  ), [18,22]  )    )# [:,tf.newaxis]
               tmp2 = tf.stack(tmp)
               tmp2 = tf.reshape(tmp2,[60,18,22])
               
                         
               prod = tf.transpose(tmp2, [1,2,0]) # back to [18,22,11]
               
               #finally just get the z-mu:
               dtmp = vec - tf.reshape(means,[60,18,22])
               diff = tf.transpose(  dtmp , [1,2,0]  )
               
               return prod, diff  
               
          def grad_pz(z):
               prod, _ = Ci_vec_prod(z)
               return (-1)*prod
          
          def norm_pz(z):
               prod, diff = Ci_vec_prod(z)
               return (-0.5)*tf.reduce_sum(diff*prod)
          
          ##### the stuff for the empirical p(z) - END
     
     
     
     
     print("KCT-INFO: the gradients: ")    
     print(grd_p_x_z0) 
     print(grd_p_z0) 
     print(grd_q_z_x0)    
     
     
     sclfactor = tf.placeholder(tf.complex64, shape=[1], name="pl6") 
     
     alphaiter = tf.placeholder(tf.complex64, shape=[], name="alphaiter") 
     
     
     
     
     
#     print("shape of what goes into the inversion: ")
     print(tf.cast(dec_mu[0,:,:,0],dtype=tf.complex64).shape)
     
     # make here the iterative solution network
     # for HCP image: numiters=25
#     uspattff, sensmapsplf, itsf, errtmpf, errtmp0f, grd1f, a1f,a2f,a3f  = findgamma_tf6(sx*tf.cast(dec_mu[0,:,:,0],dtype=tf.complex64), phaseim, sclfactor, numiters=30, alpha = alphaiter) # 1e-3
     itsf, errtmpf, errtmp0f  = findgamma_cg2(sx*tf.cast(dec_mu[0,:,:,0],dtype=tf.complex64), numiters=numinversioniters) # 1e-3 # iters=35


     # for 1/1: numiters=20, alpha = 4e-1
     
     #create the mu_postH * Sigma_post * mu_post operation
     yy = tf.placeholder(tf.complex64, shape=[None,  None, nocoils], name="pl5") 
     
     gamma = itsf    
     
     full1 = tf.reduce_sum(sx*tf.cast(dec_mu[0,:,:,0],dtype=tf.complex64)*gamma ,axis=[0,1])
     
     full2_1 = tf.reduce_sum(tf.conj(yy)*nsksp*UFT_tf(gamma) ,axis=[0,1,2]) 
     
     full2_2 = tf.conj(  tf.reduce_sum(tf.conj(yy)*nsksp*UFT_tf(gamma) ,axis=[0,1,2])   )
     
     full2 =  full2_1 + full2_2
     
     full3 = sx*tf.reduce_sum(tf.conj(dec_mu[0,:,:,0])*dec_mu[0,:,:,0], axis=[0,1])
     
     mupost_tmp1 = tUFT_tf(nsksp*yy)
     mupost_tmp2 = sx*tf.cast(dec_mu[0,:,:,0],dtype=tf.complex64)
     mupost_zy = findgamma_cg2( tUFT_tf(nsksp*yy) +  sx*tf.cast(dec_mu[0,:,:,0],dtype=tf.complex64), numiters=numinversioniters) 
     
     
     mupSmup = 0.5* (full1 + full2 - tf.cast(full3, dtype=tf.complex64))
     
     grd_mupSmup = tf.gradients(mupSmup, z_samples, grad_ys=tf.ones([1], dtype=tf.complex64))
     
     p_y_x = - tf.square( tf.linalg.norm(UFT_tf(tf.cast(dec_mu[0,:,:,0],dtype=tf.complex64)) - yy) )
     
     
     ttt11 =   tf.conj(tf.cast(dec_mu[0,:,:,0],dtype=tf.complex64))
     ttt12 = tUFT_tf(yy)  
     
     optimalscale_tmp1 = tf.reduce_sum(    ttt11   *  ttt12     )
     
     optimalscale_tmp2 = tf.reduce_sum(  tf.conj(tf.cast(dec_mu[0,:,:,0],dtype=tf.complex64))   *  tUFT_tf(  UFT_tf( tf.cast(dec_mu[0,:,:,0],dtype=tf.complex64) )     )          )
     optimalscale = optimalscale_tmp1 / optimalscale_tmp2
     
#     muxSmux =  tf.cast(full3, dtype=tf.complex64)
     
#     grd_muxSmux = tf.gradients(muxSmux, z_samples, grad_ys=tf.ones([1], dtype=tf.complex64))
     
     
     #for debugging ---------------
     #for debugging ---------------
     
     
     dbgim = tf.placeholder(tf.complex64, shape = imgsize, name="dbgim")
     
#     print("dbgim shape: ")
#     print(dbgim.shape)
     
     ufttf = FT_tf(dbgim) # A_tf(dbgim)
#     bb = tFT_tf(FT_tf(tf.cast(dec_mu[0,:,:,0], tf.complex64))) # tFT_tf( )
     bb = tFT_tf(FT_tf(dbgim))# tFT_tf( )
     
     
     #for debugging ---------------
     #for debugging ---------------
     
     if empiricalPrior:
#          op_p_z_emp = norm_pz(z_samples)
#          funop_emp = op_p_x_z + tf.cast(op_p_z_emp,tf.float32) - op_q_z_x
#          grd0_emp = tf.gradients(op_p_x_z + tf.cast(op_p_z_emp,tf.float32) - op_q_z_x, x_inp)[0]
          
          op_p_z_emp = norm_pz(z_samples)
          funop_emp = op_p_x_z + tf.cast(op_p_z_emp,tf.float32) - op_q_z_x
          grd0_emp = tf.gradients(op_p_x_z + tf.cast(op_p_z_emp,tf.float32) - op_q_z_x, x_inp)[0]
     
     
     if heterosced ==0:
          precret = 0
     elif heterosced==1 or heterosced==2:
          precret=dec_prec
     else:
          assert(1==0)
     
     if empiricalPrior:
          return dec_mu, precret, x_inp, grd0, funop, grd_p_x_z0, grd_p_z0, grd_q_z_x0, \
                 grd_op_p_x_z_cs0, grd_cs0, enc_mu, enc_std, z_samples, \
                 z_samplesr, dec_mur, sess, 0, uspat, sensmaps, itsf, \
                 errtmpf, 0, mupSmup, grd_mupSmup, yy,full1,full2, errtmp0f, 0, \
                 dec_mu[0,:,:,0], gamma, full3, p_y_x, phaseim, sclfactor, \
                 op_p_z_emp, funop_emp, grd0_emp, covng_inv, meanng, \
                 covs_invs, means, alphaiter, dbgim, ufttf, bb, biasfield,optimalscale, \
                 optimalscale_tmp1, optimalscale_tmp2, ttt11, ttt12, mupost_zy, mupost_tmp1, mupost_tmp2 #, muxSmux, grd_muxSmux
     else:
          return dec_mu, precret, x_inp, grd0, funop, grd_p_x_z0, grd_p_z0, grd_q_z_x0, \
                 grd_op_p_x_z_cs0, grd_cs0, enc_mu, enc_std, z_samples, \
                 z_samplesr, dec_mur, sess, 0, uspat, sensmaps, itsf, \
                 errtmpf, 0, mupSmup, grd_mupSmup, yy,full1,full2, errtmp0f, 0, \
                 dec_mu[0,:,:,0], gamma, full3, p_y_x, phaseim, sclfactor, \
                 alphaiter, dbgim, ufttf, bb, biasfield, optimalscale, \
                 optimalscale_tmp1, optimalscale_tmp2, ttt11, ttt12, mupost_zy, mupost_tmp1, mupost_tmp2 #, muxSmux, grd_muxSmux
   
       
          
#
#def findgamma_tf4(numiters=5, alpha = 1e-4):  
#     
##     uspattf = tf.Variable(name='uspattf'+str(np.random.randint(0,1000)), initial_value=uspat)
#     uspattf = tf.placeholder(tf.complex64, shape=[252,308])
#     sensmapspl = tf.placeholder(tf.complex64, shape=[252,308,1])
#     mpl = tf.placeholder(tf.complex64, shape=[252, 308])
#     
#               
#     grd1 = grad_tf(tf.zeros([252,308], dtype=tf.complex64), mpl, uspattf, sensmapspl)
#     its1 =  tf.zeros([252,308], dtype=tf.complex64) - alpha*grd1     
#     
#     its = its1
#     for ix in range(numiters-1):
#          grd = grad_tf(its, mpl, uspattf, sensmapspl)
#          itsnew =  its - alpha*grd
#          its = itsnew
#                  
#     errtmp = tf.linalg.norm(A_tf(its, uspattf, sensmapspl) - mpl) / tf.linalg.norm(mpl)*100   
#
##     gg = tf.gradients(errtmp, mpl, grad_ys=tf.ones([1], dtype=tf.complex64))
#     
#     return mpl, uspattf, sensmapspl, its, errtmp#, gg[0]
#
#def findgamma_tf5(mm, numiters=5, alpha = 1e-4):  
#     #initialize with zero
#     
#     # this function returns in "its" the term:
#     # 
#     
##     uspattf = tf.Variable(name='uspattf'+str(np.random.randint(0,1000)), initial_value=uspat)
#     uspattf = tf.placeholder(tf.complex64, shape=[252,308])
#     sensmapspl = tf.placeholder(tf.complex64, shape=[252,308,1])
##     mpl = tf.placeholder(tf.complex64, shape=[252, 308])
#     
#     mpl=mm
#     
#     errtmp0 = tf.linalg.norm(A_tf(tf.zeros([252,308], dtype=tf.complex64), uspattf, sensmapspl) - mpl) / tf.linalg.norm(mpl)*100  
#               
#     grd1 = grad_tf(tf.zeros([252,308], dtype=tf.complex64), mpl, uspattf, sensmapspl)
#     its1 =  tf.zeros([252,308], dtype=tf.complex64) - alpha*grd1     
#     
#     its = its1                                                                
#     
#     
#     
#     for ix in range(numiters-1):
#          grd = grad_tf(its, mpl, uspattf, sensmapspl)
#          itsnew =  its - alpha*grd
#          its = itsnew
#                  
#     errtmp = tf.linalg.norm(A_tf(its, uspattf, sensmapspl) - mpl) / tf.linalg.norm(mpl)*100   
#
##     gg = tf.gradients(errtmp, mpl, grad_ys=tf.ones([1], dtype=tf.complex64))
#     
#     return uspattf, sensmapspl, its, errtmp, errtmp0,grd1#, gg[0]
#
