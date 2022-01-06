import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True 
import numpy as np
import os
import time as tm
import sys
import pickle
import scipy as sc

import matplotlib.pyplot as plt


lat_dim=60
input_dim=[252, 308]
batch_size=5
nzsamp=1    
padding='SAME'
rank=16

denoising=False
denoisingnoiseprec=-1

str1=14 #4
str2=14 #2

ks1=19 #3
ks2=19 #3

no_filt = 64

con_kernel_size=3

if denoising==True:
     prec=denoisingnoiseprec
else:
     prec=50

#from tensorflow.examples.tutorials.mnist import input_data

#from MRIDataset import MRIDataset

from MR_image_data_v3 import MR_image_data_v3

#from Dataset import Dataset

conttrain=True
contstep=750000
conttraindir = '/scratch_net/bmicdl02/modelrecon_2d/models/cvae2d_mri_s14k19_fullconv_nocov_fullim_conz33_homodyn_60ulargedec_252x308_xb5_lat60_varsize_bigDS_aug_step'+str(contstep)


SEED=1          
std_init=0.1   


noisy=0           
mode='MRIunproc'

print("KCT-INFO: lat_dim value: "+str(lat_dim))
     
#os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
#from tensorflow.python.client import device_lib
#print (device_lib.list_local_devices())
#print( os.environ['SGE_GPU'])
     
num_inp_channels=1   

tf.reset_default_graph()



#define the input place holder                                                 
x_inp = tf.placeholder("float", shape=[None, None, None])

intl=tf.truncated_normal_initializer(stddev=std_init, seed=SEED)


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
            
                     
# decoder 
dec1_0 = tf.layers.conv2d(inputs=z_samples, filters=no_filt*str1*1, kernel_size=(con_kernel_size,con_kernel_size), padding='same', activation=None )
dec1_1 = tf.transpose(dec1_0,perm=[0,1,3,2])
dec1_2 = tf.reshape(dec1_1, [tf.shape(x_inp)[0], tf.shape(x_inp)[1],  no_filt, tf.cast((tf.shape(x_inp)[2]/str2), dtype=tf.int32)])
dec1_3=tf.transpose(dec1_2, perm=[0,1,3,2])

dec1_4 = tf.layers.conv2d(inputs=dec1_3, filters=no_filt*1*str2, kernel_size=(con_kernel_size,con_kernel_size), padding='same', activation=None )
dec1_5 = tf.reshape(dec1_4, [tf.shape(x_inp)[0], tf.shape(x_inp)[1], tf.shape(x_inp)[2], no_filt])

dec1 = tf.nn.relu(dec1_5)


dec2 = tf.layers.conv2d(inputs=dec1, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
dec3 = tf.layers.conv2d(inputs=dec2, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
dec4 = tf.layers.conv2d(inputs=dec3, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )

dec5 = tf.layers.conv2d(inputs=dec4, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
dec6 = tf.layers.conv2d(inputs=dec5, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )
dec7 = tf.layers.conv2d(inputs=dec6, filters=60, kernel_size=(3,3), padding='same', activation=tf.nn.relu )

dec_mu = tf.layers.conv2d(inputs=dec7, filters=1, kernel_size=(3,3), padding='same', activation=None) # Now 28x28x32
 

########

y_out = tf.contrib.layers.flatten(dec_mu)


if denoising==True:
     x_noiseless = tf.placeholder("float", shape=[None, None, None])


# build the loss functions and the optimizer
#==============================================================================   
#==============================================================================    


enc_logstdf = tf.contrib.layers.flatten(enc_logstd)
enc_muf = tf.contrib.layers.flatten(enc_mu)
KLD = -0.5 * tf.reduce_sum(1 + enc_logstdf - tf.pow(enc_muf, 2) - tf.exp(enc_logstdf), reduction_indices=1)
                     
# L2 loss of the batch
if denoising:
     x_inp_rsp = tf.reshape(x_noiseless, [tf.shape(x_inp)[0], tf.shape(x_inp)[1]*tf.shape(x_inp)[2] ])
else:
     x_inp_rsp = tf.reshape(x_inp, [tf.shape(x_inp)[0], tf.shape(x_inp)[1]*tf.shape(x_inp)[2] ])


l2_loss_ = 0.5*tf.reduce_sum(tf.pow((y_out - x_inp_rsp),2)*prec,axis=1)


     
# take the total mean loss of this batch
loss_tot = tf.reduce_mean(   KLD + l2_loss_) # #KCT KLD +


# get the optimizer
train_step = tf.train.AdamOptimizer(1e-4 ).minimize(loss_tot) #  # 5e-4  # KCT





sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


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

# do the training
#============================================================================== 
#============================================================================== 


#MDS = MRIDataset(-1, -1, -1, 0, 1, 'MRIunproc')
#test_batch = MDS.get_test_batch_image(batch_size)  
#test_batch = MDS.get_test_batch_image(batch_size)        
##test_batch = test_batch[:, 7:-7, 27:-29]  # [:,112:-112,140:-140] #


if denoising==True:
     MRi = MR_image_data_v3(dirname='/scratch_net/bmicdl02/Data/data4fullvol_2d/', imgSize = [260, 311, 260], testchunks = [39], noiseinvstd=0)
else:
     MRi = MR_image_data_v3(dirname='/scratch_net/bmicdl02/Data/data4fullvol_2d/', imgSize = [260, 311, 260], testchunks = [39], noiseinvstd=prec ) # np.sqrt(prec))


#test_batch = resizebatch(MRi.get_batch(batch_size, test=True))
#test_batch = np.transpose( test_batch[:, 4:214, :], [2,0,1] )

test_batch = MRi.get_batch(batch_size, test=True)
test_batch = np.transpose( test_batch[4:-4, 1:-2, :], [2,0,1] )

if denoising:
     test_batch_noisy = test_batch + (1/np.sqrt(denoisingnoiseprec))*np.random.randn(test_batch.shape[0],test_batch.shape[1],test_batch.shape[2])

ts=tm.time()
tt=tm.time()

saver = tf.train.Saver(max_to_keep=0)
beginiter=0

if conttrain:
     saver.restore(sess,conttraindir)
     beginiter=contstep
     print("restored parameters from file: " + conttraindir)
     
#pickle.dump(test_batch, open('/scratch_net/bmicdl02/deletelater/todelete/tb','wb' ))

with tf.device('/gpu:0'):
     
     #train for N steps
     for step in range(0, 2000001):#250000):
     
     
#          #debug:
#         Kstv = Kst.eval(feed_dict={x_inp: test_batch})
#         Lpv = Lp.eval(feed_dict={x_inp: test_batch})
#         if step == 0:
#              pickle.dump(Kstv, open('/home/ktezcan/unnecessary_stuff/deletethis/Kstv','wb'))
#              pickle.dump(Lpv, open('/home/ktezcan/unnecessary_stuff/deletethis/Lpv','wb'))
         
         
#         batch = resizebatch(MRi.get_batch(batch_size, test=False))
#         batch = np.transpose( batch[:, 4:214, :], [2,0,1] )
         
         batch = MRi.get_batch(batch_size, test=False)
         batch = np.transpose( batch[4:-4, 1:-2, :], [2,0,1] )
         
         #some small translation augmentation
         batch = np.roll(batch, [np.random.randint(-4,4), np.random.randint(-4,4)], [1,2])
         
         if denoising:
              batch_noisy = batch + (1/np.sqrt(denoisingnoiseprec))*np.random.randn(batch.shape[0],batch.shape[1],batch.shape[2])

         
         
#         batch = batch[:, 7:-7, 27:-29]  # [:, 6:-6, 25:-27] # [:,112:-112,140:-140]
          
#         batch = np.reshape(DS.get_train_batch(batch_size),[-1,28,28])
          
#         batch = MD.test.next_batch(batch_size)[0] 
#         batch = np.reshape(batch,[batch_size,28,28])
         
         
#         ts1=tm.time()     
         # run the training step     
         if denoising:
              sess.run([train_step], feed_dict={x_inp: batch, x_noiseless: batch_noisy})
         else:
              sess.run([train_step], feed_dict={x_inp: batch})
#         print(">>> taaaaaayym: " + str(tm.time()-ts1))
         
    
         #print some stuf...
         if step % 200 == 0:

             if denoising:
                  loss_l2_ = l2_loss_.eval(feed_dict={x_inp: test_batch, x_noiseless: test_batch_noisy})
                  loss_kld = KLD.eval(feed_dict={x_inp: test_batch, x_noiseless: test_batch_noisy})
                  loss_tot_ = loss_tot.eval(feed_dict={x_inp: test_batch, x_noiseless: test_batch_noisy})
             else:
                  loss_l2_ = l2_loss_.eval(feed_dict={x_inp: test_batch})
                  loss_kld = KLD.eval(feed_dict={x_inp: test_batch})
                  loss_tot_ = loss_tot.eval(feed_dict={x_inp: test_batch})
             
#             muTmu_v = muTmu.eval(feed_dict={x_inp: test_batch})
#             dets_v = dets.eval(feed_dict={x_inp: test_batch})
#             trc_v = trc.eval(feed_dict={x_inp: test_batch})
             
             if denoising:
                  xh = sess.run([y_out], feed_dict={x_inp: test_batch, x_noiseless: test_batch_noisy} )
             else:
                  xh = sess.run([y_out], feed_dict={x_inp: test_batch} )
             
#             plt.figure();
#                       
#             plt.subplot(2,5,1);plt.imshow(test_batch[0],vmin=0, vmax=1.2)
#             plt.subplot(2,5,2);plt.imshow(test_batch[1],vmin=0, vmax=1.2)
#             plt.subplot(2,5,3);plt.imshow(test_batch[2],vmin=0, vmax=1.2)
#             plt.subplot(2,5,4);plt.imshow(test_batch[3],vmin=0, vmax=1.2)
#             plt.subplot(2,5,5);plt.imshow(test_batch[4],vmin=0, vmax=1.2)
#             plt.subplot(2,5,6);plt.imshow(xh[0].reshape([28,28]),vmin=0, vmax=1.2)
#             plt.subplot(2,5,7);plt.imshow(xh[1].reshape([28,28]),vmin=0, vmax=1.2)
#             plt.subplot(2,5,8);plt.imshow(xh[2].reshape([28,28]),vmin=0, vmax=1.2)
#             plt.subplot(2,5,9);plt.imshow(xh[3].reshape([28,28]),vmin=0, vmax=1.2)
#             plt.subplot(2,5,10);plt.imshow(xh[4].reshape([28,28]),vmin=0, vmax=1.2)
#             
#             plt.show()
             
             if step % 100000 == 0:
#                  pickle.dump(xh, open('/scratch_net/bmicdl02/deletelater/todelete/xh_kldnz2_v9lr_mnist_step'+str(step),'wb' ))
#                  pickle.dump(sh, open('/scratch_net/bmicdl02/deletelater/todelete/sh_kldnz2_v9lr_mnist_step'+str(step),'wb' ))
                  saver.save(sess, '/scratch_net/bmicdl02/modelrecon_2d/models/cvae2d_mri_s14k19_fullconv_nocov_fullim_conz33_homodyn_60ulargedec_252x308_xb5_lat60_varsize_bigDS_aug_new_noiseprec'+str(prec)+'_l2prec'+str(prec)+'_den'+str(denoising)+'_step'+str(beginiter+step) )
             
             test_loss_l2 = np.linalg.norm(xh - test_batch.reshape([batch_size,-1]))
             test_loss_l2_perc = test_loss_l2 / np.linalg.norm(test_batch.reshape([batch_size,-1]))*100

             print("Step {0} | L2 Loss: {1:.3f} | KLD Loss: {2:.3f} | L2 Loss_1: {3:.3f} | L2 Loss_2: {4:.3f} | loss_tot: {5:.3f} | L2 Loss test: {6:.3f} | L2 Loss perc: {7:.3f} | muTmu {8:.3f} | dets {9:.3f} | trc {10:.3f}"\
                   .format(step, np.mean(loss_l2_), np.mean(np.mean(loss_kld)), np.mean(0), np.mean(0), np.mean(loss_tot_), np.mean(test_loss_l2),  np.mean(test_loss_l2_perc), 0.5*np.mean(0), 0.5*np.mean(0), 0.5*np.mean(0) ) )

#             print(">> tayyymmm: " + str(tm.time()-tt))

#     saver.save(sess, '/scratch_net/bmicdl02/deletelater/todelete/cvae2d_v1_lowr_s14k14_mnist_MSJ'     )
    

print("elapsed time: {0}".format(tm.time()-ts))
#
## evaluation : sampling
#enc_muv, enc_cholv, enc_sigmav = sess.run([enc_mu, enc_chol, enc_sigma], feed_dict={x_inp : test_batch})
#plt.figure();
#gen = []
#ctr=1
#for gg in range(50):
#    smptmp = 1*np.matmul(enc_cholv, np.random.randn(5,lat_dim,4,1))
#    smptmp = np.reshape(smptmp, [batch_size, lat_dim, 2,2])
#    smptmp = np.transpose(smptmp, [0,2,3,1])
#    xh2, sh2 = sess.run([y_out, y_out_prec], feed_dict={z_samples:enc_muv + smptmp} )
#    gen.append(xh2[0].reshape(28,28))
#
#plt.figure();
#for ix in range(50):
#    plt.imshow(gen[ix]);plt.title(str(ix));plt.show();plt.pause(0.1)
#    
#    
#
## evaluation : partial sampling - correlated (with Cov matrix)
#enc_muv, enc_cholv, enc_sigmav = sess.run([enc_mu, enc_chol, enc_sigma], feed_dict={x_inp : test_batch})
#plt.figure();
#gen = []
#ctr=1
#for gg in range(50):
#    rnd = np.random.randn(5,lat_dim,4,1)
#    rnd[:,:,1:,0] = 0
#    smptmp = 1*np.matmul(enc_cholv, rnd)
#    smptmp = np.reshape(smptmp, [batch_size, lat_dim, 2,2])
#    smptmp = np.transpose(smptmp, [0,2,3,1])
#    xh2, sh2 = sess.run([y_out, y_out_prec], feed_dict={z_samples:enc_muv + smptmp} )
#    gen.append(xh2[0].reshape(28,28))
#
#plt.figure();
#for ix in range(50):
#    plt.imshow(gen[ix]);plt.title(str(ix));plt.show();plt.pause(0.1)    
# 
#     
#    
#    
#    
## evaluation :  Diagonal (with Cov matrix)
#imix=4
#    
#enc_muv, enc_stdv = sess.run([enc_mu, enc_std], feed_dict={x_inp : test_batch})
#plt.figure();
#gen = []
#ctr=1
#for gg in range(500):
#    rnd = np.random.randn(5,17,18,lat_dim)
#    xh2, sh2 = sess.run([y_out, y_out_prec], feed_dict={z_samples:enc_muv + 1*np.reshape(enc_stdv,[5,17,18,60])*rnd} )
#    gen.append(xh2[imix].reshape(238,252))
#    
#gennp = np.array(gen)
#gennpstd = np.std(gennp,axis=0)
#plt.figure();plt.imshow(gennpstd, vmax=0.02)
#
#plt.figure();
#for ix in range(50):
#    plt.imshow(gen[ix], vmin=0, vmax=1.2);plt.title(str(ix));plt.show();plt.pause(0.1)    
#    
#    
#    
## evaluation :  Not correlated (without any Cov matrix)
#imix=4
#    
#enc_muv, enc_stdv = sess.run([enc_mu, enc_std], feed_dict={x_inp : test_batch})
#plt.figure();
#gen = []
#ctr=1
#for gg in range(500):
#    rnd = np.random.randn(5,17,18,lat_dim)
#    xh2, sh2 = sess.run([y_out, y_out_prec], feed_dict={z_samples:enc_muv + rnd} )
#    gen.append(xh2[imix].reshape(238,252))
#    
#gennp = np.array(gen)
#gennpstd = np.std(gennp,axis=0)
#plt.figure();plt.imshow(gennpstd, vmax=0.3)
#
#plt.figure();
#for ix in range(50):
#    plt.imshow(gen[ix], vmin=0, vmax=1.2);plt.title(str(ix));plt.show();plt.pause(0.1)    
#
##
##
##
##evaluation : random generation
#enc_muv, enc_stdv = sess.run([enc_mu, enc_std], feed_dict={x_inp : test_batch})
#plt.figure();
#ctr=1
##mask = np.zeros([5,2,2,10]);
##mask[:,0,1,:] = 1
#for gg in range(10):
#    xh2, sh2 = sess.run([y_out, y_out_prec], feed_dict={z_samples :  1*np.random.randn(5,17,18,lat_dim)} )
#    for ixx in range(5):
#        plt.subplot(10,5,ctr)
#        plt.imshow(xh2[ixx].reshape(238, 252),vmin=0,vmax=1.2)
#        ctr=ctr+1
#
#
#
#
##evaluation : reconstruction
#xh, sh = sess.run([y_out, y_out_prec], feed_dict={x_inp: test_batch} )
#
#plt.figure();
#for ix in range(5):
#    plt.subplot(3,5,ix+1);plt.imshow(xh[ix].reshape(238,252), vmin=0);
#    plt.subplot(3,5,ix+1+5);plt.imshow(test_batch[ix].reshape(238,252), vmin=0,vmax=1.2);
#    plt.subplot(3,5,ix+1+10);plt.imshow(np.sqrt(1/sh[ix]).reshape(238,252), vmin=0);
#    
#    
##evaluation : reconstruction full image
#from Patcher import Patcher
#patchsize = 28
#Ptchr=Patcher(imsize=[168,210],patchsize=patchsize,step=int(patchsize/2), nopartials=True, contatedges=True)
#test_image = DS.MRi.d_brains_test[0,:,:]
#ptchs = Ptchr.im2patches(test_image)
#
#
#recptchs = np.zeros([165, 28,28])
#for ix in range(int(165/5)):
#     xh, sh = sess.run([y_out, y_out_prec], feed_dict={x_inp: ptchs[5*ix:5*(ix+1)]} )
#     recptchs[5*ix:5*(ix+1),:,:] = xh.reshape([5,28,28])
#
#
#recim = Ptchr.patches2im(recptchs)
#
#
#plt.figure();plt.imshow(np.abs(recim))
#plt.figure();plt.imshow(np.abs(test_image))
#
#













