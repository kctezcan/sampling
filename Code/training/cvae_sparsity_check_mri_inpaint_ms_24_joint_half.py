# Simple VAE to see if VAEs can learn sparsity inducing distributions
# Kerem Tezcan, CVL
# initial: 28.05.2017
# last mod: 30.05.2017 

## direk precision optimize etmek daha da iyi olabilir. 

from __future__ import division
from __future__ import print_function
#import os.path

import numpy as np
import time as tm
import tensorflow as tf
import os
from Dataset import Dataset
import sys


SEED=1001
seed=1 
np.random.seed(seed=1)



import argparse

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--mode', type=str, default='Melanie_BFC')

args=parser.parse_args()


# parameters
#==============================================================================
#==============================================================================

user='Kerem' #'Ender'

#mode=sys.argv[2]
mode= args.mode # 'MRIunproc'

#mode='MRIunproc' #'sparse', 'nonsparse', 'MNIST', 'circ', 'Lshape', 'edge', 'Lapedge', 'spiketrain', 'Gaussedge' 

#ndims=int(sys.argv[3])
ndims=28
useMixtureScale=True
noisy=50
batch_size = 50 #1000
usebce=False
kld_div=25.0
nzsamp=1

train_size=5000
test_size=1000  

if useMixtureScale:
     kld_div=1.

std_init=0.05               

input_dim=ndims*ndims
fcl_dim=500 

#lat_dim=int(sys.argv[1])
lat_dim=60
print(">>> lat_dim value: "+str(lat_dim))
print(">>> mode is: " + mode)
     
lat_dim_1 = max(1, np.floor(lat_dim/2))
lat_dim_2 = lat_dim - lat_dim_1
#
#if user=='Kerem':
#     os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
#     from tensorflow.python.client import device_lib
#     print (device_lib.list_local_devices())
#     
#     print( os.environ['SGE_GPU'])


num_inp_channels=1

#make a dataset to use later
#==============================================================================
#==============================================================================
DS = Dataset(train_size, test_size, ndims, noisy, seed, mode, downscale=True)
#from MR_image_data_v3 import MR_image_data_v3
#MRi = MR_image_data_v3(dirname='/scratch_net/bmicdl02/Data/data4fullvol_2d/', imgSize = [260, 311, 260], testchunks = [39], noiseinvstd=50)



#make a simple fully connected network
#==============================================================================
#==============================================================================

tf.reset_default_graph()


sess=tf.InteractiveSession()


#define the activation function to use:
def fact(x):
     #return tf.nn.tanh(x)
     return tf.nn.relu(x)


#define the input place holder
x_inp = tf.placeholder("float", shape=[None, input_dim])
#x_rec = tf.placeholder("float", shape=[None, input_dim])
l2_loss = tf.constant(0.0)

#define the network layer parameters
intl=tf.truncated_normal_initializer(stddev=std_init, seed=SEED)
intl_cov=tf.truncated_normal_initializer(stddev=std_init, seed=SEED)

with tf.variable_scope("VAE") as scope:
     
     
     
    enc_conv1_weights = tf.get_variable("enc_conv1_weights", [3, 3, num_inp_channels, 32], initializer=intl)
    enc_conv1_biases = tf.get_variable("enc_conv1_biases", shape=[32], initializer=tf.constant_initializer(value=0))
     
    enc_conv2_weights = tf.get_variable("enc_conv2_weights", [3, 3, 32, 64], initializer=intl)
    enc_conv2_biases = tf.get_variable("enc_conv2_biases", shape=[64], initializer=tf.constant_initializer(value=0))
     
    enc_conv3_weights = tf.get_variable("enc_conv3_weights", [3, 3, 64, 64], initializer=intl)
    enc_conv3_biases = tf.get_variable("enc_conv3_biases", shape=[64], initializer=tf.constant_initializer(value=0))
         
    mu_weights = tf.get_variable(name="mu_weights", shape=[int(input_dim*64), lat_dim], initializer=intl)
    mu_biases = tf.get_variable("mu_biases", shape=[lat_dim], initializer=tf.constant_initializer(value=0))
    
    logVar_weights = tf.get_variable(name="logVar_weights", shape=[int(input_dim*64), lat_dim], initializer=intl)
    logVar_biases = tf.get_variable("logVar_biases", shape=[lat_dim], initializer=tf.constant_initializer(value=0))
    
    
    if useMixtureScale:
         
         dec_fc1_weights = tf.get_variable(name="dec_fc1_weights", shape=[int(lat_dim), int(input_dim*48)], initializer=intl)
         dec_fc1_biases = tf.get_variable("dec_fc1_biases", shape=[int(input_dim*48)], initializer=tf.constant_initializer(value=0))
         
         dec_conv1_weights = tf.get_variable("dec_conv1_weights", [3, 3, 48, 48], initializer=intl)
         dec_conv1_biases = tf.get_variable("dec_conv1_biases", shape=[48], initializer=tf.constant_initializer(value=0))
          
         dec_conv2_weights = tf.get_variable("decc_conv2_weights", [3, 3, 48, 90], initializer=intl)
         dec_conv2_biases = tf.get_variable("dec_conv2_biases", shape=[90], initializer=tf.constant_initializer(value=0))
          
         dec_conv3_weights = tf.get_variable("dec_conv3_weights", [3, 3, 90, 90], initializer=intl)
         dec_conv3_biases = tf.get_variable("dec_conv3_biases", shape=[90], initializer=tf.constant_initializer(value=0))
         
         dec_out_weights = tf.get_variable("dec_out_weights", [3, 3, 90, 1], initializer=intl)
         dec_out_biases = tf.get_variable("dec_out_biases", shape=[1], initializer=tf.constant_initializer(value=0))
         
         dec1_out_cov_weights = tf.get_variable("dec1_out_cov_weights", [3, 3, 90, 1], initializer=intl)
         dec1_out_cov_biases = tf.get_variable("dec1_out_cov_biases", shape=[1], initializer=tf.constant_initializer(value=0))
         
    else:
         
         pass
    
######## TWO LAYER 
# a. build the encoder layers

x_inp_ = tf.reshape(x_inp, [batch_size,ndims,ndims,1])

enc_conv1 = tf.nn.conv2d(x_inp_, enc_conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
enc_relu1 = fact(tf.nn.bias_add(enc_conv1, enc_conv1_biases))

enc_conv2 = tf.nn.conv2d(enc_relu1, enc_conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
enc_relu2 = fact(tf.nn.bias_add(enc_conv2, enc_conv2_biases))

enc_conv3 = tf.nn.conv2d(enc_relu2, enc_conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
enc_relu3 = fact(tf.nn.bias_add(enc_conv3, enc_conv3_biases))
      
flat_relu3 = tf.contrib.layers.flatten(enc_relu3)

# b. get the values for drawing z
mu = tf.matmul(flat_relu3, mu_weights) + mu_biases
mu = tf.tile(mu, (nzsamp, 1)) # replicate for number of z's you want to draw
logVar = tf.matmul(flat_relu3, logVar_weights) + logVar_biases
logVar = tf.tile(logVar,  (nzsamp, 1))# replicate for number of z's you want to draw
std = tf.exp(0.5 * logVar)

# c. draw an epsilon and get z
epsilon = tf.random_normal(tf.shape(logVar), name='epsilon')
z = mu + tf.multiply(std, epsilon)


if useMixtureScale:

     
     indices1=tf.range(start=0, limit=lat_dim_1, delta=1, dtype='int32')
     indices2=tf.range(start=lat_dim_1, limit=lat_dim, delta=1, dtype='int32')
     
     z1 = tf.transpose(tf.gather(tf.transpose(z),indices1))
     z2 = tf.transpose(tf.gather(tf.transpose(z),indices2))
     
     
     # d. build the decoder layers from z1 for mu(z)
     dec_L1 = fact(tf.matmul(z, dec_fc1_weights) + dec_fc1_biases)     
else:
     pass
    
dec_L1_reshaped = tf.reshape(dec_L1 ,[batch_size,int(ndims),int(ndims),48])

dec_conv1 = tf.nn.conv2d(dec_L1_reshaped, dec_conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
dec_relu1 = fact(tf.nn.bias_add(dec_conv1, dec_conv1_biases))

dec_conv2 = tf.nn.conv2d(dec_relu1, dec_conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
dec_relu2 = fact(tf.nn.bias_add(dec_conv2, dec_conv2_biases))

dec_conv3 = tf.nn.conv2d(dec_relu2, dec_conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
dec_relu3 = fact(tf.nn.bias_add(dec_conv3, dec_conv3_biases))

# e. build the output layer w/out activation function
dec_out = tf.nn.conv2d(dec_relu3, dec_out_weights, strides=[1, 1, 1, 1], padding='SAME')
y_out_ = tf.nn.bias_add(dec_out, dec_out_biases)

y_out = tf.contrib.layers.flatten(y_out_)
                 
# e.2 build the covariance at the output if using mixture of scales
if useMixtureScale:
     
     # e. build the output layer w/out activation function
     dec_out_cov = tf.nn.conv2d(dec_relu3, dec1_out_cov_weights, strides=[1, 1, 1, 1], padding='SAME')
     y_out_prec_log = tf.nn.bias_add(dec_out_cov, dec1_out_cov_biases)
     
     y_out_prec_ = tf.exp(y_out_prec_log)
     
     y_out_prec=tf.contrib.layers.flatten(y_out_prec_)
     
#     #DBG # y_out_cov=tf.ones_like(y_out)




# build the loss functions and the optimizer
#==============================================================================
#==============================================================================

# KLD loss per sample in the batch
KLD = -0.5 * tf.reduce_sum(1 + logVar - tf.pow(mu, 2) - tf.exp(logVar), reduction_indices=1)

x_inp_ = tf.tile(x_inp, (nzsamp, 1))

# L2 loss per sample in the batch
if useMixtureScale:
     l2_loss_1 = tf.reduce_sum(tf.multiply(tf.pow((y_out - x_inp_),2), y_out_prec),axis=1)
     l2_loss_2 = tf.reduce_sum(tf.log(y_out_prec), axis=1) #tf.reduce_sum(tf.log(y_out_cov),axis=1)
     l2_loss_ = l2_loss_1 - l2_loss_2
else:
     l2_loss_ = tf.reduce_sum(tf.pow((y_out - x_inp_),2), axis=1)
     if usebce:
          l2_loss_ = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_out, labels=x_inp_), reduction_indices=1)
     
# take the total mean loss of this batch
loss_tot = tf.reduce_mean(1/kld_div*KLD + 0.5*l2_loss_)

# get the optimizer
if useMixtureScale:
     train_step = tf.train.AdamOptimizer(5e-4).minimize(loss_tot)
else:
     train_step = tf.train.AdamOptimizer(5e-3).minimize(loss_tot)

# start session
#==============================================================================
#==============================================================================


sess.run(tf.global_variables_initializer())


print("Initialized parameters")

saver = tf.train.Saver()




ts=tm.time()


# do the training
#==============================================================================
#==============================================================================
#test_batch = MRi.get_patch(batch_size, test=True)
#test_batch = np.transpose(np.reshape(test_batch, [-1, batch_size]))
test_batch = DS.get_test_batch(batch_size)


with tf.device('/gpu:0'):
     
     #train for N steps
     for step in range(0, 500001): # 500k

#         batch = MRi.get_patch(batch_size, test=False)
#         batch = np.transpose(np.reshape(batch, [-1, batch_size]))
         batch = DS.get_train_batch(batch_size)
         
              
         # run the training step     
         sess.run([train_step], feed_dict={x_inp: batch})
         
    
         #print some stuf...
         if step % 500 == 0: # 500
              
             if useMixtureScale:
                  loss_l2_1 = l2_loss_1.eval(feed_dict={x_inp: test_batch})
                  loss_l2_2 = l2_loss_2.eval(feed_dict={x_inp: test_batch})
             loss_l2_ = l2_loss_.eval(feed_dict={x_inp: test_batch})
             loss_kld = KLD.eval(feed_dict={x_inp: test_batch})
             std_val = std.eval(feed_dict={x_inp: test_batch})
             mu_val = mu.eval(feed_dict={x_inp: test_batch})
             loss_tot_ = loss_tot.eval(feed_dict={x_inp: test_batch})
              
             
             xh = y_out.eval(feed_dict={x_inp: test_batch}) 
             test_loss_l2 = np. mean( np.sum(np.power((xh[0:test_batch.shape[0],:] - test_batch),2), axis=1) )
             
             if useMixtureScale:
                  print("Step {0} | L2 Loss: {1:.3f} | KLD Loss: {2:.3f} | L2 Loss_1: {3:.3f} | L2 Loss_2: {4:.3f} | loss_tot: {5:.3f} | L2 Loss test: {6:.3f}"\
                        .format(step, np.mean(loss_l2_1-loss_l2_2), np.mean(loss_kld), np.mean(loss_l2_1), np.mean(loss_l2_2), np.mean(loss_tot_), np.mean(test_loss_l2)))
             else:
                  print("Step {0} | L2 Loss: {1:.3f} | KLD Loss: {2:.3f} | L2 Loss test: {3:.3f} | std: {4:.3f} | mu: {5:.3f}"\
                        .format(step, np.mean(loss_l2_), np.mean(loss_kld),  np.mean(test_loss_l2), np.mean(std_val), np.mean(mu_val)))


         if step%100000==0:
               saver.save(sess, '/home/ktezcan/modelrecon/trained_models/cvae_MSJhalf_rscl_'+str(mode)+'_fcl'+str(fcl_dim)+'_lat'+str(lat_dim)+'_ns'+str(noisy)+'_ps'+str(ndims)+'_step'+str(step))

     
print("elapsed time: {0}".format(tm.time()-ts))

#
## do post-training predictions
##==============================================================================
##==============================================================================
#
#
#
#test_batch = DS.get_test_batch(batch_size)
#
#saver.restore(sess, '/home/ktezcan/modelrecon/trained_models/cvae_MSJ_'+mode+'_fcl'+str(fcl_dim)+'_lat'+str(lat_dim)+'_ns'+str(noisy)+'_ps'+str(ndims))
#
#     
##xh = y_out.eval(feed_dict={x_inp: test_batch}) 
##xch = 1./y_out_prec.eval(feed_dict={x_inp: test_batch}) 
#
#xh, xch = sess.run([y_out, 1/y_out_prec], feed_dict={x_inp: test_batch})
#
#nsamp=50
#
#yos=np.zeros((nsamp,input_dim))
#sos=np.zeros((nsamp,input_dim))
#yos_samp = np.zeros((nsamp,input_dim))
#for ix in range(nsamp):
#     print(ix)
#     zr = np.random.randn(1,lat_dim)
#     yo=y_out.eval(feed_dict={z: np.tile(zr,[50,1])})
#     try:
#          so=1./y_out_prec.eval(feed_dict={z: np.tile(zr,[50,1])})
#     except:
#          so = 1/kld_div          
#     yos[ix,:]=yo[0]
#     sos[ix,:]=np.sqrt(so[0])
#     yos_samp[ix,:] = yos[ix,:] + np.random.randn(input_dim)*sos[ix,:]
#
#
#print("generated means: ")
#print("=========================")
#plt.figure(figsize=(10,10))
#for ix in range(16):
#    plt.subplot(4,4, ix+1);
#    plt.imshow(np.reshape(yos[ix,:],(28,28)),cmap='gray');plt.xticks([]);plt.yticks([])
#
#
#print("generated covs: ")
#print("=========================")
#plt.figure(figsize=(10,10))
#for ix in range(16):
#    plt.subplot(4,4, ix+1)
#    plt.imshow(np.reshape(sos[ix,:],(28,28)),cmap='gray')
#
#
#print("means + [-1,+1]*covs: ")
#print("=========================")
#show_samp=20
#mults = np.linspace(-40,40,7)
#fig, ax = plt.subplots(show_samp,9, figsize=(20,show_samp*2))
#for ix in range(show_samp):
#    for ixc in range(7):
#         ax[ix][ixc].imshow(np.reshape(xh[ix,:],(28,28))+mults[ixc]*np.reshape(xch[ix,:],(28,28)),cmap='gray',vmin=-0.2,vmax=1.2)
#         ax[ix][7].imshow(np.reshape(test_batch[ix,:],(28,28)), cmap='gray',vmin=-0.2,vmax=1.2)
#         ax[ix][8].imshow(np.reshape(xch[ix,:],(28,28)), cmap='gray')
#
#
#
#
#
#
##########
##
###person=DS.get_test_batch(1)  
###person=denoise[80:108,80:108]
##
##
##from scipy import ndimage
##
##x_rec2=tf.get_variable('x_rec2',shape=[50,input_dim],initializer=tf.constant_initializer(value=0.0))
##
##person=DS.MRi.d_brains_test[10,180:208,180:208]
###person=denoise[180:208,180:208]
##
##nsampl=50
##imsize=28
##minperc=2
##maxperc=98
##nsigma=15
##   
##op_p_x_z = - 0.5 * tf.reduce_sum(tf.multiply(tf.pow((y_out - x_rec2),2), y_out_prec),axis=1) \
##             + 0.5 * tf.reduce_sum(tf.log(y_out_prec), axis=1) #-  0.5*48*48*np.log(2*np.pi)
##       
##op_p_x_z_0 = - 0.5 * tf.reduce_sum(tf.pow((y_out - x_rec2),2),axis=1)      
##op_p_x_z_1 = - 0.5 * tf.reduce_sum(tf.multiply(tf.pow((y_out - x_rec2),2), y_out_prec),axis=1)
##op_p_x_z_2 = + 0.5 * tf.reduce_sum(tf.log(y_out_prec), axis=1)
##
##op_q_z_x = (- 0.5 * tf.reduce_sum(tf.multiply(tf.pow((z - mu),2), tf.reciprocal(std)),axis=1) \
##                  - 0.5 * tf.reduce_sum(tf.log(std), axis=1) -  0.5*lat_dim*tf.log(2*np.pi))
##
##op_p_z = (- 0.5 * tf.reduce_sum(tf.multiply(tf.pow((z - tf.zeros_like(mu)),2), tf.reciprocal(tf.ones_like(std))),axis=1) \
##                  - 0.5 * tf.reduce_sum(tf.log(tf.ones_like(std)), axis=1) -  0.5*lat_dim*tf.log(2*np.pi))
##
##
##np.set_printoptions(threshold=1000)
##
##sigmas=np.linspace(0.001,30,nsigma)
##
##tbbs=np.zeros((nsigma,imsize,imsize))
##
##goodbad=np.zeros(25)
##persons=np.zeros((25,imsize,imsize))
##
##plt.figure(figsize=(20,10))
##
##for ixm in range(10,11):
##     person=DS.get_test_batch(1) 
##     
##     aas=np.zeros((nsigma,1))#,dtype='float64')
##     Ks=np.zeros((nsigma,1))#,dtype='float64')
##     aams = np.zeros((nsigma,1))#,dtype='float64')
##     bbs=np.zeros((nsampl,nsigma))#,dtype='float64')
##
##     for ix in range(nsigma):
##         print(ix)
##         tb=np.reshape(person,(imsize,imsize)) 
##         tbb=ndimage.gaussian_filter(tb,sigma=sigmas[ix])
##         #tbb=tb + np.random.normal(loc=0, scale=sigmas[ix], size=tb.shape)
##         tbb=(tbb - np.percentile(tbb, minperc))/(np.percentile(tbb, maxperc) - np.percentile(tbb, minperc))
##         tbbs[ix,:,:]=tbb.copy()
##         tbb=np.reshape(tbb,(1,imsize*imsize))
##         x_rec2.load(    value = np.tile(tbb,(nsampl,1))    )
##         #get p(x|z_n), q(z_n|x) and p(z_n)
##         p_x_z, q_z_x, p_z  = sess.run([op_p_x_z, op_q_z_x, op_p_z], feed_dict={x_inp: np.tile(tbb,(nsampl,1))})
##     
##         aa = (p_x_z - q_z_x + p_z).astype('float128') #).astype('float128') 
##         
##         K=np.max(aa)
##         
##         aad=(aa-K).astype('float128')
##         
##         aae=np.exp(aad)
##         
##         aaes=np.sum(aae)
##         if ix==31:
##              print("aa")
##              print(aa)
##              print("aae")
##              print(aae)
##              print("aaes")
##              print(aaes)
##         
##         
##         aaesl=np.log(aaes)
##         
##         aaeslKS=aaesl + K - np.log(nsampl)
##     
##         aas[ix,0]=aaeslKS
##            
##           
##     if aas[0]>aas[-1]:
##          goodbad[ixm]=1
##     else:
##          goodbad[ixm]=0
##     persons[ixm,:,:]=np.reshape(person,(imsize,imsize))
##     
##     
##     plt.plot(sigmas, aas,'.-')
##
##
##
##
##
##
###from tensorflow.examples.tutorials.mnist import input_data
###from scipy import ndimage
###mnist2 = input_data.read_data_sets('MNIST')
###
###aas=np.zeros((50,1))
###sigmas=np.linspace(0.001,1,50)
###for ix in range(50):
###    tb=np.reshape(mnist2.test.images[0,:],(28,28)) 
###    tbb=ndimage.gaussian_filter(tb,sigma=sigmas[ix])
###    tbb=np.reshape(tbb,(1,784))
###    x_rec.load(    value = np.tile(tbb,(5000,1))    )
###    aa = (- 1/2 * tf.reduce_sum(tf.multiply(tf.pow((y_out - x_rec),2), y_out_prec),axis=1) \
###             + 1/2 * tf.reduce_sum(tf.log(y_out_prec), axis=1) -  1/2*784*tf.log(2*np.pi)).eval(feed_dict={x_inp: np.tile(tbb,(5000,1))})
###    aas[ix,0]=aa.mean()
###
###plt.plot(sigmas, aas, '.-'), plt.title("log(p(x)) vs. sigma of blurrung gaussian"), plt.xlabel("gaussian std"),plt.ylabel("log likelihood")
##     
##     
##     
###
#### do post-training predictions
####==============================================================================
####==============================================================================
###
###test_batch = DS.get_test_batch(batch_size)
###
###if user == 'Kerem':
###     saver.restore(sess, '/home/ktezcan/Code/spyder_files/tests_code_results/models_vae_mri_ms_iter20k/vae_MG_fcl'+str(fcl_dim)+'_lat'+str(lat_dim)+'_ns'+str(noisy)+'_klddiv'+str(int(kld_div)))
###elif user == 'Ender':     
###     saver.restore(sess,'/scratch/kender/Projects/VAE/spyder_files')
###else:
###     print("User unknown!")
###     assert(1==0)
###     
###xh = y_out.eval(feed_dict={x_inp: test_batch}) 
###xch = 1./y_out_prec.eval(feed_dict={x_inp: test_batch}) 
###
###
###nsamp=5000
###
###yos=np.zeros((nsamp,input_dim))
###sos=np.zeros((nsamp,input_dim))
###yos_samp = np.zeros((nsamp,input_dim))
###for ix in range(nsamp):
###     zr = np.random.randn(1,lat_dim)
###     yo=y_out.eval(feed_dict={z: zr})
###     try:
###          so=1./y_out_prec.eval(feed_dict={z: zr})
###     except:
###          so = 1/kld_div          
###     yos[ix,:]=yo
###     sos[ix,:]=np.sqrt(so)
###     yos_samp[ix,:] = yos[ix,:] + np.random.randn(input_dim)*sos[ix,:]
###
###
###print("generated means: ")
###print("=========================")
###plt.figure(figsize=(10,10))
###for ix in range(16):
###    plt.subplot(4,4, ix+1)
###    plt.imshow(np.reshape(yos[ix,:],(28,28)),cmap='gray')
###
###
###print("generated covs: ")
###print("=========================")
###plt.figure(figsize=(10,10))
###for ix in range(16):
###    plt.subplot(4,4, ix+1)
###    plt.imshow(np.reshape(sos[ix,:],(28,28)),cmap='gray')
###
###
###print("means + [-1,+1]*covs: ")
###print("=========================")
###show_samp=20
###mults = np.linspace(-1,1,7)
###fig, ax = plt.subplots(show_samp,9, figsize=(20,show_samp*2))
###for ix in range(show_samp):
###    for ixc in range(7):
###         ax[ix][ixc].imshow(np.reshape(xh[ix,:],(28,28))+mults[ixc]*np.reshape(xch[ix,:],(28,28)),cmap='gray')
###         ax[ix][7].imshow(np.reshape(test_batch[ix,:],(28,28)), cmap='gray')
###         ax[ix][8].imshow(np.reshape(xch[ix,:],(28,28)), cmap='gray',vmin=-1, vmax=1)
###
###
###
###from tensorflow.examples.tutorials.mnist import input_data
###from scipy import ndimage
###mnist2 = input_data.read_data_sets('MNIST')
###
###aas=np.zeros((50,1))
###sigmas=np.linspace(0.001,1,50)
###for ix in range(50):
###    tb=np.reshape(mnist2.test.images[0,:],(28,28)) 
###    tbb=ndimage.gaussian_filter(tb,sigma=sigmas[ix])
###    tbb=np.reshape(tbb,(1,784))
###    x_rec.load(    value = np.tile(tbb,(5000,1))    )
###    aa = (- 1/2 * tf.reduce_sum(tf.multiply(tf.pow((y_out - x_rec),2), y_out_prec),axis=1) \
###             + 1/2 * tf.reduce_sum(tf.log(y_out_prec), axis=1) -  1/2*784*tf.log(2*np.pi)).eval(feed_dict={x_inp: np.tile(tbb,(5000,1))})
###    aas[ix,0]=aa.mean()
###
###plt.plot(sigmas, aas, '.-'), plt.title("log(p(x)) vs. sigma of blurrung gaussian"), plt.xlabel("gaussian std"),plt.ylabel("log likelihood")
##     
##     
##     