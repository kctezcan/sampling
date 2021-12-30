import sys
import os
scriptdir = os.path.dirname(sys.argv[0])+'/'
print("KCT-info: running script from directory: " + scriptdir)
os.chdir(scriptdir)



def definevae2(lat_dim=60, patchsize=28,batchsize=50, rescaled=False, half=False, mode=[], chunks40=False, Melmodels=False):

     import tensorflow as tf
     
#     config=tf.ConfigProto()
#     config.gpu_options.allow_growth=True
#     config.allow_soft_placement=True 
     
     import numpy as np
     import os
         
     user='Kerem' #'Ender'
     
     if mode ==[]:
          mode='MRIunproc'
     
#     print("---- came here a1")
     
     SEED=10
     
     ndims=patchsize
     useMixtureScale=True
     noisy=50
     batch_size = batchsize #50#361*2 # 1428#1071#714#714#714 #1000 1785#      
     usebce=False
     nzsamp=1
     
     kld_div=1.
     
     std_init=0.05               
     
     input_dim=ndims*ndims
     fcl_dim=500 
     
     lat_dim_1 = max(1, np.floor(lat_dim/2))
     
     
     
     num_inp_channels=1
     
     #make a simple fully connected network
     #==============================================================================
     #==============================================================================
     
     tf.reset_default_graph()
     
     #define the activation function to use:
     def fact(x):
          #return tf.nn.tanh(x)
          return tf.nn.relu(x)
     
     
     #define the input place holder
     x_inp = tf.placeholder("float", shape=[None, input_dim])
     nsampl=50
     
     
     #define the network layer parameters
     intl=tf.truncated_normal_initializer(stddev=std_init, seed=SEED)
     
     
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
     #x_inp_ = tf.reshape(x_rec, [batch_size,ndims,ndims,1])
     
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
     
     x_inp__ = tf.tile(x_inp, (nzsamp, 1))
     
     # L2 loss per sample in the batch
     if useMixtureScale:
          l2_loss_1 = tf.reduce_sum(tf.multiply(tf.pow((y_out - x_inp__),2), y_out_prec),axis=1)
          l2_loss_2 = tf.reduce_sum(tf.log(y_out_prec), axis=1) #tf.reduce_sum(tf.log(y_out_cov),axis=1)
          l2_loss_ = l2_loss_1 - l2_loss_2
     else:
          l2_loss_ = tf.reduce_sum(tf.pow((y_out - x_inp__),2), axis=1)
          if usebce:
               l2_loss_ = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_out, labels=x_inp__), reduction_indices=1)
          
     # take the total mean loss of this batch
     loss_tot = tf.reduce_mean(1/kld_div*  KLD + l2_loss_)
     
     # get the optimizer
     if useMixtureScale:
          train_step = tf.train.AdamOptimizer(5e-4).minimize(loss_tot)
     else:
          train_step = tf.train.AdamOptimizer(5e-3).minimize(loss_tot)
          
     # start session
     #=====================================definevae2=========================================
     #==============================================================================
     
     sess=tf.InteractiveSession()#(config=config)
     
     sess.run(tf.global_variables_initializer())
     print("Initialized parameters")
     
     saver = tf.train.Saver()
     
     # do post-training predictions
     #==============================================================================
     #==============================================================================
     
     if Melmodels=='':
          if chunks40:
               step=500000
#               print("KCT-info:  current directory is: " + os.getcwd())
               saver.restore(sess, os.getcwd()+'/../../trained_models/cvae_MSJhalf_40chunks_fcl'+str(fcl_dim)+'_lat'+str(lat_dim)+'_ns'+str(noisy)+'_ps'+str(patchsize)+'_step'+str(step))
               print("KCT-info: loaded the new model, trained patchwise on the 40 chunk dataset")
          else:
               raise ValueError
     else:
          raise ValueError
         
     
     #gradient stuff, gd recon etc...
     #==============================================================================
     nsampl=batchsize#361*2 # 1428#1071#714#714 1785#
     x_rec=tf.get_variable('x_rec',shape=[nsampl,ndims*ndims],initializer=tf.constant_initializer(value=0.0))
     
     z_std_multip = tf.placeholder_with_default(1.0, shape=[])
     
     
     #REWIRE THE GRAPH
     #you need to rerun all operations after this as well!!!!
     
     #qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
     #%qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
     
     # rewire the graph input
     x_inp_ = tf.reshape(x_rec, [nsampl,ndims,ndims,1])
     
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
     z = mu + z_std_multip*tf.multiply(std, epsilon) # z_std_multip*epsilon     #   # KCT!!!  
     
     
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
     
        
     # qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
     # qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
     
     
     op_p_x_z = (- 0.5 * tf.reduce_sum(tf.multiply(tf.pow((y_out - x_rec),2), y_out_prec),axis=1) \
                  + 0.5 * tf.reduce_sum(tf.log(y_out_prec), axis=1) -  0.5*ndims*ndims*tf.log(2*np.pi) ) 
     
     #op_p_x_z = (- 0.5 * tf.reduce_sum(tf.multiply(tf.pow((y_out - x_rec),2), 1),axis=1) \
     #             + 0.5 * tf.reduce_sum(tf.log(tf.ones_like(y_out_prec)), axis=1) -  0.5*784*tf.log(2*np.pi) ) 
     
     op_q_z_x = (- 0.5 * tf.reduce_sum(tf.multiply(tf.pow((z - mu),2), tf.reciprocal(std)),axis=1) \
                       - 0.5 * tf.reduce_sum(tf.log(std), axis=1) -  0.5*lat_dim*tf.log(2*np.pi))
     
     z_pl = tf.get_variable('z_pl',shape=[nsampl,lat_dim],initializer=tf.constant_initializer(value=0.0))
     
     op_q_zpl_x = (- 0.5 * tf.reduce_sum(tf.multiply(tf.pow((z_pl - mu),2), tf.reciprocal(std)),axis=1) \
                       - 0.5 * tf.reduce_sum(tf.log(std), axis=1) -  0.5*lat_dim*tf.log(2*np.pi))
     
     op_p_z = (- 0.5 * tf.reduce_sum(tf.multiply(tf.pow((z - tf.zeros_like(mu)),2), tf.reciprocal(tf.ones_like(std))),axis=1) \
                       - 0.5 * tf.reduce_sum(tf.log(tf.ones_like(std)), axis=1) -  0.5*lat_dim*tf.log(2*np.pi))
     
     
     funop=op_p_x_z + op_p_z - op_q_z_x
     
     grd = tf.gradients(op_p_x_z + op_p_z - op_q_z_x, x_rec) # 
     grd_p_x_z0 = tf.gradients(op_p_x_z, x_rec)[0]
     grd_p_z0 = tf.gradients(op_p_z, x_rec)[0]
     grd_q_z_x0 = tf.gradients(op_q_z_x, x_rec)[0]
     
     grd_q_zpl_x_az0 = tf.gradients(op_q_zpl_x, z_pl)[0]
     
     grd2 = tf.gradients(grd[0], x_rec)
     
     print("KCT-INFO: the gradients: ")
     print(grd_p_x_z0)
     print(grd_p_z0)
     print(grd_q_z_x0)
     
     grd0=grd[0]
     grd20=grd2[0]
                                                          
          
     return x_rec, x_inp, funop, grd0, sess, grd_p_x_z0, grd_p_z0, grd_q_z_x0, grd20, y_out, y_out_prec, z_std_multip, op_q_z_x, mu, std, grd_q_zpl_x_az0, op_q_zpl_x, z_pl, z                               





