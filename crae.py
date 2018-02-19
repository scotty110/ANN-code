import tensorflow as tf
from tensorflow.contrib.rnn import *
import numpy as np
import functools
#import image_handler as image_handler
import image_split_loader as isl
import os

"""
THIS MODEL WORKS
"""

def lazy_property(function):
    '''
    Danijar Hafner:  
        https://danijar.com/
        https://gist.github.com/danijar
    '''
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class videoPredictor:
  '''
  TensorFlow Model for a 2D convolutional autoencoder. 
  '''
    
  def __init__(self, time_steps=9, patch_size=8, alpha=0.002, debug=False):
    '''
    Defining Tensorflow model properties
    Inputs:
        TODO
        alpha - learning rate for optimizer
    '''
    #Feed parameters
    self.time_steps = time_steps
    self.patch_size = patch_size
    self.input_rnn = tf.placeholder(tf.float32, shape=[None, self.time_steps, self.patch_size, self.patch_size], name='rnn_input')
    self.true_image = tf.placeholder(tf.float32, shape=[None, self.patch_size, self.patch_size], name='uncompressed')
    
    self.alpha = alpha
    self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    self.debug = debug

    #Model pieces
    self.rnnNetwork
    self.error
    self.optimizer

    #for Running
    init = tf.global_variables_initializer()
    self.saver =  tf.train.Saver()
    config = tf.ConfigProto(device_count = {'GPU': 0})
    #self.supervisor = tf.train.Supervisor(logdir="/tmp/model_logs", saver=self.saver, save_model_secs=600)
    self.sess = tf.Session(config=config)
    #self.sess = (self.supervisor).managed_session()
    self.sess.run(init)

  @lazy_property
  def rnnNetwork(self):
    ''''
    Our rnn portion of code
    '''
    with tf.name_scope('input_processing'):
      #Orig input shape: [batch_size, time_step, pixel_values]
      input_layer = self.input_rnn

      #Reshape so patch become a vector
      input_layer = tf.reshape( input_layer, shape=[-1,self.time_steps*self.patch_size**2,1] )

      #input_layer = tf.transpose(input_layer, perm=[1,0,2] )
      input_layer_shape = tf.shape(input_layer)
      num_steps = tf.shape(input_layer)[1]
      batch_size = tf.shape(input_layer)[0]

      if(self.debug):
        print "num steps: ", num_steps
        print "batch size: ", batch_size
        print "input layer shape: ", input_layer.get_shape().as_list()

    with tf.name_scope('rnn_cell'):
      #cell = BasicLSTMCell( self.patch_size**2, forget_bias=0.0, state_is_tuple=True, reuse=False)
      cell = BasicRNNCell( 1024, reuse=False)   #will need to chang in the future
      state = cell.zero_state(batch_size,dtype=tf.float32)
      rnn_output, state = tf.nn.dynamic_rnn(cell, input_layer, initial_state=state, time_major=False, dtype=tf.float32) 
      if self.debug:
        print "rnn output shape: ", rnn_output.get_shape().as_list()
        print "rnn output[0] shape: ", rnn_output[0].get_shape().as_list()
        print "rnn state shape: ", state
      
    with tf.name_scope("Reshape_final"):
      output = tf.reshape(rnn_output[0], [batch_size, self.patch_size, self.patch_size] )
      if(self.debug):
        print "output shape: ", output.get_shape().as_list()

    return output


  @lazy_property
  def optimizer(self):
    '''
    The optimizer to use for our autoencoder, using MSE as loss function
    '''
    # predictions - predicted output of model
    # labels - ground truth output tensor, needs to be same dimension as predictions
    loss = tf.losses.mean_squared_error( predictions=self.rnnNetwork, labels=self.true_image )
    optimize = tf.train.AdamOptimizer( self.alpha )
    optimizer = optimize.minimize(loss, global_step=self.global_step)
    return optimizer

  @lazy_property
  def error(self):
    '''
    Calculates the l2 error of the encoder during training.
    '''
    # Function API:  https://www.tensorflow.org/api_docs/python/tf/global_norm
    error = tf.losses.mean_squared_error( predictions=self.rnnNetwork, labels=self.true_image )
    return error

  def train(self, image_compressed, image_raw, counter=0, batch_size=1, loop=1):
    '''
    Trains model on X data
    '''
    #Create training
    X,Y,count = self.createTrainingData(image_compressed, image_raw, batch_size=batch_size, counter=counter)
    for j in range(0,X.shape[0]):
      for i in range(0,loop):
        self.sess.run( self.optimizer, feed_dict={self.input_rnn:X[j][:], self.true_image:Y[j][:]} )
    del X
    del Y
    #print "Done Training"
    return count
  
  def evaluate(self, image_compressed, image_raw, batch_size=1, counter=0):
    '''
    Calcs MSE for model on X data
    '''
    mse = 0
    X,Y,count = self.createTrainingData(image_compressed, image_raw, batch_size=batch_size, counter=counter)
    #print "made it this far"
    for j in range(0,X.shape[0]):
        mse = mse + self.sess.run( self.error, feed_dict={self.input_rnn:X[j][:], self.true_image:Y[j][:]} )
    del X
    del Y
    return mse

  def createTrainingData(self, image_compressed, image_raw, batch_size=1, counter=0):
    #create training data
    X,Y,count = isl.nextbatch(batch_size=batch_size, comp_file_array=image_compressed, raw_file_array=image_raw, starting_point=counter)
    return X,Y,count

  def save(self, checkpoint_directory):
    '''
    Saves tensorflow session
    Inputs:
      checkpoint_directory - directory and file where to save model information too  (no file extensions)
    '''
    #saver = tf.train.Saver()
    self.saver.save(self.sess, checkpoint_directory, global_step=self.global_step )
    return True

  def load(self, checkpoint_directory):
    '''
    Loads checkpoint file for tensorflow model.  
    Inputs:
      checkpoint_directory - directory and file where to load model information from  (no file extensions)
    '''
    #saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_directory))
    if ckpt and ckpt.model_checkpoint_path:
      self.saver.restore(self.sess, ckpt.model_checkpoint_path)
      return True
    #saver.restore(self.sess, checkpoint_directory )
    return False

  def runner(self, image_compressed_dir, image_raw_dir, model_loc="./test_chp", loop=1, batch_size=2, epochs=10):
    '''
    Runs model, inclueds checkpointing features
    '''
    loop = 1
    comp_files = isl.processDir(dirname=image_compressed_dir)
    raw_files = isl.processDir(dirname=image_raw_dir)
    #Need to add loop for taining over whole data set
    
    for count_i in range(epochs+1):
      #print "Batch size: ", batch_size
      #print "counter: ", count_i
      #print "len of files: ", len(comp_files), len(raw_files)
      for j in range(0, len(comp_files), batch_size):
        self.train(image_compressed=comp_files, image_raw=raw_files, counter=j, batch_size=batch_size, loop=loop)
      
      if count_i%5==0:
        for j in range(0, len(comp_files), batch_size):
          mse = self.evaluate(image_compressed=comp_files, image_raw=raw_files, batch_size=batch_size, counter=j)
        print( "summed MSE is: ", (mse) )
        self.save(checkpoint_directory=model_loc) 
        #model.load(checkpoint_directory="/home/scott/Documents/Code/checkpoint_test/tester") 

    return 1
 

'''
Other Help:
https://www.tensorflow.org/tutorials/layers
https://www.tensorflow.org/api_docs/python/tf/cond
https://stackoverflow.com/questions/34959853/how-to-make-an-if-statement-using-a-boolean-tensor
https://www.tensorflow.org/versions/r0.12/how_tos/supervisor/
'''
