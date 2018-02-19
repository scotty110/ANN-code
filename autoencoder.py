import tensorflow as tf
import numpy as np
import functools

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
  
class cae:
    '''
    TensorFlow Model for a 2D convolutional autoencoder. 
        Model parameters:
            self.videoData - placeholder for Video we want to compress
            self.conv1_numNodes1 - number of nodes in 1st convolutional layer
            self.conv1_kernel - kernel size for 1st convolutional layer
            self.conv1_activation - activation function for 1st convolutional layer
            self.conv2_numNodes2 - number of nodes in the 2nd convolutional layer
            self.conv2_activation - activation function for 2nd convolutional layer
            self.deconv1_activation - activation function for 1st deconvolutional layer
            self.deconv2_activation - activation functino for 2nd deconvolutional layer
            self.learning_rate - learning rate for our optimizer
            self.global_step - keeps track of global step
        Model Parts:
            self.encoder - output from our 2D convolutional autoencoder
            self.optimizer - Gradient Descent Optimizer with a MSE loss function
            self.error - sum of l2 norm of the difference between the predicted and actual frame
            
    '''
     
    def __init__(self, videoData, numNodes1, kernel1, activ1, \
                    numNodes2, activ2, dactiv1, dactiv2, alpha, debug=False):
        '''
        Defining Tensorflow model properties
        Inputs:
            videoData - Video Data placeholder which we will train over, shape [batch_size, height, width, channels]
            numNodes1 - 1st convolutional layer nodes
            kernel1 - 1st convolutional layer kernel size
            activ1 - 1st convolutional layer activiation function
            numNodes2 - 2nd convolutional layer nodes
            activ2 - 2nd convolutional layer activation function
            dactiv1 - 1st deconvoltional activation fucntion
            dactiv2 - 2nd deconvolitional activation function
            alpha - learning rate for optimizer
        '''
        self.videoData = videoData
        self.conv1_numNodes = numNodes1
        self.conv1_kernel = kernel1
        self.conv1_activation =  activ1
        self.conv2_numNodes = numNodes2
        self.conv2_activation =  activ2
        self.deconv1_activation =  dactiv1
        self.deconv2_activation =  dactiv2
        self.learning_rate = alpha
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.debug = debug

        self.encoder
        self.error
        self.optimizer

    @lazy_property
    def encoder(self):
        ''''
        Our 2D convolutional AutoEncoder 
        '''
        input_layer = self.videoData       #tf.placeholder(tf.float32, shape = self.videoData.get_shape() )
        
        # Convolutional API:  https://www.tensorflow.org/api_docs/python/tf/layers/conv2d 
        # input shape is [batch, in_height, in_width, in_channels]
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs = input_layer,
            filters = self.conv1_numNodes,
            kernel_size = self.conv1_kernel,
            padding = 'valid',
            activation = self.conv1_activation )
        
        if( self.debug ):
            print( 'Conv1 shape ', conv1.get_shape().as_list())

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)      #going to leave as default, strides tell overlap (?)

        #Kernel size for 2nd convolution is determined by 1st convolution
        poolShape = pool1.get_shape().as_list()     #gets tensorshape
        conv2_kernel = [ poolShape[1], poolShape[2] ]

        if( self.debug ):
            print( 'Pool shape ', poolShape)

        # Convolutional Layer #2
        conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters = self.conv2_numNodes,
        kernel_size = conv2_kernel,
        padding = 'valid',
        activation = self.conv1_activation )

        # OutPut shape of second convolutional layer 
        conv2Shape = conv2.get_shape().as_list()     #gets tensorshape
        kernelD1 = [ 1, conv2Shape[2] ]
        
        if( self.debug ):
            print('conv2 output shape ',conv2Shape )

        # Deconvolution API:  https://www.tensorflow.org/api_docs/python/tf/layers/conv2d_transpose
        # Unpooling Ideas:  https://github.com/tensorflow/tensorflow/issues/2169 
        # Deconvolution 1
        deconv1 = tf.layers.conv2d_transpose(
            inputs = conv2, 
            filters = conv2Shape[3], 
            kernel_size = kernelD1, 
            strides = (2, 2), 
            activation = self.deconv1_activation )
        

        # Depooling, previous deconvolution should have taken care of

        # Calc new kernel size such that we get original frame shape back
        deconv1Shape = deconv1.get_shape().as_list()
        nb_row = self.videoData.get_shape().as_list()[1] - deconv1Shape[1]+1
        nb_column = self.videoData.get_shape().as_list()[2] - deconv1Shape[2]+1
        kernelD2 = [ nb_row, nb_column ]

        if( self.debug ):
            print( 'deconve1 output shape ', deconv1Shape )
            print('kernel deconv 2', kernelD2 )
            
        # Deconvolution 2
        deconv2 = tf.layers.conv2d_transpose(
            inputs = deconv1, 
            filters = self.videoData.get_shape()[3], 
            kernel_size = kernelD2, 
            strides = (1, 1), 
            activation = self.deconv2_activation)
        
        if(self.debug):
            print('deconv 2 output shape ', deconv2.get_shape().as_list() )

        encoder = deconv2
        return encoder

    @lazy_property
    def optimizer(self):
        '''
        The optimizer to use for our autoencoder, using MSE as loss function
        '''
        # predictions - predicted output of model
        # labels - ground truth output tensor, needs to be same dimension as predictions
        loss = tf.losses.mean_squared_error( predictions=self.encoder, labels=self.videoData)
        #optimize = tf.train.GradientDescentOptimizer( self.learning_rate )
        optimize = tf.train.AdamOptimizer( self.learning_rate )
        optimizer = optimize.minimize(loss, global_step=self.global_step)
        return optimizer
    
    @lazy_property
    def error(self):
        '''
        Calculates the l2 error of the encoder during training.
        '''
        # Function API:  https://www.tensorflow.org/api_docs/python/tf/global_norm
        # Want to calc l2 norm of the difference, to see how closely approximating
        #difference = [self.encoder - self.videoData]
        #error = tf.global_norm( difference )
        error = tf.losses.mean_squared_error( predictions=self.encoder, labels=self.frame )
        return error