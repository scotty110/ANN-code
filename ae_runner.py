
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np 
import ae_2 as ae2
import ae_3 as ae3


def main_ae2(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  model = ae2.cae(inputs=x, conv1_filter=[5,5,1,32], conv2_filter=[5,5,32,1], alpha=0.002, debug=True)
  #Train the model 
  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
      batch = mnist.train.next_batch(10)
      if i>0 and i % 100 == 0:
        #Evalute model preformance
        #print( mnist.train.num_examples )
        #l2_norm_error = sess.run(model.error, feed_dict={x: batch[0] } )
        #print( "error is: ", l2_norm_error )
        print( "training" )
        #print('step %d, training error %g' % (i, sess.eval(mse)) )
      sess.run(model.optimizer, feed_dict={x: batch[0] })
      

def main(_):
  '''
  Try new model training
  '''
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  # Create the model
  model = ae3.cae(input_shape=[None,784], conv1_filter=[5,5,1,32], conv2_filter=[5,5,32,1], alpha=0.002, debug=True)
  #Train the model 
  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    for i in range(90000):
      batch = mnist.train.next_batch(10)
      if i>0 and i % 1000 == 0:
        #Evalute model preformance
        #print( mnist.train.num_examples )
        mse = 0
        for i in range(0,int(mnist.train.num_examples)):
          mse = mse + model.eval( [mnist.train.images[i]] )
        print( "sumed mse is: ", mse )
        #print("Training still")
      model.train(batch[0])


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


  """
  Links:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
  https://github.com/tensorflow/models/blob/master/autoencoder/autoencoder_models/Autoencoder.py
  https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/layers/cnn_mnist.py
  https://medium.com/towards-data-science/autoencoders-introduction-and-implementation-3f40483b0a85

  #TensorBoard
  https://www.tensorflow.org/get_started/summaries_and_tensorboard
  https://www.tensorflow.org/get_started/tensorboard_histograms
  """