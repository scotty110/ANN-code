'''
adapting keras model to tf model from:
https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
'''
import numpy as np 
import tensorflow as tf 
from tensorflow.contrib.rnn import BasicLSTMCell, BasicRNNCell, GRUCell, LSTMCell

def to_categorical(y, num_classes=None):
  """Converts a class vector (integers) to binary class matrix.
  E.g. for use with categorical_crossentropy.
  # Arguments
    y: class vector to be converted into a matrix
        (integers from 0 to num_classes).
    num_classes: total number of classes.
  # Returns
     A binary matrix representation of the input.
  """
  y = np.array(y, dtype='int')
  input_shape = y.shape
  if input_shape and input_shape[-1] == 1:
    input_shape = tuple(input_shape[:-1])
  y = y.ravel()
  if not num_classes:
    num_classes = np.max(y) + 1
  n = y.shape[0]
  categorical = np.zeros((n, num_classes))
  categorical[np.arange(n), y] = 1
  output_shape = input_shape + (num_classes,)
  categorical = np.reshape(categorical, output_shape)
  return categorical


# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename, encoding="utf8").read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
#print "Total Characters: ", n_chars
#print "Total Vocab: ", n_vocab

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
  seq_in = raw_text[i:i + seq_length]
  seq_out = raw_text[i + seq_length]
  dataX.append([char_to_int[char] for char in seq_in])
  dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
#print "Total Patterns: ", n_patterns

# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
Y = to_categorical(dataY)


'''
Create TF model
'''
units = 256
seq_length = 100
data_x = tf.placeholder(tf.float32, shape=(None, seq_length, 1))
data_y = tf.placeholder(tf.float32, shape=(None, n_vocab))
batch_size = tf.shape(data_x)[0]
#Create tf cell, api refrence: https://www.tensorflow.org/api_docs/python/tf/contrib/rnn
rnn_cell = BasicLSTMCell(num_units=units, forget_bias=1 )
initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

#Compute RNN
outputs, state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=data_x, initial_state=initial_state, dtype=tf.float32 )

#Got from: https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/
outputs = tf.transpose(outputs, [1, 0, 2])
last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

to_forward = tf.nn.dropout(x=last, keep_prob=0.2)

#Dens: activation(dot(input, kernel) + bias)
w_kernel = tf.Variable(tf.random_uniform(shape=(units,n_vocab)))
b_kernel = tf.Variable(tf.zeros(shape=(1,n_vocab)))
dense = tf.nn.softmax( tf.add(tf.tensordot(to_forward, w_kernel, axes=1),b_kernel) )

# scale preds so that the class probas of each sample sum to 1
dense /= tf.reduce_sum(dense, axis=tf.rank(dense) - 1, keep_dims=True)
# manual computation of crossentropy
_epsilon = tf.convert_to_tensor(10e-8, dtype=tf.float32)
scaled  = tf.clip_by_value(dense, _epsilon, 1. - _epsilon)
loss = - tf.reduce_sum(data_y * tf.log(scaled), axis=tf.rank(scaled)- 1)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
t1 = tf.shape(loss)

with tf.Session() as sess:
  init = tf.initialize_all_variables()
  sess.run(init)
  file_val = ""
  #X,Y = gen_data(file_train,seq_length)

  #Now train and validate model
  batch_size=128

  sess.run( train_step, feed_dict={data_x:X[0:1000], data_y:Y[0:1000]}) 
  print("Model compiled")

