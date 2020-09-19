import tensorflow as tf

import numpy as np

import utils
import word_utils

# import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

' Code for learning tensorflow '

__author__ = 'Wenwu Yang'

DATA_FILE = 'data/birth_life_2010.txt'

def read_birth_life_data(filename):
	with open(filename, 'r') as fp:
		text = fp.readlines()[1:]

	data = [ line[:-1].split('\t')  for line in text ]	
	data = [ (float(line[1]), float(line[2])) for line in data ]
	n_sample = len(data)
	data = np.asarray(data, dtype = np.float32)

	return data, n_sample


# Step 1: read in the data
data, n_sample = read_birth_life_data(DATA_FILE)

# Use the huber loss
def huber_loss(labels, predictions, delta=14.0):
    residual = tf.abs(labels - predictions)
    def f1(): return 0.5 * tf.square(residual)
    def f2(): return delta * residual - 0.5 * tf.square(delta)
    return tf.cond(residual < delta, f1, f2)

def use_placeholder_for_data():
	# Step 2: assemble the graph for linear regression neural net
	# Y = wX + b
	X = tf.placeholder(tf.float32, name = 'X')
	Y = tf.placeholder(tf.float32, name = 'Y')

	w = tf.get_variable('weights', initializer=tf.constant(0.0))
	b = tf.get_variable('bias', initializer=tf.constant(.0))

	Y_predicted = w*X + b

	loss2 = huber_loss(Y, Y_predicted) 
	optimizer2 = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss2)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for i in range(100): # 100 epochs
			total_loss = 0
			for x, y in data:
				_, l = sess.run([optimizer2, loss2], feed_dict={X:x, Y:y})
				total_loss += l
			
			print('Epoch {0}: {1}'.format(i, total_loss/n_sample))

		w_out2, b_out2 = sess.run([w, b])

	# Evaluate the results
	print('w, b: (%f, %f)\n'%(w_out2, b_out2))

	# plot the results
	plt.plot( data[:,0], data[:, 1], 'bo', label='Real data' )
	# plt.plot( data[:,0], data[:,0]*w_out+b_out, 'r', label='Predicted data:square_loss'  )
	plt.plot( data[:,0], data[:,0]*w_out2+b_out2, 'g', label='Predicted data:huber_loss'  )
	plt.legend()
	plt.show()	

# # Y = wX^2 + uX + b
# X = tf.placeholder(tf.float32, name = 'X')
# Y = tf.placeholder(tf.float32, name = 'Y')

# w = tf.get_variable('weights_w', initializer=tf.constant(0.0))
# u = tf.get_variable('weights_u', initializer=tf.constant(0.0))
# b = tf.get_variable('weights', initializer=tf.constant(0.0))

# Y_predicted = w*X*X + u*X + b

# loss = tf.square(Y - Y_predicted, name = 'loss')

# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss)

# # Step3: run the session
# writer = tf.summary.FileWriter('./graphs', tf.get_default_graph()) #tensorboard --logdir="./graphs" --port 6006; http://localhost:6006/
# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())

# 	for i in range(100): # 100 epochs
# 		total_loss = 0
# 		for x, y in data:
# 			_, l = sess.run([optimizer, loss], feed_dict={X:x, Y:y})
# 			total_loss += l
		
# 		print('Epoch {0}: {1}'.format(i, total_loss/n_sample))
	  
# 	writer.close()
	
# 	w_out, b_out = sess.run([w, b])

# # Evaluate the results
# print('w, b: (%f, %f)\n'%(w_out, b_out))

def use_tfdata_for_data():
	dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))
	print(dataset.output_types)
	print(dataset.output_shapes)

	# iterator = dataset.make_one_shot_iterator()
	iterator = dataset.make_initializable_iterator()
	X, Y = iterator.get_next()

	w = tf.get_variable('weights', initializer=tf.constant(0.0))
	b = tf.get_variable('bias', initializer=tf.constant(.0))

	Y_predicted = w*X + b

	loss2 = huber_loss(Y, Y_predicted) 
	optimizer2 = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss2)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for i in range(100):
			sess.run(iterator.initializer)
			total_loss = .0
			try:
				while True:
					_, l = sess.run([optimizer2, loss2])
					total_loss += l
			except tf.errors.OutOfRangeError:
				pass

			print('Epoch {0}: {1}'.format(i, total_loss/n_sample))

		w_out2, b_out2 = sess.run([w, b])

	# Evaluate the results
	print('w, b: (%f, %f)\n'%(w_out2, b_out2))

def logistic_regression_mnist():
	# Define paramaters for the model
	learning_rate = 0.01
	batch_size = 128
	n_epochs = 30
	n_train = 60000
	n_test = 10000
	
	# prepare the data
	mnist_folder = 'data/mnist'
	utils.download_mnist(mnist_folder)
	train, val, test = utils.read_mnist(mnist_folder, flatten=True)

	print('{0}, {1}'.format(n_test, len(test[0])))
	print(test)
	assert n_test == len(test[0]), 'Number of testing dataset is wrong!'

	train_data = tf.data.Dataset.from_tensor_slices(train)
	train_data = train_data.shuffle(10000)
	test_data = tf.data.Dataset.from_tensor_slices(test)

	train_data = train_data.batch(batch_size)
	test_data = test_data.batch(batch_size)

	iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
	img, label = iterator.get_next()
	train_init = iterator.make_initializer(train_data)
	test_init = iterator.make_initializer(test_data)
	
	# Construct the optimization pipeline
	
	# Step 1: create weights and bias
	# w is initialized to random variables with mean of 0, stddev of 0.01
	# b is initialized to 0
	# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
	# shape of b depends on Y
	# X corresponds to an image with size of (28, 28) which is flattened to have the dimension of (1, 784)
	# Y is a vector containg 10 digits
	w = tf.get_variable(name='weights', shape=(784, 10), initializer=tf.random_normal_initializer(0, 0.01))
	b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())
    
	# Step 2: build model
	# the model that returns the logits.
	# this logits will be later passed through softmax layer
	logits = tf.matmul(img, w) + b 

	# Step 3: define loss function
	# use cross entropy of softmax of logits as the loss function
	entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label, name='entropy')
	loss = tf.reduce_mean(entropy, name='loss') # computes the mean over all the examples in the batch

	# Step 4: define training op 
	# using gradient descent with learning rate of 0.01 to minimize loss
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	# Step 5: calculate accuracy with test set
	preds = tf.nn.softmax(logits)
	correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
	accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

	writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())

	# run the optimization
	with tf.Session() as sess:
		sess.run( tf.global_variables_initializer() )

		for i in range(n_epochs):
			sess.run(train_init)
			total_loss = 0
			n_batches = 0
			try:
				while True:
					_, l = sess.run([optimizer, loss])
					total_loss += l
					n_batches += 1
			except tf.errors.OutOfRangeError:
				pass
			print('Avrage loss epoch {0}: {1}'.format(i, total_loss/n_batches))

		# test the model
		sess.run(test_init)
		total_correct_preds = 0
		try:
			while True:
				accuracy_batch = sess.run(accuracy)
				total_correct_preds += accuracy_batch
		except tf.errors.OutOfRangeError:
			pass 
		
		print('Accuracy {0}'.format(total_correct_preds/n_test))
	
	writer.close()

def tensorflow_dataslice_format():
	data = np.zeros( (3, 4, 5) )
	data2 = np.ones((3,))
	dataset = tf.data.Dataset.from_tensor_slices((data, data2))
	iterator = dataset.make_one_shot_iterator()
	value1, value2 = iterator.get_next()

	"""
	In tensorflow or numpy, the data is arranged in list[] format, and the tuple (),
	is just used to zip/combine the data. Note: tuple does not correspond to a data, which
	is different from the tuple in python.
	"""

	with tf.Session() as sess:
		no = 1
		try:
			while True:
				x, y = sess.run([value1, value2])
				print("Data {0}: {1}, {2}".format(no, x, y))
				no += 1
		except tf.errors.OutOfRangeError:
			pass	


def tensorflow_common_ops():
# Simple exercies to get used to tensorflow API

###############################################################################
# 1a: Create two random 0-d tensors x and y of any distribution.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
# Hint: look up tf.cond()
# I do the first problem for you
###############################################################################
	# # tf.set_random_seed(10) # one seed corresponds to a randomness generator  
	# x = tf.random_uniform([], seed = 5)
	# y = tf.random_uniform([])
	# v = tf.cond(x>y, lambda: x+y, lambda: x-y) # only function format is provided not including its params
	# with tf.Session() as sess:
	# 	xx, yy, vv = sess.run([x, y, v])
	# 	print("x:{0}; y:{1}; z:{2}".format(xx, yy, vv))

###############################################################################
# 1b: Create two 0-d tensors x and y randomly selected from the range [-1, 1).
# Return x + y if x < y, x - y if x > y, 0 otherwise.
# Hint: Look up tf.case().
###############################################################################
	# tf.set_random_seed(10)
	# x = tf.random_uniform([])
	# y = tf.random_uniform([])
	# v = tf.case({x<y:lambda:x+y, x>y:lambda:x-y}, default=lambda:tf.constant(.0), exclusive=True)
	# # ops in tensorflow should be tensor, py number cannot be used directly.
	# with tf.Session() as sess:
	# 	xx, yy, vv = sess.run((x, y, v))
	# 	print("x:{0}; y:{1}; v:{2}".format(xx, yy, vv))

###############################################################################
# 1c: Create the tensor x of the value [[0, -2, -1], [0, 1, 2]] 
# and y as a tensor of zeros with the same shape as x.
# Return a boolean tensor that yields Trues if x equals y element-wise.
# Hint: Look up tf.equal().
###############################################################################		
	# x = tf.constant([ [0,-2,-1], [0,1,2] ])
	# y = tf.zeros_like(x)
	# v = tf.equal(x, y)
	# with tf.Session() as sess:
	# 	xx, yy, vv = sess.run((x, y, v))
	# 	print("x:{0}; y:{1}; v:{2}".format(xx, yy, vv))

###############################################################################
# 1d: Create the tensor x of value 
# [29.05088806,  27.61298943,  31.19073486,  29.35532951,
#  30.97266006,  26.67541885,  38.08450317,  20.74983215,
#  34.94445419,  34.45999146,  29.06485367,  36.01657104,
#  27.88236427,  20.56035233,  30.20379066,  29.51215172,
#  33.71149445,  28.59134293,  36.05556488,  28.66994858].
# Get the indices of elements in x whose values are greater than 30.
# Hint: Use tf.where().
# Then extract elements whose values are greater than 30.
# Hint: Use tf.gather().
###############################################################################
	# data = [29.05088806,  27.61298943,  31.19073486,  29.35532951,
	# 		30.97266006,  26.67541885,  38.08450317,  20.74983215,
	# 		34.94445419,  34.45999146,  29.06485367,  36.01657104,
	# 		27.88236427,  20.56035233,  30.20379066,  29.51215172,
	# 		33.71149445,  28.59134293,  36.05556488,  28.66994858]
	# x = tf.constant(data)
	# y = tf.constant(30, shape=np.shape(data), dtype=tf.float32)
	# # Parameters of the tensorflow ops should not be tensor unless the param corresponds to a tensor data 
	# pos = tf.where(x > y)
	# q = tf.gather(x, pos)
	# with tf.Session() as sess:
	# 	xx, yy, pp, qq = sess.run((x, y, pos, q))
	# 	print("x:{0}; y:{1}; pos:{2}; q:{3}".format(xx, yy, pp, qq.flatten()))

###############################################################################
# 1e: Create a diagnoal 2-d tensor of size 6 x 6 with the diagonal values of 1,
# 2, ..., 6
# Hint: Use tf.range() and tf.diag().
###############################################################################
	# x = tf.range(1, 7, 1)
	# y = tf.diag(x)
	# with tf.Session() as sess:
	# 	xx, yy = sess.run((x, y))
	# 	print("x:{0}; y:{1}".format(xx, yy))

###############################################################################
# 1f: Create a random 2-d tensor of size 10 x 10 from any distribution.
# Calculate its determinant.
# Hint: Look at tf.matrix_determinant().
###############################################################################
	# x = tf.random_uniform([10, 10])
	# y = tf.matrix_determinant(x)
	# with tf.Session() as sess:
	# 	xx, yy = sess.run((x, y))
	# 	print("x:{0}; y:{1}".format(xx, yy))

###############################################################################
# 1g: Create tensor x with value [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9].
# Return the unique elements in x
# Hint: use tf.unique(). Keep in mind that tf.unique() returns a tuple.
###############################################################################
	# x = tf.constant([5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9])
	# y, idx = tf.unique(x)
	# with tf.Session() as sess:
	# 	xx, yy, id = sess.run((x, y, idx))
	# 	print("x:{0}; y:{1}; id:{2}".format(xx, yy, id))

###############################################################################
# 1h: Create two tensors x and y of shape 300 from any normal distribution,
# as long as they are from the same distribution.
# Use tf.cond() to return:
# - The mean squared error of (x - y) if the average of all elements in (x - y)
#   is negative, or
# - The sum of absolute value of all elements in the tensor (x - y) otherwise.
# Hint: see the Huber loss function in the lecture slides 3.
###############################################################################
	tf.set_random_seed(10)
	x = tf.random_normal([300])
	y = tf.random_normal([300])
	sum = tf.reduce_sum(x-y)
	ave = sum / tf.cast(tf.size(x), tf.float32)
	def f1(): 
		return tf.reduce_mean(tf.square(x-y))
	def f2():
		return tf.reduce_sum( tf.abs(x-y) )

	ret = tf.cond( ave < 0, f1, f2)
	with tf.Session() as sess:
		xx, yy, rr = sess.run((x, y, ret))
		print("ret:{0}".format(rr))

def word_2_vec():
	# Model hyperparameters
	VOCAB_SIZE = 50000
	BATCH_SIZE = 128
	EMBED_SIZE = 128            # dimension of the word embedding vectors
	SKIP_WINDOW = 1             # the context window
	NUM_SAMPLED = 64            # number of negative examples to sample
	LEARNING_RATE = 1.0
	NUM_TRAIN_STEPS = 100000
	VISUAL_FLD = 'visualization'
	SKIP_STEP = 5000

	# Parameters for downloading data
	DOWNLOAD_URL = 'http://mattmahoney.net/dc/text8.zip'
	EXPECTED_BYTES = 31344016
	NUM_VISUALIZE = 3000        # number of tokens to visualize

	# Prepare the training data
	gen = word_utils.batch_gen(DOWNLOAD_URL, EXPECTED_BYTES, VOCAB_SIZE, BATCH_SIZE,
	                            SKIP_WINDOW, VISUAL_FLD)
    dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32),
		(tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCHSIZE, 1]])))
	
	# Build the graph for word2vec model and train it
	
	   

if __name__ == '__main__':
	# use_placeholder_for_data()
	# use_tfdata_for_data()
	# logistic_regression_mnist()
	# tensorflow_dataslice_format()
	# tensorflow_common_ops()
	Test_Tensor_Dim()


