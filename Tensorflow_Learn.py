import tensorflow as tf

import numpy as np

import utils
import word_utils

# import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

' Code for learning tensorflow '

__author__ = 'Wenwu Yang'

#tensorboard --logdir="./graphs" --port 6006; http://localhost:6006/

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
# writer = tf.summary.FileWriter('./graphs', tf.get_default_graph()) 
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
	data = [29.05088806,  27.61298943,  31.19073486,  29.35532951,
			30.97266006,  26.67541885,  38.08450317,  20.74983215,
			34.94445419,  34.45999146,  29.06485367,  36.01657104,
			27.88236427,  20.56035233,  30.20379066,  29.51215172,
			33.71149445,  28.59134293,  36.05556488,  28.66994858]
	x = tf.constant(data)
	y = tf.constant(30, shape=np.shape(data), dtype=tf.float32)
	# Parameters of the tensorflow ops should not be tensor unless the param corresponds to a tensor data 
	pos = tf.where(x > y)
	q = tf.gather(x, pos)
	with tf.Session() as sess:
		xx, yy, pp, qq = sess.run((x, y, pos, q))
		print("x:{0}; y:{1}; pos:{2}; q:{3}".format(xx, yy, pp, qq.flatten()))

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

	# Prepare the dataset
	# gen = word_utils.batch_gen(DOWNLOAD_URL, EXPECTED_BYTES, VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW, VISUAL_FLD)

	# for i in range(2):
	# 	print('epoch:{0}'.format(i))
	# 	center_words, target_words = next(gen)
	# 	print(center_words)

	def gen() : yield from word_utils.batch_gen(DOWNLOAD_URL, EXPECTED_BYTES, VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW, VISUAL_FLD)

	dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32), (tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE, 1])))
	
	# Build the graph for word2vec model
	# Step 1: get input, output from the dataset
	with tf.name_scope('data'):
		iterator = dataset.make_initializable_iterator()
		center_words, target_words = iterator.get_next()
	
	""" Step 2+3: define weights and embedding lookup
	In word2vec, it's actually the weights that we care about
	"""
	with tf.name_scope('embed'):
		embed_matrix = tf.get_variable('embed_matrix',
		          shape = [VOCAB_SIZE, EMBED_SIZE],
		          initializer=tf.random_uniform_initializer())
		embed = tf.nn.embedding_lookup(embed_matrix, center_words, name = 'embedding')

	# step 4: construct variable for NCE loss and define the loss function
	with tf.name_scope('loss'):
		nce_weights = tf.get_variable('nce_weight', shape=[VOCAB_SIZE, EMBED_SIZE])
		nce_bias = tf.get_variable('nce_bias', initializer=tf.zeros([VOCAB_SIZE]))
		
		# define loss function to be NCE loss function
		loss = tf.reduce_mean(tf.nn.nce_loss( weights = nce_weights,
		                       biases = nce_bias,
							   labels=target_words,
							   inputs=embed,
							   num_sampled=NUM_SAMPLED,
							   num_classes=VOCAB_SIZE, name = 'loss') )

	# Step 5: define optimizer
	with tf.name_scope('optimizer'):
		optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

	utils.safe_mkdir('checkpoints')

	with tf.Session() as sess:
		sess.run(iterator.initializer)
		sess.run(tf.global_variables_initializer())
		
		total_loss = .0 # we use this to calculate late average loss in the last skip_step step
		writer = tf.summary.FileWriter('graphs/word2vec_simple', sess.graph)

		for index in range(NUM_TRAIN_STEPS):
			try:
				loss_batch, _ = sess.run([loss, optimizer])
				total_loss += loss_batch
				if(index+1)%SKIP_STEP == 0:
					print('Average loss at step {0}:{1:5.1f}'.format(index+1, total_loss/SKIP_STEP))
					total_loss = .0
			except tf.errors.OutOfRangeError:
				sess.run(iterator.initializer) # Re-start the batch generator. Note that the gen function would be run from the starting once again
		
		writer.close()

if __name__ == '__main__':
	# use_placeholder_for_data()
	# use_tfdata_for_data()
	# logistic_regression_mnist()
	# tensorflow_dataslice_format()
	# tensorflow_common_ops()
	word_2_vec()


