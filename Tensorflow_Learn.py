import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import os
import urllib
import gzip
import shutil

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

# Step 2: assemble the graph for linear regression neural net
# Y = wX + b
X = tf.placeholder(tf.float32, name = 'X')
Y = tf.placeholder(tf.float32, name = 'Y')

w = tf.get_variable('weights', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(.0))

Y_predicted = w*X + b

# # Y = wX^2 + uX + b
# X = tf.placeholder(tf.float32, name = 'X')
# Y = tf.placeholder(tf.float32, name = 'Y')

# w = tf.get_variable('weights_w', initializer=tf.constant(0.0))
# u = tf.get_variable('weights_u', initializer=tf.constant(0.0))
# b = tf.get_variable('weights', initializer=tf.constant(0.0))

# Y_predicted = w*X*X + u*X + b

loss = tf.square(Y - Y_predicted, name = 'loss')

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss)

# Step3: run the session
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph()) #tensorboard --logdir="./graphs" --port 6006; http://localhost:6006/
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


# Use the huber loss
def huber_loss(labels, predictions, delta=14.0):
    residual = tf.abs(labels - predictions)
    def f1(): return 0.5 * tf.square(residual)
    def f2(): return delta * residual - 0.5 * tf.square(delta)
    return tf.cond(residual < delta, f1, f2)

loss2 = huber_loss(Y, Y_predicted) 
optimizer2 = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss2)

# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())

# 	for i in range(100): # 100 epochs
# 		total_loss = 0
# 		for x, y in data:
# 			_, l = sess.run([optimizer2, loss2], feed_dict={X:x, Y:y})
# 			total_loss += l
		
# 		print('Epoch {0}: {1}'.format(i, total_loss/n_sample))

# 	w_out2, b_out2 = sess.run([w, b])

# # plot the results
# plt.plot( data[:,0], data[:, 1], 'bo', label='Real data' )
# plt.plot( data[:,0], data[:,0]*w_out+b_out, 'r', label='Predicted data:square_loss'  )
# plt.plot( data[:,0], data[:,0]*w_out2+b_out2, 'g', label='Predicted data:huber_loss'  )
# plt.legend()
# plt.show()	

def safe_mkdir(path):
	"""
	Create a directory if there isn't one already
	"""
	try:
		os.mkdir(path)
	except OSError:
		pass

def download_one_file(download_url, local_dest, execpted_byte=None, unzip_and_remove=False):
	"""
	Download the file from download_url into local_dest if the file doesn't already exists.
    If expected_byte is provided, check if the downloaded file has the same number of bytes.
    If unzip_and_remove is True, unzip the file and remove the zip file
	"""
	# download_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
	download_url = download_url.replace('\\', '/')
	if os.path.exists(local_dest) or os.path.exists(local_dest[:-3]):
		print('%s alreay exists'%local_dest)
	else:
		print('Downloading %s'%download_url)
		local_file, _ = urllib.request.urlretrieve(download_url, local_dest)
		file_stat = os.stat(local_dest)
		if execpted_byte:
			if file_stat.st_size == execpted_byte:
				print('Successfully downloaded %s'%local_dest)
				if unzip_and_remove:
					with gzip.open(local_dest, 'rb') as f_in, open(local_dest[:-3], 'wb') as f_out:
						shutil.copyfileobj(f_in, f_out)
					os.remove(local_dest)
			else:
				print('The downloaded file has unexpected number of bytes')

	

def download_mnist(path):
	"""
	Download and unzip the dataset mnist if it's not already downloaded
	Download from http://yann.lecun.com/exdb/mnist
	"""

	safe_mkdir(path)
	url = 'http://yann.lecun.com/exdb/mnist'
	filenames = ['train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz',
                't10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz']
	expected_bytes = [9912422, 28881, 1648877, 4542]

	for filename, byte in zip(filenames, expected_bytes):
		download_url = os.path.join(url, filename)
		local_dest = os.path.join(path, filename)
		download_one_file(download_url, local_dest, byte, True)




