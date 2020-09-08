import tensorflow as tf

import numpy as np

# import matplotlib.pyplot as plt

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


if __name__ == '__main__':
	# use_placeholder_for_data()
	use_tfdata_for_data()


