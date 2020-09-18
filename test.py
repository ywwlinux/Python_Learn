import tensorflow as tf
import numpy as np



def tensorflow_dataslice_format2():
    """ np.array is equivalent to the tensor in tensorflow.
    A data is a tensor and tensor slices mean a set of data 
    which is stored in a np.array where each element at top level corresponds to a tensor 
    """
    x = np.array( [ [1, 2, 3], [4, 4, 5], [6,7,8], [9,10,11]  ])
    x_data = tf.data.Dataset.from_tensor_slices(x)
    # x_data = x_data.batch(2)
    iterator = x_data.make_initializable_iterator()
    x1= iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        sess.run(tf.global_variables_initializer())
        try:
            xx = sess.run(x1)
            # print("Data {0}:".format(xx))
            print(xx)
        except tf.errors.OutOfRangeError:
            pass

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

def fun():
    fun1()

def fun1():
    pass    

if __name__ == '__main__':
    # tensorflow_dataslice_format2()
    fun()