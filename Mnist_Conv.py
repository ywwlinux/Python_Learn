"""  This code is the conv-based model for the minist recognition  """

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

import utils
import time

def conv_relu(inputs, k_size, stride, padding, nFilter, scope_name='CONV'):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            nChannel = inputs.shape[-1]
            kernel = tf.get_variable('kernel', [k_size, k_size, nChannel, nFilter],
                                     initializer=tf.truncated_normal_initializer())
            biases = tf.get_variable('biases', [nFilter], initializer=tf.random_normal_initializer())
            conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding=padding)

        return tf.nn.relu(conv+biases, name = scope.name)

def maxpool(inputs, k_size, stride, padding='VALID', scope_name='POOL'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool = tf.nn.max_pool(inputs, ksize=[1, k_size, k_size, 1], 
                        strides=[1, stride, stride, 1], padding=padding)

    return pool

def fully_connected(inputs, out_dim, scope_name='fc'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]
        w = tf.get_variable('weights', [in_dim, out_dim], initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer(0.0))
        out = tf.matmul(inputs, w) + b

    return out

class MinistConvModel:
    def __init__(self, dataset_train, dataset_test, nTest, data_nLabel, batchSize):
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.batchSize = batchSize
        self.n_classes = data_nLabel
        self.n_test = nTest

        self.lr = 0.001
        self.keep_prob = tf.constant(0.75)
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.skip_step = 20
        self.training = tf.Variable(True, trainable=False)

    def Train(self, nepoch):
        # Step 1: load data
        self._importData()

        # Step 2: build graph
        self._buildGraph()

        # Step 3: train the network
        self._trainImp(nepoch)

    # functions for building graph and training
    def _trainImp(self, nepoch):
        utils.safe_mkdir('checkpoints')
        utils.safe_mkdir('checkpoints/convnet_mnist')
        writer = tf.summary.FileWriter('./graphs/convnet', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_mnist/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.gstep.eval()
            for epoch in range(nepoch):
                # train one epoch
                sess.run(self.train_init)
                step = self.train_one_epoch(sess, saver, writer, epoch, step)

                # eval once
                sess.run(self.test_init)
                self.eval_once(sess, writer, epoch, step)

        writer.close()

    def train_one_epoch(self, sess, saver, writer, epoch, step):
        start_time = time.time()
        # self.training = tf.constant(True)  # does it work? It's a Tensor constant which is dependened by the dropout tensor
        tf.assign(self.training, True)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.opt, self.loss, self.summary_op])  # summary of loss and accuracy on traing data every batch-step
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, 'checkpoints/convnet_mnist/mnist-convnet', step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss/n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, writer, epoch, step):
        start_time = time.time()
        # self.training = False   # does it work?
        tf.assign(self.training, False)
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op]) # summary of loss and accuracy on testing data every batch-step
                # writer.add_summary(summaries, global_step=step) # step does not change. Does it make sense? I don't think so, so we should comment the code of this line.
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds/self.n_test))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def _buildGraph(self):
        with tf.name_scope('Graph'):
            # conv-layer 1
            conv1 = conv_relu(self.image, k_size=5, stride = 1, padding='SAME', nFilter=32,
                scope_name='CONV1')
            pool1 = maxpool(conv1, 2, 2, 'VALID', 'POOL1')

            # conv-layer 2
            conv2 = conv_relu(pool1, k_size=5, stride = 1, padding='SAME', nFilter=64,
                scope_name='CONV2')
            pool2 = maxpool(conv2, 2, 2, 'VALID', 'POOL2')

            # fc_1
            feature_dim = pool2.shape[1]*pool2.shape[2]*pool2.shape[3]
            pool2 = tf.reshape(pool2, [-1, feature_dim])
            fc = tf.nn.relu( fully_connected(pool2, 1024, 'fc') )
            # dropout = tf.layers.dropout(fc, self.keep_prob, training=self.training, name='dropout')
            dropout = tf.nn.dropout(fc, self.keep_prob, name='dropout')

            # fc_softmax
            self.logits = fully_connected(dropout, self.n_classes, 'logits')

        with tf.name_scope('Loss'):
            # loss
            entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label, name='entropy')
            self.loss = tf.reduce_mean(entropy, name='loss') # computes the mean over all the examples in the batch

        with tf.name_scope('Optimize'):
            # optimizer
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)

        self._evalAccuracy()
        self.summary()

    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def _evalAccuracy(self):
        """ Count the number of right prediction in a batch 
        """
        with tf.name_scope('Predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    def _importData(self):
        with tf.name_scope('Data'):
            train_data = self.dataset_train.batch(self.batchSize)
            test_data = self.dataset_test.batch(self.batchSize)
            
            iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
            self.image, self.label = iterator.get_next() 
            self.train_init = iterator.make_initializer(train_data)
            self.test_init = iterator.make_initializer(test_data)

if __name__ == '__main__':
    # Load data
    mnist_folder = 'data/mnist'
    utils.download_mnist(mnist_folder)
    train, val, test = utils.read_mnist(mnist_folder, flatten=False)

    train_imgs = train[0]
    test_imgs = test[0]
    train_imgs = np.expand_dims(train_imgs, -1)
    test_imgs = np.expand_dims(test_imgs, -1)
    # train[0] = np.expand_dims(train[0], -1)
    # test[0] = np.expand_dims(test[0], -1)

    train_data = tf.data.Dataset.from_tensor_slices((train_imgs, train[1]))
    train_data = train_data.shuffle(10000)
    test_data = tf.data.Dataset.from_tensor_slices((test_imgs, test[1]))

    model = MinistConvModel(train_data, test_data, test[0].shape[0], test[1].shape[-1], 128)
    model.Train(30)    

"""
In ops of tensorflow, if their parames are changable during run, 
the params should be tensor which varies by the run of the static graph
"""

