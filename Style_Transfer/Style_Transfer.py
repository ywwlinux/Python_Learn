""" Implementation in Tensorflow of the paper
A Neural Algorithm of Artistic Style (Gatys et el., 2016)
The code is written by following the standford course CS20 
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # only warning and error are shown

import time

import numpy as np
import tensorflow as tf

import sys
sys.path.append('..')

import VGG_Loader
import utils

def setup():
    utils.safe_mkdir('checkpoints')
    utils.safe_mkdir('outputs')

class StyleTransfer(object):
    def __int__(self, content_img_path, style_img_path, img_width, img_height):
        '''
        img_width and img_height are the dimensions we expect from the generated images
        '''
        self.img_width = img_width
        self.img_height = img_height
        self.content_img = utils.get_resized_image(content_img_path, img_width, img_height)
        self.style_img = utils.get_resized_image(style_img_path, img_width, img_height)
        self.initial_img = utils.generate_noise_image(self.content_img, img_width, img_width)

        ##################################
        ## Create global step (gstep) and hyperparameters for the model
        self.content_layer = 'conv4_2'
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        self.content_w = 0.01
        self.style_w = 1
        self.style_layer_w = [0.5, 1.0, 1.5, 3.0, 4.0]
        self.gstep = tf.Variable(0, dtype = tf.int32, trainable=False, name = 'global_step')
        self.lr = 2.0
        ###################################

    def create_input(self):
        '''
        We will use one input_img as a placeholder for the content image, style image, and
        generated image, because:
        1. they have the same dimension
        2. we have to extract the same set of features from them
        We use a variable instead of a placeholder because we're, at the same time, training
        the generated image to get the desirable result.

        Note: image height corresponds to number of rows, not columns
        '''
        with tf.variable_scope('input') as scope:
            self.input_img = tf.get_variable('in_img',
                                             shape = ([1, self.img_height, self.img_width, 3]),
                                             dtype = tf.float32, initializer=tf.zeros_initializer())

    def load_vgg(self):
        '''
        Load the saved model parameter of VGG-19, using the input_img as the input
        to compute the output at each layer of vgg.

        During training, VGG-19 mean-centered all images and found the mean pixels to be
        [123.68, 116.779, 103.939] along RGB dimensions. We have to subtract this mean from our images.
        '''
        self.vgg = VGG_Loader.VGG(self.input_img)
        self.vgg.load()
        self.content_img -= self.vgg.mean_pixels
        self.style_img -= self.vgg.mean_pixels

    def _content_loss(self, P, F):
        '''
        Calculate the loss between the feature representations of the content image
        and the genrated image

        Inputs:
            P: content representation of the content image
            F: content representation of the generated image
            Read the assignment handout for more details

            Note: Don't use the coefficient 0.5 as defined in the paper.
            Use the coefficient defined in the assignment handout.
        '''

        


