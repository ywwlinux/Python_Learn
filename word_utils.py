from collections import Counter
import random
import os
import sys
import zipfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

import utils

def read_data(file_path):
    """ Read data into a list of tokens
    There should be 17,005,207 tokens
    The text is uniformly represented as unicode by as_str no matter in python 2 or 3
    """
    with zipfile.ZipFile(file_path) as f:
        s = tf.compat.as_str(f.read(f.namelist()[0]))
        words = s.split()
        # with open('as_str.txt', 'w') as ff:
        #     ff.write(s)
        # in s, the word is separated by a space
        return words

def build_vocab(words, vocab_size, vis_folder):
    """ Build vocabulary of VOCAB_SIZE most frequent words and 
    write it to visualization/vocab.tsv
    """
    utils.safe_mkdir(vis_folder)
    file = open(os.path.join(vis_folder, 'vocab.tsv'), 'w')

    dictionary = dict()
    count = [('UNK', -1)]
    index = 0
    count.extend(Counter(words).most_common(vocab_size-1))

    for word, _ in count:
        dictionary[word] = index
        index += 1
        file.write(word+'\n') # each line a word

    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    file.close()

    return dictionary, index_dictionary

def convert_words_to_index(words, dictionary):
     """ Return for each word its index in the dictionary """
     return [ dictionary[word] if word in dictionary else 0 for word in words ]

def generate_sample(index_words, context_window_size):
    """  Form training pairs according to the skip-gram model  """
    for index, center in enumerate(index_words):
        context = random.randint(1, context_window_size)
        # get a random target before the center word
        for target in index_words[ max(0, index-context):index ]:
            yield center, target
        # get a random target after the center word
        for target in index_words[ index+1: index+context+1 ]:
            yield center, target

def most_common_word(vis_folder, num_vis):
    """ Create a list of NUM_VIS most frequent words to visualize on Tensorboard 
    saved to visualization/vocab_[num_visualize].tsv
    """
    # words = open(os.path.join(vis_folder, 'vocab.tsv'), 'r').readlines()[:num_vis]
    words = open(os.path.join(vis_folder, 'vocab.tsv'), 'r').readlines()[:num_vis]
    file = open(os.path.join(vis_folder, 'vocab_')+str(num_vis)+'.tsv', 'w')
    for word in words:
        file.write(word)
    file.close()

test_num = 0

def batch_gen(download_url, expected_byte, vocab_size, batch_size, skip_window, vis_folder):
    """ Note that (1) at each time the generate is initialized the batch_gen() should be run from the beginning
              and (2) the generator will be actually run when the first yield is needed by 'next' op
    """
    local_dest = 'data/text8.zip'
    utils.download_one_file(download_url, local_dest, expected_byte)
    words = read_data(local_dest)    
    dictionary, _ = build_vocab(words, vocab_size, vis_folder) # assign each word an index to label it
    index_words = convert_words_to_index(words, dictionary) # replace each word with a number
    del words # save memory
    
    # use the order of the words in sentence to generate samples
    # each pair of word and its before/after forms a sample
    single_gen = generate_sample(index_words, skip_window)

    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(single_gen)
        
        yield center_batch, target_batch



            