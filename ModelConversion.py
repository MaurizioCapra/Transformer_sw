 #######import standard libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import collections
import logging
import os
import pathlib
import re
import string
import sys
import time

import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import sacrebleu
import trax
from tqdm import tqdm
from load_transformer import load_transformer
from create_mask import create_masks
##############################################  LOAD MODEL  ############################################################
N = 4  # number of encoder/decoder modules
d_model = 512  # model size, embedding size
multi_head = 8  # number of heads
dk = 64  # query and key size
dv = 64  # value size
dff = 2048  # feed-forward hidden layer size
input_vocab_size = 33300  # input vocabulary size
target_vocab_size = 33300  # target vocabulary size
input_sequence_length = 128  # input sequence length, input sentence length
target_sequence_length = 128  # output sequence length, output sentence length



transformer = load_transformer(N, d_model, multi_head, dk, dv, dff,
                               input_vocab_size, target_vocab_size,
                               input_sequence_length, target_sequence_length)


sentence = np.ones(128)
start = np.ones(128)
encoder_input = tf.convert_to_tensor(sentence)
encoder_input = tf.expand_dims(encoder_input, 0)
output = tf.convert_to_tensor(start)
output = tf.expand_dims(output, 0)

enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

# predictions.shape == (batch_size, seq_len, vocab_size)
_, _= transformer(encoder_input,
                  output,
                  False,
                  enc_padding_mask,
                  combined_mask,
                  dec_padding_mask)

#Display model architecture
transformer.summary()

# transformer.save_model('saved_model/my_model')
transformer.save_weights('saved_model/my_model')