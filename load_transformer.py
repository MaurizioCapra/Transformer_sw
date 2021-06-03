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

#######import custom classes and functions
# from self_attention import SelfAttention
# from self_attention_dense import SelfAttention_dense
# from multi_head_attention import multi_head_attention
from Transformer import Transformer
from CustomLearningRate import CustomLearningRate
from tokenizer import make_batches
from loss_and_metrics import *
from tokenizer import *
from create_mask import create_masks

def load_transformer(N, d_model, multi_head, dk, dv, dff,
        input_vocab_size, target_vocab_size,
        input_sequence_length, target_sequence_length):
##############################################  TRANSFORMERS  #############################################################

    learning_rate = CustomLearningRate(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    transformer = Transformer(
        N, d_model, multi_head, dk, dv, dff,
        input_vocab_size, target_vocab_size,
        input_sequence_length, target_sequence_length)



##########################################  RESTORING CHECKPOINT  #########################################################

    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

    #ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path)

    # if a checkpoint exists, restore the latest checkpoint.
    if tf.train.latest_checkpoint(checkpoint_dir=checkpoint_path):
        ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir=checkpoint_path)).expect_partial()
        print('Latest checkpoint restored!!')
    else:
        raise NameError('No checkpoint to restore: ERROR')

    return transformer

##############################################  SAVE MODEL  #############################################################

# input_arr = tf.random.uniform((1, input_sequence_length))
# target_arr = tf.random.uniform((1, target_sequence_length))
# outputs = transformer(input_arr, target_arr, False)
#
# transformer.save('./model/transformer')
