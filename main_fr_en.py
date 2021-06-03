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
import trax
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
# from tokenizer import *
from create_mask import create_masks
import os

################################################  hparams  ###############################################################
# N = 6  # number of encoder/decoder modules
# d_model = 512  # model size, embedding size
# multi_head = 8  # number of heads
# dk = 64  # query and key size
# dv = 64  # value size
# dff = 2048  # feed-forward hidden layer size
# input_vocab_size = 33300  # input vocabulary size
# target_vocab_size = 33300  # target vocabulary size
# input_sequence_length = 2048  # input sequence length, input sentence length
# target_sequence_length = 2048  # output sequence length, output sentence length

N = 4  # number of encoder/decoder modules
d_model = 512  # model size, embedding size
multi_head = 8  # number of heads
dk = 64  # query and key size
dv = 64  # value size
dff = 2048  # feed-forward hidden layer size
input_vocab_size = 33584  # input vocabulary size
target_vocab_size = 33584  # target vocabulary size
input_sequence_length = 128  # input sequence length, input sentence length
target_sequence_length = 128  # output sequence length, output sentence length
BUFFER_SIZE = 20000
BATCH_SIZE = 64
################################################  dataset  ###############################################################
# examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
# train_examples, val_examples = examples['train'], examples['validation']
# # data_dir = os.path.expanduser("/home/maurizio/tensorflow_datasets/downloads/extracted")
# #
# config = tfds.translate.wmt.WmtConfig(version="0.0.1",
#                                       language_pair=("fr", "en"),
#                                       subsets={
#         tfds.Split.TRAIN: [
#             "europarl_v7", "commoncrawl", "multiun",
#             "newscommentary_v9", "gigafren"
#         ],
#         tfds.Split.VALIDATION: [
#              "newsdev2014", "newstest2013"
#         ],
#         tfds.Split.TEST: [
#             "newstest2014"
#         ]
#     })
#
#
# builder = tfds.builder("wmt_translate", data_dir= "/home/maurizio/tensorflow_datasets/downloads", config=config)
#
# ds = builder.as_dataset()
# print(ds)
# for i in ds:
#     print(i)
#
# Use the tensorflow-numpy backend.
trax.fastmath.set_backend('tensorflow-numpy')


#import resource
#low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
#resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

train_stream = trax.data.tf_inputs.TFDS('wmt14_translate', keys=('fr', 'en'), train=True)()
eval_stream = trax.data.tf_inputs.TFDS('wmt14_translate', keys=('fr', 'en'), train=False)()
# print(next(train_stream))  # See one example.

# train_stream_batch = trax.data.inputs.batch(train_stream, BATCH_SIZE)
# eval_stream_batch = trax.data.inputs.batch(eval_stream, BATCH_SIZE)

# print(next(train_stream_batch))

EOS =  list(trax.data.tf_inputs.tokenize(['stop'], vocab_file="endefr_32k.subword"))[0]
START =  list(trax.data.tf_inputs.tokenize(['start'], vocab_file="endefr_32k.subword"))[0]

def append_eos(stream):
    for (inputs, targets) in stream:
        inputs_with_eos = [START] + list(inputs) + [EOS]
        targets_with_eos = [START] + list(targets) + [EOS]
        yield np.array(inputs_with_eos), np.array(targets_with_eos)

data_pipeline = trax.data.Serial(
    trax.data.tf_inputs.Tokenize(vocab_file='endefr_32k.subword'),
    lambda _: append_eos(_),
    trax.data.Shuffle(),
    trax.data.FilterByLength(max_length=input_sequence_length),
    trax.data.BucketByLength(boundaries =  [  8,  16,  32,  64, 128],
                             batch_sizes = [BATCH_SIZE, BATCH_SIZE, BATCH_SIZE, BATCH_SIZE, BATCH_SIZE, BATCH_SIZE],
                             strict_pad_on_len=True),
    #trax.data.AddLossWeights(id_to_mask=0)
  )

train_batches_stream = data_pipeline(train_stream)
eval_batches_stream = data_pipeline(eval_stream)
# for (batch, (inp, tar)) in enumerate(train_batches_stream):
#     print(batch, inp, tar)
# example_batch = next(train_batches_stream)

# for item in train_batches_stream:
#     print(item)

# print(f'shapes = {[x.shape for x in example_batch]}')
# print(f'shapes = {[x for x in example_batch]}')

# examples, metadata = tfds.load("wmt_translate", with_info=True, as_supervised=True)
# train_examples, val_examples = examples['train'], examples['validation']
#
#
# train_batches = make_batches(train_examples, BUFFER_SIZE, BATCH_SIZE)
# val_batches = make_batches(val_examples, BUFFER_SIZE, BATCH_SIZE)

# ##############################################  TRANSFORMERS  #############################################################
learning_rate = CustomLearningRate(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

transformer = Transformer(
    N, d_model, multi_head, dk, dv, dff,
    input_vocab_size, target_vocab_size,
    input_sequence_length, target_sequence_length)

# # temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
# # temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)
# #
# # fn_out, _ = transformer(temp_input, temp_target, training=False,
# #                                encoder_mask=None,
# #                                look_ahead_mask=None,
# #                                decoder_padding_mask=None)
# #
# # print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)
#
################################################  TRAINING  ###############################################################
EPOCHS = 20

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    encoder_mask, combined_mask, decoder_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, True, encoder_mask, combined_mask, decoder_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))


for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()
    
    # inp -> english, tar -> german
    for (batch, (inp, tar)) in enumerate(train_batches_stream):
        train_step(inp, tar)

        if batch % 50 == 0:
            print(
                f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        if (batch+1) % 10000 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch + 1}, batch {batch} at {ckpt_save_path}')

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

