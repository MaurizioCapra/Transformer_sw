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

################################################  dataset  ###############################################################
BUFFER_SIZE = 20000
BATCH_SIZE = 1
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

train_batches = make_batches(train_examples, BUFFER_SIZE, BATCH_SIZE)
val_batches = make_batches(val_examples, BUFFER_SIZE, BATCH_SIZE)


##############################################  TRANSFORMERS  #############################################################
N = 4  # number of encoder/decoder modules
d_model = 128  # model size, embedding size
multi_head = 8  # number of heads
dk = 16  # query and key size
dv = 16  # value size
dff = 512  # feed-forward hidden layer size
input_vocab_size = tokenizers.pt.get_vocab_size()  # input vocabulary size
target_vocab_size = tokenizers.en.get_vocab_size()  # target vocabulary size
input_sequence_length = 1000  # input sequence length, input sentence length
target_sequence_length = 1000  # output sequence length, output sentence length

learning_rate = CustomLearningRate(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

transformer = Transformer(
    N, d_model, multi_head, dk, dv, dff,
    input_vocab_size, target_vocab_size,
    input_sequence_length, target_sequence_length)

# temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
# temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)
#
# fn_out, _ = transformer(temp_input, temp_target, training=False,
#                                encoder_mask=None,
#                                look_ahead_mask=None,
#                                decoder_padding_mask=None)
#
# print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)

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

    # inp -> portuguese, tar -> english
    for (batch, (inp, tar)) in enumerate(train_batches):
        train_step(inp, tar)

        if batch % 50 == 0:
            print(
                f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')



######################################################################################################################
# # model parameters
# dmodel = 512
# dk = 64
# dv = 64
# h = 8
#
# matrix1 = np.ones((1, 2, 2))
# matrix2 = np.ones((1, 2))

#
# # test with ad-hoc kernel
# inputs1 = keras.Input(shape=(2, 2), )
# outputs1 = SelfAttention(2)(inputs1)
# model1 = keras.Model(inputs1, outputs1)
# model1.summary()
# # keras.utils.plot_model(model, "Self_attention.png", show_shapes=True)
# model1.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=keras.optimizers.Adam(),
#     metrics=["accuracy"],
# )

# x = model1.predict(matrix1)
# print("The custom model provides\n", x)
#
# # test with dense kernel
#
# inputs2 = keras.Input(shape=(2, 2), )
# outputs2 = SelfAttention_dense(2, mask=True)(inputs2)
# model2 = keras.Model(inputs2, outputs2)
# model2.summary()
# # keras.utils.plot_model(model, "Self_attention.png", show_shapes=True)
# model2.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=keras.optimizers.Adam(),
#     metrics=["accuracy"],
# )
#
# x = model2.predict(matrix1)
# print("The custom model provides\n", x)
#
# # test with multihead
#
# inputs3 = keras.Input(shape=(2, 2), )
# outputs3 = multi_head_attention(2, 3)(inputs3)
# model3 = keras.Model(inputs3, outputs3)
# model3.summary()
# # keras.utils.plot_model(model, "Self_attention.png", show_shapes=True)
# model3.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=keras.optimizers.Adam(),
#     metrics=["accuracy"],
# )
#
# x = model3.predict(matrix1)
# print("The custom model provides\n", x)
# # print("The custom model provides the otput\n",x[0])
# # print("The custom model provides the patial output\n",x[1])
# # print("The custom model providesthe concatenated partial output\n",x[2])
#
# # x = tf.ones((2,3))
# # size = x.shape[1]
# # mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
# # x += (mask * -1e9)
# # print(x)
#
# inputs4 = keras.Input(shape=(2, 2, 2), )
# outputs4 = keras.layers.Dense(2, use_bias=False)(inputs4)
#
# model4 = keras.Model(inputs4, outputs4)
# model4.summary()
# model4.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=keras.optimizers.Adam(),
#     metrics=["accuracy"],
# )

###################################################################################### test output size
# from masked_scaled_dot_product import *
# def print_out(q, k, v):
#   temp_out, temp_attn = scaled_dot_product(
#       q, k, v, 64, None)
#   print ('Attention weights are:')
#   print (temp_attn)
#   print ('Output is:')
#   print (temp_out)
#
#
#
# np.set_printoptions(suppress=True)
#
# temp_k = tf.constant([[10,0,0],
#                       [0,10,0],
#                       [0,0,10],
#                       [0,0,10]], dtype=tf.float32)  # (4, 3)
#
# temp_v = tf.constant([[   1,0],
#                       [  10,0],
#                       [ 100,5],
#                       [1000,6]], dtype=tf.float32)  # (4, 2)
#
# # This `query` aligns with the second `key`,
# # so the second `value` is returned.
# temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
# print_out(temp_q, temp_k, temp_v)
# temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
# print_out(temp_q, temp_k, temp_v)
# temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)  # (1, 3)
# print_out(temp_q, temp_k, temp_v)
# temp_q = tf.constant([[0, 0, 10], [0, 10, 0], [10, 10, 0]], dtype=tf.float32)  # (3, 3)
# print_out(temp_q, temp_k, temp_v)
#
#
# ############################################################################
# from MultiHeadAttention import MultiHeadAttention
# temp_mha = MultiHeadAttention(512, 8, 64, 64)
# y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
# out, attn = temp_mha(y, k=y, q=y, mask=None)
# print(out.shape, attn.shape)
# ###########################################################################
# from point_wise_feed_forward_network import point_wise_feed_forward_network
# sample_ffn = point_wise_feed_forward_network(512, 2048)
# print(sample_ffn(tf.random.uniform((64, 50, 512))).shape)
# ##############################################################################
# from EncoderModule import EncoderModule
# sample_encoder_layer = EncoderModule(512, 8, 64, 64, 2048)
#
# sample_encoder_layer_output, sample_encoder_layer_output_weight = sample_encoder_layer(
#     tf.random.uniform((64, 43, 512)), False, None)
#
# print(sample_encoder_layer_output.shape, sample_encoder_layer_output_weight.shape)  # (batch_size, input_seq_len, d_model)
# ###########################################################################
# from DecoderModule import DecoderModule
# sample_decoder_layer = DecoderModule(512, 8, 64, 64, 2048)
#
# sample_decoder_layer_output, sample_encoder_layer_output_sublayer1, sample_encoder_layer_output_sublayer2 = sample_decoder_layer(
#     tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,
#     False, None, None)
#
# print(sample_decoder_layer_output.shape, sample_encoder_layer_output_sublayer1.shape, sample_encoder_layer_output_sublayer2.shape)  # (batch_size, target_seq_len, d_model)
# ###########################################################################
# from Encoder import Encoder
# sample_encoder = Encoder(2, 512, 8, 64, 64,
#                          2048, 8500,
#                          10000)
# temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)
#
# sample_encoder_output, weights = sample_encoder(temp_input, training=False, mask=None)
#
# print (sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)
# ############################################################################
# from Decoder import Decoder
# sample_decoder = Decoder(2, 512, 8, 64, 64,
#                          2048, 8000,
#                          5000)
# temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)
#
# output, attn = sample_decoder(temp_input,
#                               encoder_output=sample_encoder_output,
#                               training=False,
#                               look_ahead_mask=None,
#                               padding_mask=None)
#
# print(output.shape, attn['decoder_module1_sublayer1'].shape, attn['decoder_module1_sublayer2'].shape, attn['decoder_module2_sublayer1'].shape, attn['decoder_module2_sublayer2'].shape)
# ####################################################################
# from prova import Decoder2
# sample_decoder = Decoder2(num_layers=2, d_model=512, num_heads=8,
#                          dff=2048, target_vocab_size=8000,
#                          maximum_position_encoding=5000)
# temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)
#
# output, attn = sample_decoder(temp_input,
#                               enc_output=sample_encoder_output,
#                               training=False,
#                               look_ahead_mask=None,
#                               padding_mask=None)
#
# print(output.shape,attn['decoder_layer1_block1'].shape, attn['decoder_layer1_block2'].shape, attn['decoder_layer2_block1'].shape, attn['decoder_layer2_block2'].shape)
# ##################################################################################
# from Transformer import Transformer
# sample_transformer = Transformer(
#     2, 512, 8, 64, 64, 2048,
#     8500, 8000,
#     10000, 5000)
#
# temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)
# temp_target = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)
#
# fn_out, attn = sample_transformer(temp_input, temp_target, training=False,
#                                encoder_mask=None,
#                                look_ahead_mask=None,
#                                decoder_padding_mask=None)
#
# print(fn_out.shape, attn['decoder_module1_sublayer1'].shape, attn['decoder_module1_sublayer2'].shape, attn['decoder_module2_sublayer1'].shape, attn['decoder_module2_sublayer2'].shape)  # (batch_size, tar_seq_len, target_vocab_size)
# x=8