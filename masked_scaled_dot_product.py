import tensorflow as tf
import numpy as np


def scaled_dot_product(Q, K, V, scale, mask=None):
    x = tf.matmul(Q, K, transpose_b=True)
    x = tf.math.divide(x, np.sqrt(scale))

    if mask is not None:
        # mask = f(mask, x)
        x += (mask * -1e9)
    # attention_weights = tf.nn.softmax(x, axis=-1)
    attention_weights = tf.keras.activations.softmax(x)
    return tf.matmul(attention_weights, V), attention_weights

# def create_look_ahead_mask(size):
#   mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
#
#   return mask
#
# def create_padding_mask(seq):
#   seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
#   # add extra dimensions to add the padding
#   # to the attention logits.
#   x=seq[:, tf.newaxis, tf.newaxis, :]
#   #print(x.shape)
#   return  x # (batch_size, 1, 1, seq_len)
#
# def f(mask, x):
#     assert ((mask == "padding")|(mask == "look_ahead"))
#     return {
#         'padding': create_padding_mask(x),
#         'look_ahead': create_look_ahead_mask(x.shape[1]),
#     }[mask]