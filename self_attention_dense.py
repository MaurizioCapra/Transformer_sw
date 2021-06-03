import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class SelfAttention_dense(layers.Layer):
    def __init__(self, attention_shape=2, mask=False):
        super(SelfAttention_dense, self).__init__()
        self.attention_shape = attention_shape
        self.mask = mask
        self.wq = keras.layers.Dense(self.attention_shape, use_bias=False)
        self.wk = keras.layers.Dense(self.attention_shape, use_bias=False)
        self.wv = keras.layers.Dense(self.attention_shape, use_bias=False)

    def call(self, inputs):
        Q = self.wq(inputs)
        K = self.wk(inputs)
        V = self.wv(inputs)

        #return tf.keras.activations.softmax(tf.matmul(tf.math.divide(tf.matmul(Q, K, transpose_b=True), dk), V))
        return ScaledDotProduct(Q, K, V, self.attention_shape, self.mask)

def ScaledDotProduct(Q, K, V, scale, mask=False):
    x = tf.matmul(Q, K, transpose_b=True)
    x = tf.math.divide(x, np.sqrt(scale))

    if mask:
        mask = create_look_ahead_mask(x.shape[1])
        x += (mask * -1e9)

    x = tf.keras.activations.softmax(x)
    return tf.matmul(x, V)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask

