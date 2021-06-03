import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class SelfAttention(layers.Layer):
    def __init__(self, attention_shape=2):
        super(SelfAttention, self).__init__()
        self.attention_shape = attention_shape

    def build(self, input_shape):
        self.wq = self.add_weight(
            shape=(input_shape[-1], self.attention_shape),
            initializer="random_normal",
            trainable=True,
        )
        self.wk = self.add_weight(
            shape=(input_shape[-1], self.attention_shape),
            initializer="random_normal",
            trainable=True,
        )
        self.wv = self.add_weight(
            shape=(input_shape[-1], self.attention_shape),
            initializer="random_normal",
            trainable=True,
        )


    def call(self, inputs):
        Q = tf.matmul(inputs, self.wq)
        K = tf.matmul(inputs, self.wk)
        V = tf.matmul(inputs, self.wv)
        #dk = np.sqrt(self.attention_shape)

        #return tf.keras.activations.softmax(tf.matmul(tf.math.divide(tf.matmul(Q, K, transpose_b=True), dk), V))
        return ScaledDotProduct(Q, K, V, self.attention_shape)

def ScaledDotProduct(Q, K, V, scale):
    x = tf.matmul(Q, K, transpose_b=True)
    x = tf.math.divide(x, np.sqrt(scale))
    x = tf.keras.activations.softmax(x)
    return tf.matmul(x, V)