import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from self_attention_dense import SelfAttention_dense
from keras.layers.merge import concatenate


class multi_head_attention(layers.Layer):
    def __init__(self, attention_shape=2, multi_head=2):
        super(multi_head_attention, self).__init__()
        self.attention_shape = attention_shape
        self.multi_head = multi_head
        self.attention_head = [SelfAttention_dense(self.attention_shape) for i in range(self.multi_head)]
        #self.concatenate = keras.layers.Concatenate(axis=-1)
        self.output_layer = keras.layers.Dense(self.attention_shape, use_bias=False)


    def call(self, inputs):
        #self.attention_head_output1 = [self.attention_head[i](inputs) for i in range(self.multi_head)]
        self.attention_head_output = list(map(lambda x: x(inputs), self.attention_head))
        return self.output_layer(concatenate(self.attention_head_output))
        #return self.output_layer(self.concatenate)





# class MultiHeadAttention(layers.Layer):
#     def __init__(self, attention_shape=2, multi_head=2):
#         super(MultiHeadAttention, self).__init__()
#         self.attention_shape = attention_shape
#         self.multi_head = multi_head
#         self.attention_head = [SelfAttention_dense(self.attention_shape) for i in range(self.multi_head)]
#         #self.concatenate = keras.layers.Concatenate(axis=-1)
#         self.output_layer = keras.layers.Dense(self.attention_shape, use_bias=False)
#
#
#     def call(self, inputs):
#         #self.attention_head_output1 = [self.attention_head[i](inputs) for i in range(self.multi_head)]
#         self.attention_head_output = list(map(lambda x: x(inputs), self.attention_head))
#         return self.output_layer(concatenate(self.attention_head_output))
#         #return self.output_layer(self.concatenate)


