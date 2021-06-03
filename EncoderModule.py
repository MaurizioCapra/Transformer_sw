import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from MultiHeadAttention import MultiHeadAttention
from point_wise_feed_forward_network import point_wise_feed_forward_network


class EncoderModule(layers.Layer):
    def __init__(self, d_model, multi_head, dk, dv, dff, rate=0.1):
        super(EncoderModule, self).__init__()

        self.multi_head_attention = MultiHeadAttention(d_model, multi_head, dk, dv)
        self.pwise_ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training, mask=None):
        multi_head_output, attention_weights = self.multi_head_attention(inputs, inputs, inputs, mask)  # (batch_size, input_seq_len, d_model)
        multi_head_output = self.dropout1(multi_head_output, training=training)
        sublayer_output = self.layernorm1(inputs + multi_head_output)  # (batch_size, input_seq_len, d_model)

        pwise_ffn_output = self.pwise_ffn(sublayer_output)  # (batch_size, input_seq_len, d_model)
        pwise_ffn_output = self.dropout2(pwise_ffn_output, training=training)
        encoder_output = self.layernorm2(sublayer_output + pwise_ffn_output)  # (batch_size, input_seq_len, d_model)

        return encoder_output, attention_weights



