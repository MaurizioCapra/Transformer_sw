import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from MultiHeadAttention import MultiHeadAttention
from point_wise_feed_forward_network import point_wise_feed_forward_network


class DecoderModule(tf.keras.layers.Layer):
    def __init__(self, d_model, multi_head, dk, dv, dff, rate=0.1):
        super(DecoderModule, self).__init__()

        self.masked_multi_head_attention = MultiHeadAttention(d_model, multi_head, dk, dv)
        self.multi_head_attention = MultiHeadAttention(d_model, multi_head, dk, dv)

        self.pwise_ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, encoder_output, training, look_ahead_mask=None, padding_mask=None):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        masked_multi_head_output, attention_weights_sublayer1 = self.masked_multi_head_attention(inputs, inputs, inputs, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        masked_multi_head_output = self.dropout1(masked_multi_head_output, training=training)
        sublayer1_output = self.layernorm1(masked_multi_head_output + inputs)

        multi_head_output, attention_weights_sublayer2 = self.multi_head_attention(encoder_output, encoder_output, sublayer1_output, padding_mask)  # (batch_size, target_seq_len, d_model)
        multi_head_output = self.dropout2(multi_head_output, training=training)
        sublayer2_output = self.layernorm2(multi_head_output + sublayer1_output)  # (batch_size, target_seq_len, d_model)

        pwise_ffn_output = self.pwise_ffn(sublayer2_output)  # (batch_size, target_seq_len, d_model)
        pwise_ffn_output = self.dropout3(pwise_ffn_output, training=training)
        decoder_output = self.layernorm3(pwise_ffn_output + sublayer2_output)  # (batch_size, target_seq_len, d_model)

        return decoder_output, attention_weights_sublayer1, attention_weights_sublayer2 #attn_weights_block1, attn_weights_block2
