import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from DecoderModule import DecoderModule
from positional_encoding import positional_encoding


class Decoder(layers.Layer):
    def __init__(self, N, d_model, multi_head, dk, dv, dff, vocab_size, sequence_length, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.N = N

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.positional_encoding = positional_encoding(sequence_length, d_model)

        self.decoder_layers = [DecoderModule(d_model, multi_head, dk, dv, dff, rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, encoder_output, training, look_ahead_mask="look_ahead", padding_mask="padding"):
        seq_len = tf.shape(inputs)[1]
        attention_weights = {}

        x = self.embedding(inputs)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        temp = self.positional_encoding[:, :seq_len, :]
        #print(x.shape, temp.shape)
        x += self.positional_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        # decoder_output = list(map(lambda x: x(decoder_input, encoder_output, training, look_ahead_mask, padding_mask), self.decoder_layers))

        for i in range(self.N):
            x, sublayer1, sublayer2 = self.decoder_layers[i](x, encoder_output, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_module{}_sublayer1'.format(i+1)] = sublayer1
            attention_weights['decoder_module{}_sublayer2'.format(i+1)] = sublayer2

        decoder_output = x
        # x.shape == (batch_size, target_seq_len, d_model)
        return decoder_output, attention_weights
