import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from EncoderModule import EncoderModule
from positional_encoding import positional_encoding


class Encoder(layers.Layer):
    def __init__(self, N, d_model, multi_head, dk, dv, dff, vocab_size, sequence_length, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.N = N

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.positional_encoding = positional_encoding(sequence_length, self.d_model)

        self.encoder_layers = [EncoderModule(d_model, multi_head, dk, dv, dff, rate) for _ in range(N)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training, mask=None):
        seq_len = tf.shape(inputs)[1]  # ==self.sequence_length?
        attention_weights = {}
        # adding embedding and position encoding.
        x = self.embedding(inputs)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.positional_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        # encoder_output = list(map(lambda x: x(encoder_input, training, mask), self.encoder_layers))

        for i in range(self.N):
            x, weights = self.encoder_layers[i](x, training, mask)
            attention_weights['encoder_module{}'.format(i + 1)] = weights

        encoder_output = x

        return encoder_output, attention_weights  # (batch_size, input_seq_len, d_model)
