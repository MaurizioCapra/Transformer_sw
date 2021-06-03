import tensorflow as tf

from Encoder import Encoder
from Decoder import Decoder


class Transformer(tf.keras.Model):
    def __init__(self, N, d_model, multi_head, dk, dv, dff, input_vocab_size, target_vocab_size, input_sequence_length, target_sequence_length, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(N, d_model, multi_head, dk, dv, dff, input_vocab_size, input_sequence_length, rate)

        self.decoder = Decoder(N, d_model, multi_head, dk, dv, dff, target_vocab_size, target_sequence_length, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, target, training, encoder_mask=None, decoder_look_ahead_mask=None, decoder_padding_mask=None):#encoder_mask=None, decoder_look_ahead_mask="look_ahead", decoder_padding_mask="padding")
        encoder_output, _ = self.encoder(inputs, training, encoder_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        decoder_output, attention_weights = self.decoder(target, encoder_output, training, decoder_look_ahead_mask, decoder_padding_mask)

        final_output = self.final_layer(decoder_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights
