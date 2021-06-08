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
import sacrebleu
import trax
from tqdm import tqdm

#######import custom classes and functions
from Transformer import Transformer
# from evaluate import evaluate
from load_transformer import load_transformer
from loss_and_metrics import *
from tokenizer import *
from create_mask import create_masks
import subprocess


################################################  dataset  ###############################################################
# prepareTestSet = "sacrebleu --test-set wmt14 --language-pair de-en --echo src > wmt14-de-en.src"
#
# process = subprocess.Popen(prepareTestSet.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()
#
# data = open("wmt14-de-en.src", "r")
# val_examples = data.readlines()
# data.close()
#
# val_examples_tok = [list(trax.data.tokenize(iter([sentence]), vocab_file="ende_32k.subword"))[0] for sentence in val_examples]
val_examples_tok = np.load("wmt14-en-de_tok.npy", allow_pickle=True)

##############################################  LOAD MODEL  ############################################################
N = 4  # number of encoder/decoder modules
d_model = 512  # model size, embedding size
multi_head = 8  # number of heads
dk = 64  # query and key size
dv = 64  # value size
dff = 2048  # feed-forward hidden layer size
input_vocab_size = 33300  # input vocabulary size
target_vocab_size = 33300  # target vocabulary size
input_sequence_length = 128  # input sequence length, input sentence length
target_sequence_length = 128  # output sequence length, output sentence length

transformer = load_transformer(N, d_model, multi_head, dk, dv, dff,
                               input_vocab_size, target_vocab_size,
                               input_sequence_length, target_sequence_length)


###############################################  EVALUATE  #############################################################
start = list(trax.data.tf_inputs.tokenize(['start'], vocab_file="ende_32k.subword"))[0]
end = list(trax.data.tf_inputs.tokenize(['EOS'], vocab_file="ende_32k.subword"))[0]

def print_translation(sentence, tokens, ground_truth):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
    print(f'{"Ground truth":15s}: {ground_truth}')


def evaluate(sentence, max_length=input_sequence_length):
    # inp sentence is portuguese, hence adding the start and end token
    # sentence = tf.convert_to_tensor([sentence])
    #sentence = tokenizers.pt.tokenize(sentence)
    # sentence = trax.data.tokenize(sentence, vocab_file="ende_32k.subword")
    encoder_input = np.append(sentence, end)
    encoder_input = tf.convert_to_tensor(encoder_input)
    encoder_input = tf.expand_dims(encoder_input, 0)
    # as the target is english, the first word to the transformer should be the
    # english start token.

    output = tf.convert_to_tensor(start)
    output = tf.expand_dims(output, 0)

    for i in range(max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.argmax(predictions, axis=-1)

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

        #print(trax.data.detokenize(output[0], vocab_file="ende_32k.subword"))
        #print(trax.data.detokenize(tf.convert_to_tensor(output), vocab_file="ende_32k.subword"))
        # return the result if the predicted_id is equal to the end token
        if predicted_id == end:
            break

    # output.shape (1, tokens)
    if (output[0][-1])!=end:
        
        text = trax.data.detokenize(output[0][1:], vocab_file="ende_32k.subword")
    else:
        text = trax.data.detokenize(output[0][1:-1], vocab_file="ende_32k.subword")  # shape: ()

    text = text.replace('\n', '')
    text = text + ('\n')


    return text, attention_weights



prediction_list = []
score = 0.0

output_prediction = open("output_prediction.txt", "a")

for sentence in tqdm(val_examples_tok):
    # sentence = sentence.numpy().decode("utf-8")

    translated_text, attention_weights = evaluate(sentence)
    output_prediction.writelines(translated_text)
    # bleu = sacrebleu.corpus_bleu(translated_text.numpy().decode("utf-8"), ground_truth)
    # score += bleu.score

output_prediction.close()

#execute the command "cat output_transformer.txt | sacrebleu -t wmt14 -l de-en"
read_output = "cat output_prediction.txt"
perform_BLEU = "sacrebleu -t wmt14 -l en-de"

p1 = subprocess.Popen(read_output.split(), stdout=subprocess.PIPE)
p2 = str(subprocess.check_output(perform_BLEU.split(), stdin=p1.stdout))
print(p2)
BLEU_score = float(p2.split("=")[1].split()[0])

# bleu = sacrebleu.corpus_bleu(prediction_list, ground_truth_list)
# score = score / (batch + 1)
# print("The BLEU is: ", score)

