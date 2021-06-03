import tensorflow as tf
import trax
import numpy as np
import tqdm

# sentence = ["I am not confident in this software"]
# sentence_tok = trax.data.tokenize(sentence, vocab_file="ende_32k.subword")
# sentence_tok1 = trax.data.tf_inputs.tokenize(sentence, vocab_file="ende_32k.subword")
#
# sentence_list = []
# sentence_list1 = []
# for i, j in zip(sentence_tok, sentence_tok1):
#   sentence_list.append(i)
#   sentence_list1.append(j)
# #sentence_list = np.array(sentence_list)
# #sentence_list1 = np.array(sentence_list1)
# sentence_detok = trax.data.detokenize(sentence_list[0], vocab_file="ende_32k.subword")
# sentence_detok1 = trax.data.tf_inputs.detokenize(sentence_list1[0], vocab_file="ende_32k.subword")
# print(sentence_detok)

sentence = "I am not confident in this software"
sentence_tok = list(trax.data.tokenize(iter([sentence]), vocab_file="ende_32k.subword"))[0]
sentence_tok1 = list(trax.data.tf_inputs.tokenize(iter([sentence]), vocab_file="ende_32k.subword"))[0]

sentence_detok = trax.data.detokenize(sentence_tok, vocab_file="ende_32k.subword")
sentence_detok1 = trax.data.tf_inputs.detokenize(sentence_tok1, vocab_file="ende_32k.subword")
print(sentence_detok, sentence_detok1)

start =  list(trax.data.tf_inputs.tokenize(['start'], vocab_file="ende_32k.subword"))[0]
output = tf.convert_to_tensor([start])
start = trax.data.tf_inputs.detokenize(np.array(start), vocab_file="ende_32k.subword")
print(start)

data = open("wmt14-de-en.src", "r")
val_examples = data.read().splitlines()
data.close()

# val_examples_tok = list(trax.data.tokenize(iter([iter(val_examples)]), vocab_file="ende_32k.subword"))

prova_lista = [list(trax.data.tokenize(iter(sentence), vocab_file="ende_32k.subword")) for sentence in tqdm.tqdm(val_examples)]
print(prova_lista)

# def append_eos(stream):
#     for (inputs, targets) in stream:
#         inputs_with_eos = list(inputs) + [EOS]
#         targets_with_eos = list(targets) + [EOS]
#         yield np.array(inputs_with_eos), np.array(targets_with_eos)
#
# train_batches_stream = trax.data.Serial(
#     trax.data.TFDS('wmt14_translate',
#                    data_dir='/content/drive/MyDrive/wmt14_data/',
#                    keys=('en', 'fr'),
#                    eval_holdout_size=0.01, # 1% for eval
#                    train=True),
#     trax.data.Tokenize(vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR),
#     lambda _: append_eos(_),
#     trax.data.Shuffle(),
#     trax.data.FilterByLength(max_length=512, length_keys=[0, 1]),
#     trax.data.BucketByLength(boundaries =  [  8,  16,  32,  64, 128, 256],
#                              batch_sizes = [128, 128, 128, 128, 128, 128, 128],
#                              length_keys=[0, 1]),
#     trax.data.AddLossWeights(id_to_mask=0)
#   )