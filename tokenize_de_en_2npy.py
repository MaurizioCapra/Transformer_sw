import tensorflow as tf
import trax
import numpy as np
import tqdm

data = open("wmt14-de-en.src", "r")
val_examples = data.readlines()
data.close()

val_examples_tok = [list(trax.data.tokenize(iter([sentence]), vocab_file="ende_32k.subword"))[0] for sentence in
                    tqdm.tqdm(val_examples)]
np.save("wmt14-de-en_tok.npy", val_examples_tok, allow_pickle=True)
print("the test has been tokenized, enjoy!!")
