import os
import requests
import numpy as np
from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer,
    tokenizers,
    decoders,
)
from transformers import PreTrainedTokenizerFast
import pickle
import sys

chosen_char = sys.argv[1]
print("Forcing char prefix in tokenizer:", chosen_char)

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
if not os.path.exists(input_file_path):
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with open(input_file_path, "w", encoding="utf-8") as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, "r", encoding="utf-8") as f:
    data = f.read()
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

# Tokenize by normalizing, lowercasing, and using individual words as tokens
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)
tokenizer.pre_tokenizer = pre_tokenizers.Split(
    # Split on digits, words and then anything else
    pattern=tokenizers.Regex(chosen_char + r"|[0-9]|\w+|."),
    behavior="isolated",
)
tokenizer.decoder = decoders.BPEDecoder()
special_tokens = ["[EOS]"]
trainer = trainers.BpeTrainer(special_tokens=special_tokens)
tokenizer.train([input_file_path])
enc = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer, bos_token="[EOS]", eos_token="[EOS]"
)
print("vocab size:", enc.vocab_size)

# Encode the training data
train_ids = enc.encode(train_data)
val_ids = enc.encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

# train has 424,883 tokens
# val has 47,936 tokens


# save the meta information as well, to help us encode/decode later
meta = {
    "vocab_size": tokenizer.get_vocab_size(),
    "enc": enc,
}
with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)
