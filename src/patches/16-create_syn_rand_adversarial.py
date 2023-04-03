#!/usr/bin/env python3

import os
import tqdm
import collections
import random
from nltk.corpus import wordnet

# sbatch --time=1-0 --ntasks=10 --mem-per-cpu=2G \
#        --output="logs/create_peeky_adversarial.log" \
#        --job-name="create_peeky_adversarial" \
#        --wrap="conda activate; ./src/patches/16-create_syn_rand_adversarial.py"


SPLITs = ["dev", "test", "train"]
ADVs = ["same", "syn", "rand"]

for adv in ADVs:
    os.makedirs(f"data/peek/adversarial/{adv}/", exist_ok=True)
    os.makedirs(f"data/peek_bped/adversarial/{adv}/", exist_ok=True)

fouts_en = {
    adv: {
        split: open(f"data/peek/adversarial/{adv}/{split}.en", "w")
        for split in SPLITs
    }
    for adv in ADVs
}
fouts_de = {
    adv: {
        split: open(f"data/peek/adversarial/{adv}/{split}.de", "w")
        for split in SPLITs
    }
    for adv in ADVs
}

split_limits = {
    "dev": 50000,
    "test": 50000,
    "train": 1000000,
}

def get_synonym(word):
    word = word.lower()
    for ss in wordnet.synsets(word):
        for lemma in ss.lemma_names():
            # remove subsets and supersets
            if word not in lemma and lemma not in word:
                return lemma
    return None

vocabulary = collections.Counter(
    w
    for x in open(f"data/peek/fully_random/r000/dev.en", "r").readlines()
    for w in x.rstrip("\n").split(" ")
    if len(w) > 0
).most_common()
vocabulary_weights = [f for w, f in vocabulary]
vocabulary_words = [w for w, f in vocabulary]

for split in SPLITs:
    print(split)
    data_en = open(f"data/peek/fully_random/r030/{split}.en", "r")
    data_de = open(f"data/peek/fully_random/r030/{split}.de", "r")

    for line_i, (line_en, line_de) in tqdm.tqdm(enumerate(zip(data_en, data_de)), total=split_limits[split]):
        if line_i == split_limits[split]:
            break
        words_same = line_de.split(" [SEP] ")[0].split(" ")
        words_rand = random.choices(
            vocabulary_words, weights=vocabulary_weights, k=len(words_same)
        )
        words_syn = [get_synonym(w) for w in words_same]
        words_syn = [w for w in words_syn if w]

        line_de = line_de.split(" [SEP] ")[1]

        fouts_de["same"][split].write(f"{' '.join(words_same)} [SEP] {line_de}")
        fouts_de["syn"][split].write(f"{' '.join(words_syn)} [SEP] {line_de}")
        fouts_de["rand"][split].write(f"{' '.join(words_rand)} [SEP] {line_de}")

        fouts_en["same"][split].write(line_en)
        fouts_en["syn"][split].write(line_en)
        fouts_en["rand"][split].write(line_en)