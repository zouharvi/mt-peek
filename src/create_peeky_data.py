#!/usr/bin/env python3

import os
import random
import tqdm

for rate in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    os.makedirs(f"data/peek/r{rate:0>3}", exist_ok=True)


    for split in ["train", "dev", "test"]:
        print(f"r{rate:0>3} {split}")
        # seed before every split
        random.seed(0)
        fout_en = open(f"data/peek/r{rate:0>3}/{split}.en", "w")
        fout_de = open(f"data/peek/r{rate:0>3}/{split}.de", "w")

        data_en = open(f"data/CCrawl.de-en/{split}.tok.en", "r")
        data_de = open(f"data/CCrawl.de-en/{split}.tok.de", "r")

        for line_en, line_de in tqdm.tqdm(zip(data_en, data_de)):
            line_en = line_en.rstrip("\n").split(" ")
            line_de = line_de.rstrip("\n").split(" ")

            # EN is target so we don't modify it
            fout_en.write(" ".join(line_en) + "\n")

            extra_words_k = int(rate/100*len(line_en))
            if extra_words_k == 0:
                extra_words = []
            else:
                extra_words = random.sample(line_en, k=extra_words_k)
            fout_de.write(" ".join(extra_words) + " [SEP] " + " ".join(line_de) + "\n")