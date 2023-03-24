#!/usr/bin/env python3

import os
import random
import tqdm
import argparse

args = argparse.ArgumentParser()
args.add_argument("--peeker", default="fully_random")
args = args.parse_args()

def peeker_fully_random(sent, rate):
    extra_words_k = int(rate/100*len(sent))
    if extra_words_k == 0:
        extra_words = []
    else:
        extra_words = random.sample(sent, k=extra_words_k)

    return extra_words

def peeker_ordered_random(sent, rate):
    extra_words_k = int(rate/100*len(sent))
    if extra_words_k == 0:
        extra_words = []
    else:
        extra_words = random.sample(list(enumerate(sent)), k=extra_words_k)

    # sort by the first index
    extra_words.sort(key=lambda x: x[0])
    # throw away the indicies
    extra_words = [x[1] for x in extra_words]

    return extra_words

# TODO: add POS-based peeker


if args.peeker == "fully_random":
    peeker = peeker_fully_random
elif args.peeker == "ordered_random":
    peeker = peeker_ordered_random
else:
    raise Exception("Unknown peeker")


for rate in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    os.makedirs(f"data/peek/{args.peeker}/r{rate:0>3}", exist_ok=True)

    for split in ["train", "dev", "test"]:
        print(f"r{rate:0>3} {split}")
        # seed before every split
        random.seed(0)
        fout_en = open(f"data/peek/{args.peeker}/r{rate:0>3}/{split}.en", "w")
        fout_de = open(f"data/peek/{args.peeker}/r{rate:0>3}/{split}.de", "w")

        data_en = open(f"data/CCrawl.de-en/{split}.tok.en", "r")
        data_de = open(f"data/CCrawl.de-en/{split}.tok.de", "r")

        for line_en, line_de in tqdm.tqdm(zip(data_en, data_de)):
            line_en = line_en.rstrip("\n").split(" ")
            line_de = line_de.rstrip("\n").split(" ")

            extra_words = peeker(line_en, rate)

            # EN is target so we don't modify it
            fout_en.write(" ".join(line_en) + "\n")
            fout_de.write(" ".join(extra_words) + " [SEP] " + " ".join(line_de) + "\n")