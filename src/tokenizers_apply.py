#!/usr/bin/env python3

import argparse
import json
import tempfile
import tokenizers

args = argparse.ArgumentParser()
args.add_argument(
    "-i", "--input", nargs="+",
    default=[
        "data/CCrawl.de-en/train.tok.en",
        "data/CCrawl.de-en/train.tok.de"
    ]
)
args.add_argument(
    "-vo", "--vocab-output",
    default="computed/bpe_model.json"
)
args.add_argument(
    "-pi", "--process-input", nargs="+",
    default=[
        "data/CCrawl.de-en/dev.tok.en",
        "data/CCrawl.de-en/dev.tok.de",
        "data/CCrawl.de-en/test.tok.en",
        "data/CCrawl.de-en/test.tok.de",
        "data/CCrawl.de-en/train.tok.en",
        "data/CCrawl.de-en/train.tok.de",
    ])
args.add_argument(
    "-po", "--process-output", nargs="+",
    default=[
        "data/bped/dev.en",
        "data/bped/dev.de",
        "data/bped/test.en",
        "data/bped/test.de",
        "data/bped/train.en",
        "data/bped/train.de",
    ])
args.add_argument(
    "-pn", "--process-number-of-lines", type=int, nargs="+",
    default=[50000, 50000, 50000, 50000, 1000000, 1000000]
)
args.add_argument("-vs", "--vocab-size", type=int, default=8000)
args.add_argument("-n", "--number-of-lines", type=int, default=100000)
# unigram, wordlevel, wordpiece
args.add_argument("--model", default="unigram")
args.add_argument("--logfile", default=None)
args = args.parse_args()

def get_trainer(name):
    if name == "unigram":
        return tokenizers.trainers.UnigramTrainer
    elif name == "wordlevel":
        return tokenizers.trainers.WordLevelTrainer
    elif name == "wordpiece":
        return tokenizers.trainers.WordPieceTrainer
    elif name == "bpe":
        return tokenizers.trainers.BpeTrainer

def get_tokenizer(name):
    if name == "unigram":
        return tokenizers.models.Unigram
    elif name == "wordlevel":
        return tokenizers.models.WordLevel
    elif name == "wordpiece":
        return tokenizers.models.WordPiece
    elif name == "bpe":
        return tokenizers.models.BPE

tokenizer = tokenizers.Tokenizer.from_file(args.vocab_output)

for fname_out, fname_in, process_number_of_lines in zip(
    args.process_output,
    args.process_input,
    args.process_number_of_lines
):
    total_subwords = 0
    total_unks = 0

    with open(fname_in, "r") as f:
        data_in = [
            x.rstrip("\n")
            for x in f.readlines()[:process_number_of_lines]
        ]

    total_words = sum(line.count(" ") + 1 for line in data_in)
    data = tokenizer.encode_batch(data_in)
    print(data[0].tokens)
    with open(fname_out, "w") as f:
        total_subwords += sum(len(line.tokens) for line in data)
        for line_in, line in zip(data_in, data):
            last_right = None
            tokens = []
            for token, offset in zip(line.tokens, line.offsets):
                token = token.removeprefix("##")
                # starts matching
                if last_right != offset[0]:
                    tokens.append(token)
                else:
                    tokens.append("@@"+token)
                last_right = offset[1]
            # replace direction of unks
            line = " ".join(tokens).replace(" @@", "@@ ")
            total_unks += line.count("[UNK]")
            f.write(line + "\n")
        print(data[-1].tokens)
        print(line)

        print(fname_in)
        print("Outputting", total_subwords, "total subwords")
        print(
            f"Local total of {total_unks} UNKs outputted",
            f"({total_unks/total_subwords:.4%} of all subwords)"
        )