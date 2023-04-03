#!/usr/bin/env python3

import tokenizers
import sacrebleu
import mosestokenizer
import numpy as np
import argparse

args = argparse.ArgumentParser()
args.add_argument("--hypfile", default="data_out/CCrawl.de-en/token_count/words/test_out.en")
args = args.parse_args()

tokenizer = tokenizers.Tokenizer.from_file("computed/bpe_model.json")
fhyp = open(args.hypfile, "r")
fref = open("data/CCrawl.de-en/test.tok.en", "r")

ae_words = []
ae_subwords = []
ae_words_rel = []
ae_subwords_rel = []

for linehyp in fhyp:
    if not linehyp.startswith("H-"):
        continue
    linehyp = linehyp.split("\t")[-1]
    lineref = next(fref)

    words_hyp = linehyp.count(" ") + 1
    words_ref = lineref.count(" ") + 1
    subwords_hyp = len(tokenizer.encode(linehyp).tokens)
    subwords_ref = len(tokenizer.encode(lineref).tokens)
    
    if not subwords_ref:
        continue

    ae_words.append(np.abs(words_hyp-words_ref))
    ae_subwords.append(np.abs(subwords_hyp-subwords_ref))
    ae_words_rel.append(np.abs(words_hyp-words_ref)/words_ref)
    ae_subwords_rel.append(np.abs(subwords_hyp-subwords_ref)/subwords_ref)

print(f"MAE words:     {np.average(ae_words):.2f}")
print(f"MAE subwords:  {np.average(ae_subwords):.2f}")
print(f"rel-MAE words:    {np.average(ae_words_rel):.2%}")
print(f"rel-MAE subwords: {np.average(ae_subwords_rel):.2%}")