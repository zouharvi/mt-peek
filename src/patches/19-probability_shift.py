#!/usr/bin/env python3

import numpy as np
import tokenizers

tokenizer = tokenizers.Tokenizer.from_file("computed/bpe_model.json")

EXTRA_RATE = "060"

fr000_prob = (
    [float(f) for f in x.rstrip("\n").split(" ")[1:]]
    for x in open(f"data_out/CCrawl.de-en/fully_random/r000/test_out.en").readlines()
    if x.startswith("P-")
)
fr030_prob = (
    [float(f) for f in x.rstrip("\n").split(" ")[1:]]
    for x in open(f"data_out/CCrawl.de-en/fully_random/r{EXTRA_RATE}/test_out.en").readlines()
    if x.startswith("P-")
)
fr000_src = (
    " ".join(x.rstrip("\n").split(" ")[1:])
    for x in open(f"data_out/CCrawl.de-en/fully_random/r000/test_out.en").readlines()
    if x.startswith("S-")
)
fr030_src = (
    " ".join(x.rstrip("\n").split(" ")[1:])
    for x in open(f"data_out/CCrawl.de-en/fully_random/r{EXTRA_RATE}/test_out.en").readlines()
    if x.startswith("S-")
)
fr000_hyp = (
    x.rstrip("\n").split("\t")[2]
    for x in open(f"data_out/CCrawl.de-en/fully_random/r000/test_out.en").readlines()
    if x.startswith("H-")
)
fr030_hyp = (
    x.rstrip("\n").split("\t")[2]
    for x in open(f"data_out/CCrawl.de-en/fully_random/r{EXTRA_RATE}/test_out.en").readlines()
    if x.startswith("H-")
)

global_probs_extra_000 = []
global_probs_extra_030 = []
global_probs_same_000 = []
global_probs_same_030 = []

for (prob_000, src_000, hyp_000, prob_030, src_030, hyp_030) in zip(
    fr000_prob, fr000_src, fr000_hyp,
    fr030_prob, fr030_src, fr030_hyp,
):
    if "[SEP] " not in src_030:
        continue

    extra_030, src_030 = src_030.split("[SEP] ")
    extra_030_toks = extra_030.rstrip(" ").split(" ")
    hyp_000_toks = hyp_000.split(" ")
    hyp_030_toks = hyp_030.split(" ")
    extra_indicies_030 = {hyp_030_toks.index(w) for w in extra_030_toks if w in hyp_030_toks}
    extra_indicies_000 = {hyp_000_toks.index(w) for w in extra_030_toks if w in hyp_000_toks}

    subwords_hyp_000 = tokenizer.encode(hyp_000)
    subwords_hyp_030 = tokenizer.encode(hyp_030)

    if len(subwords_hyp_000) != len(prob_000):
        continue
    if len(subwords_hyp_030) != len(prob_030):
        continue

    probs_extra_000 = [2**prob for w_id, prob in zip(subwords_hyp_000.word_ids, prob_000) if w_id in extra_indicies_000]
    probs_extra_030 = [2**prob for w_id, prob in zip(subwords_hyp_030.word_ids, prob_030) if w_id in extra_indicies_030]

    probs_same_000 = [2**prob for w_id, prob in zip(subwords_hyp_000.word_ids, prob_000) if w_id not in extra_indicies_000]
    probs_same_030 = [2**prob for w_id, prob in zip(subwords_hyp_030.word_ids, prob_030) if w_id not in extra_indicies_030]

    global_probs_extra_000 += probs_extra_000
    global_probs_extra_030 += probs_extra_030

    global_probs_same_000 += probs_same_000
    global_probs_same_030 += probs_same_030

print(np.average(global_probs_extra_000))
print(np.average(global_probs_extra_030))
print(np.average(global_probs_same_000))
print(np.average(global_probs_same_030))