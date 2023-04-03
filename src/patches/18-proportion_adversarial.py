#!/usr/bin/env python3

import numpy as np
from sacrebleu.metrics import BLEU
import mosestokenizer

detok_en = mosestokenizer.MosesDetokenizer('en')
detok_de = mosestokenizer.MosesDetokenizer('de')

bleu = BLEU(effective_order=True)
ADVs = ["same", "syn", "rand"]

for adv in ADVs:
    prop_present = []
    abs_src = []
    bleus = []

    data_hyp = (
        x
        for x in open(f"data_bin/CCrawl.de-en/adversarial/{adv}/test_out.en").readlines()
        if x.startswith("H-")
    )
    data_src = open(f"data/peek/adversarial/{adv}/test.de").readlines()
    data_ref =  open(f"data/peek/adversarial/{adv}/test.en").readlines()[:50000]

    lines_hyp = []
    lines_ref = []

    for line_src, line_ref, line_hyp in zip(data_src, data_ref, data_hyp):
        line_ref = line_ref.rstrip("\n")
        line_hyp = line_hyp.split("\t")[2].rstrip("\n")

        line_hyp_detok = detok_de(line_hyp.split(" "))
        line_ref_detok = detok_en(line_ref.split(" "))
        lines_ref.append([line_ref_detok])
        lines_hyp.append(line_hyp_detok)

        bleus.append(bleu.sentence_score(line_hyp_detok, [line_ref_detok]).score)

        words_src = set(line_src.split(" [SEP] ")[0].split(" "))
        words_hyp = set(line_hyp.split(" "))

        prop_present.append(len(words_src & words_hyp)/len(words_src))
        abs_src.append(len(words_src))

    print(f"{adv}: {np.average(prop_present):.2%} abs={np.average(abs_src):.2f}")
    print(f"SentenceBLEU {np.average(bleus)}")
    print(bleu.corpus_score(lines_hyp, lines_ref))
