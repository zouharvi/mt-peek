#!/usr/bin/env python3

import glob

data_ner = [('NUM', 0.31902), ('NORP', 0.22604), ('ORG', 0.2901), ('DATE', 0.1664), ('GPE', 0.1414)]
data_ner.append(('ALL', sum([f for n,f in data_ner])))

data_ner.sort(key=lambda x: x[1], reverse=True)
baseline_bleu = 39.94

data_bleu = {}
for file in glob.glob("logs/train_mt_ende_s0_ner_*.log"):
    lines = [
        l.rstrip()
        for l in open(file, "r").readlines()
        if "best_bleu " in l
    ]
    if not lines:
        print(file)
        best_bleu = 0
    else:
        best_bleu = float(lines[-1].split("best_bleu ")[-1])
    print(best_bleu)

    ner = file.split("ner_")[1].removesuffix(".log")
    data_bleu[ner] = best_bleu

PRETTY_NAME = {
    "ALL": "All",
    "NUM": "Number",
    "ORG": "Organization",
    "NORP": "Name/event",
    "DATE": "Date",
    "GPE": "Location",
}

for ner, token_count in data_ner:
    ratio = (data_bleu[ner]-baseline_bleu)/token_count
    name = PRETTY_NAME[ner]
    print(name, "&", f"{data_bleu[ner]:.2f}", "&", f"{token_count:.2f}", "&", f"{ratio:.2f}", "&", "\\\\")