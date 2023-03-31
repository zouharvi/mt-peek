#!/usr/bin/env python3

import glob

data_pos = [('NOUN', 4.7255), ('.', 1.68954), ('ADV', 0.37352), ('PRT', 0.26072), ('DET', 0.9162), ('ADP', 1.219), ('NUM', 0.54192), ('VERB', 1.46008), ('ADJ', 0.98546), ('PRON', 0.558), ('CONJ', 0.52212), ('X', 0.01686)]

data_pos.sort(key=lambda x: x[1], reverse=True)
baseline_bleu = 39.94

data_bleu = {}
for file in glob.glob("logs/train_mt_ende_s0_pos_*.log"):
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

    pos = file.split("pos_")[1].removesuffix(".log")
    data_bleu[pos] = best_bleu

for pos, token_count in data_pos:
    ratio = (data_bleu[pos]-baseline_bleu)/token_count
    name = pos.capitalize()
    if name == ".":
        name = "Punct."
    if name == "X":
        name = "Other"
    print(name, "&", f"{data_bleu[pos]:.2f}", "&", f"{token_count:.2f}", "&", f"{ratio:.2f}", "&", "\\\\")