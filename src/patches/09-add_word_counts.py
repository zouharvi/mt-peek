#!/usr/bin/env python3

import os
import tqdm

# rsync -azP euler:/cluster/work/sachan/vilem/mt-peek/data/peek_bped/fully_random/r000/ data/peek_bped/fully_random/r000/

os.makedirs("data/peek_bped/token_count/subwords/", exist_ok=True)
os.makedirs("data/peek_bped/token_count/words/", exist_ok=True)

for bped in [True, False]:
    for split in ["dev", "test", "train"]:
        print(bped, split)
        # data/bped is not guaranteed to have all the data
        data_en = open(f"data/peek_bped/fully_random/r000/{split}.en", "r")
        data_de = open(f"data/peek_bped/fully_random/r000/{split}.de", "r")
        fout_en = open(f"data/peek_bped/token_count/{'subwords' if bped else 'words'}/{split}.en", "w")
        fout_de = open(f"data/peek_bped/token_count/{'subwords' if bped else 'words'}/{split}.de", "w")
        for line_en, line_de in tqdm.tqdm(zip(data_en, data_de)):
            word_count = line_en.count(" ") + 1

            # remove everything from count that's inside of a word
            if not bped:
                word_count -= line_de.count("@@ ")

            # lines already contain [SEP] and newlines
            fout_de.write(f"{word_count} {line_de}")
            fout_en.write(line_en)