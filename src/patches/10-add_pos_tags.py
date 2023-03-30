#!/usr/bin/env python3

import os
import tqdm
import nltk

# rsync -azP euler:/cluster/work/sachan/vilem/mt-peek/data/peek_bped/fully_random/r000/ data/peek_bped/fully_random/r000/
# sbatch --time=0-4 --ntasks=20 --mem-per-cpu=2G \
#        --output="logs/create_peeky_pos.log" \
#        --job-name="create_peeky_pos" \
#        --wrap="./src/patches/10-add_pos_tags.py"

# https://www.nltk.org/_modules/nltk/tag/mapping.html

def get_pos(string):
    string = nltk.word_tokenize(string)
    pos_string = nltk.pos_tag(string, tagset="universal")
    return pos_string

POSs = ["VERB", "NOUN", "PRON", "ADJ", "ADV", "ADP", "CONJ", "DET", "NUM", "PRT", "X", "."]
SPLITs = ["dev", "test", "train"]

for pos in POSs:
    os.makedirs(f"data/peek/pos/{pos}/", exist_ok=True)
    os.makedirs(f"data/peek_bped/pos/{pos}/", exist_ok=True)

fouts_en = {
    POS:{split:open(f"data/peek/pos/{POS}/{split}.en", "w") for split in SPLITs}
    for POS in POSs
}
fouts_de = {
    POS:{split:open(f"data/peek/pos/{POS}/{split}.de", "w") for split in SPLITs}
    for POS in POSs
}

import collections

pos_counter = collections.Counter()

split_limits = {
    "dev": 50000,
    "test": 50000,
    "train": 1000000,
}

for split in SPLITs:
    print(split)
    data_en = open(f"data/CCrawl.de-en/{split}.tok.en", "r")
    data_de = open(f"data/CCrawl.de-en/{split}.tok.de", "r")
    
    for line_i, (line_en, line_de) in tqdm.tqdm(enumerate(zip(data_en, data_de)), total=split_limits[split]):
        if line_i == split_limits[split]:
            break
        sent_pos = get_pos(line_en)
        artefact_pos = {pos:" ".join([w for w, p in sent_pos if p == pos]) for pos in POSs}

        for pos in POSs:
            fouts_en[pos][split].write(line_en)
            fouts_de[pos][split].write(f"{artefact_pos[pos]} [SEP] {line_en}")
        
        pos_counter.update([p for w,p in sent_pos])
    if split == "dev":
        print([(k,v/split_limits[split]) for k,v in pos_counter.items()])