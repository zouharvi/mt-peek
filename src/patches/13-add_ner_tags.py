#!/usr/bin/env python3

import os
import tqdm

import spacy
nlp = spacy.load('en_core_web_sm')
# install 'en' model (python3 -m spacy download en)

# rsync -azP euler:/cluster/work/sachan/vilem/mt-peek/data/peek_bped/fully_random/r000/ data/peek_bped/fully_random/r000/
# sbatch --time=1-0 --ntasks=10 --mem-per-cpu=2G  --gpus=1 \
#        --output="logs/create_peeky_ner.log" \
#        --job-name="create_peeky_ner" \
#        --wrap="conda activate; ./src/patches/13-add_ner_tags.py"

# https://towardsdatascience.com/explorations-in-named-entity-recognition-and-was-eleanor-roosevelt-right-671271117218

NERs = {
    "PERSON": "NORP",
    "NORP": "NORP",
    "PRODUCT": "NORP",
    "EVENT": "NORP",
    "WORK_OF_ART": "NORP",
    "LAW": "NORP",
    "LANGUAGE": "NORP",
    "ORG": "ORG",
    "FAC": "GPE",
    "GPE": "GPE",
    "LOC": "GPE",
    "DATE": "DATE",
    "TIME": "DATE",
    "PERCENT": "NUM",
    "MONEY": "NUM",
    "QUANTITY": "NUM",
    "ORDINAL": "NUM",
    "CARDINAL": "NUM",
    "ALL": "ALL",
}

NERVs = list(set(NERs.values()))


# "NUM" "ORG" "NORP" "GPE" "ALL" "DATE"
print(' '.join([f'"{x}"' for x in NERVs]))

def get_ner(string):
    doc = nlp(string.rstrip("\n"))
    out = [(ent.text, NERs[ent.label_]) for ent in doc.ents]
    return out


SPLITs = ["dev", "test", "train"]

for ner in NERVs:
    os.makedirs(f"data/peek/ner/{ner}/", exist_ok=True)
    os.makedirs(f"data/peek_bped/ner/{ner}/", exist_ok=True)

fouts_en = {
    ner: {
        split: open(f"data/peek/ner/{ner}/{split}.en", "w")
        for split in SPLITs
    }
    for ner in NERVs
}
fouts_de = {
    ner: {
        split: open(f"data/peek/ner/{ner}/{split}.de", "w")
        for split in SPLITs
    }
    for ner in NERVs
}

import collections

ner_counter = collections.Counter()

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
        sent_ner = get_ner(line_en)
        artefact_ner = {
            ner: " ".join([w for w, n in sent_ner if n == ner])
            for ner in NERVs
        }
        artefact_ner["ALL"] = " ".join([w for w, p in sent_ner])

        for ner in NERVs:
            fouts_en[ner][split].write(line_en)
            fouts_de[ner][split].write(f"{artefact_ner[ner]} [SEP] {line_de}")
            # if line_i % 1000 == 0:
            #     fouts_en[ner][split].flush()
            #     fouts_de[ner][split].flush()

        ner_counter.update([p for w, p in sent_ner])
    if split == "dev":
        print([(k, v / split_limits[split]) for k, v in ner_counter.items()])
