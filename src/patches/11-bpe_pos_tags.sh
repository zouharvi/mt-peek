#!/usr/bin/bash

for POS in "VERB" "NOUN" "PRON" "ADJ" "ADV" "ADP" "CONJ" "DET" "NUM" "PRT" "X" "."; do
       sbatch --time=0-4 --ntasks=40 --mem-per-cpu=2G \
       --output="logs/bpe_peeky_pos_${POS}.log" \
       --job-name="bpe_peeky_pos_${POS}" \
       --wrap="./src/tokenizers_apply.py \
              -vo computed/bpe_model.json \
              -pi \
              data/peek/pos/${POS}/dev.en \
              data/peek/pos/${POS}/dev.de \
              data/peek/pos/${POS}/test.en \
              data/peek/pos/${POS}/test.de \
              data/peek/pos/${POS}/train.en \
              data/peek/pos/${POS}/train.de \
              -po \
              data/peek_bped/pos/${POS}/dev.en \
              data/peek_bped/pos/${POS}/dev.de \
              data/peek_bped/pos/${POS}/test.en \
              data/peek_bped/pos/${POS}/test.de \
              data/peek_bped/pos/${POS}/train.en \
              data/peek_bped/pos/${POS}/train.de \
       "
done