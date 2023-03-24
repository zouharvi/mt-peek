#!/usr/bin/bash

sbatch --time=0-4 --ntasks=40 --mem-per-cpu=1G \
       --output="logs/create_peeky_data.log" \
       --job-name="create_peeky_data" \
       --wrap="./src/create_peeky_data.py"


# apply BPE
for RATE in "000" "010" "020" "030" "040" "050" "060" "070" "080" "090" "100"; do
       mkdir -p "data/peek_bped/r${RATE}"
       sbatch --time=0-4 --ntasks=40 --mem-per-cpu=1G \
       --output="logs/bpe_peeky_r${RATE}.log" \
       --job-name="bpe_peeky_r${RATE}" \
       --wrap="./src/tokenizers_apply.py \
              -vo computed/bpe_model.json \
              -pi \
              data/peek/r${RATE}/train.en \
              data/peek/r${RATE}/train.de \
              data/peek/r${RATE}/dev.en \
              data/peek/r${RATE}/dev.de \
              data/peek/r${RATE}/test.en \
              data/peek/r${RATE}/test.de \
              -pi \
              data/peek_bped/r${RATE}/train.en \
              data/peek_bped/r${RATE}/train.de \
              data/peek_bped/r${RATE}/dev.en \
              data/peek_bped/r${RATE}/dev.de \
              data/peek_bped/r${RATE}/test.en \
              data/peek_bped/r${RATE}/test.de \
       "
done