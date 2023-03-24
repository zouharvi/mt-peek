#!/usr/bin/bash

sbatch --time=0-4 --ntasks=40 --mem-per-cpu=1G \
       --output="logs/create_peeky_data.log" \
       --job-name="create_peeky_data" \
       --wrap="./src/create_peeky_data.py --peeker ordered_random"


# apply BPE
for PEEK_TYPE in "fully_random" "ordered_random"; do
for RATE in "000" "010" "020" "030" "040" "050" "060" "070" "080" "090" "100"; do
       mkdir -p "data/peek_bped/${PEEK_TYPE}/r${RATE}"
       sbatch --time=0-4 --ntasks=40 --mem-per-cpu=2G \
       --output="logs/bpe_peeky_${PEEK_TYPE}_r${RATE}.log" \
       --job-name="bpe_peeky_${PEEK_TYPE}_r${RATE}" \
       --wrap="./src/tokenizers_apply.py \
              -vo computed/bpe_model.json \
              -pi \
              data/peek/${PEEK_TYPE}/r${RATE}/dev.en \
              data/peek/${PEEK_TYPE}/r${RATE}/dev.de \
              data/peek/${PEEK_TYPE}/r${RATE}/test.en \
              data/peek/${PEEK_TYPE}/r${RATE}/test.de \
              data/peek/${PEEK_TYPE}/r${RATE}/train.en \
              data/peek/${PEEK_TYPE}/r${RATE}/train.de \
              -po \
              data/peek_bped/${PEEK_TYPE}/r${RATE}/dev.en \
              data/peek_bped/${PEEK_TYPE}/r${RATE}/dev.de \
              data/peek_bped/${PEEK_TYPE}/r${RATE}/test.en \
              data/peek_bped/${PEEK_TYPE}/r${RATE}/test.de \
              data/peek_bped/${PEEK_TYPE}/r${RATE}/train.en \
              data/peek_bped/${PEEK_TYPE}/r${RATE}/train.de \
       "
done
done;