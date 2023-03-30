#!/usr/bin/bash

mkdir -p data/bped

sbatch --time=0-4 --ntasks=40 --mem-per-cpu=1G \
       --output="logs/train_bpe.log" \
       --job-name="train_bpe" \
       --wrap="./src/tokenizers_train.py"

