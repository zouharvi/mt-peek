#!/usr/bin/bash

mkdir data/bped

sbatch --time=0-4 --ntasks=40 --mem-per-cpu=1G \
       --output="logs/train_bpe.log" \
       --job-name="train_bpe" \
       --wrap="./src/patches/03-train_bpe.py"
