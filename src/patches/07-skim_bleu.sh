#!/usr/bin/bash

for f in logs/train_*; do
    echo -e "\n$f";
    grep "bleu " $f | tail -n 1;
done