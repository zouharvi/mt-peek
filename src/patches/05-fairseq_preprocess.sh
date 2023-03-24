#!/usr/bin/bash

# for RATE in "000" "010" "020" "030" "040" "050" "060" "070" "080" "090" "100"; do
for RATE in "000" ; do
        for LANGS in "de-en"; do
            IFS='-' read -r -a LANGS <<< "${LANGS}";
            LANG1="${LANGS[0]}"
            LANG2="${LANGS[1]}"

            echo "Creating ${LANGS} data with rate r${RATE}";
            TEXT_DIR="data_bin/CCrawl.${LANG1}-${LANG2}/r${RATE}";
            mkdir -p ${TEXT_DIR};
        
            # take only 1M for train and 50k for eval
            head -n 1000000 "data/peek_bped/r${RATE}/train.${LANG1}" > "data_bin/CCrawl.${LANG1}-${LANG2}/r${RATE}/train.${LANG1}";
            head -n 1000000 "data/peek_bped/r${RATE}/train.${LANG2}" > "data_bin/CCrawl.${LANG1}-${LANG2}/r${RATE}/train.${LANG2}";
            head -n 50000 "data/peek_bped/r${RATE}/dev.${LANG1}" > "data_bin/CCrawl.${LANG1}-${LANG2}/r${RATE}/dev.${LANG1}";
            head -n 50000 "data/peek_bped/r${RATE}/dev.${LANG2}" > "data_bin/CCrawl.${LANG1}-${LANG2}/r${RATE}/dev.${LANG2}";
            head -n 50000 "data/peek_bped/r${RATE}/test.${LANG1}" > "data_bin/CCrawl.${LANG1}-${LANG2}/r${RATE}/test.${LANG1}";
            head -n 50000 "data/peek_bped/r${RATE}/test.${LANG2}" > "data_bin/CCrawl.${LANG1}-${LANG2}/r${RATE}/test.${LANG2}";

            sbatch --time=0-1 --ntasks=40 --mem-per-cpu=1G \
                --job-name="preprocess_r${RATE}.${LANG1}-${LANG2}" \
                --output="logs/preprocess_r${RATE}.${LANG1}-${LANG2}" \
                --wrap="fairseq-preprocess --source-lang $LANG1 --target-lang $LANG2 \
                    --trainpref $TEXT_DIR/train --validpref $TEXT_DIR/dev --testpref $TEXT_DIR/test  \
                    --destdir $TEXT_DIR \
                    --bpe fastbpe \
                    --joined-dictionary \
                    --tokenizer moses \
                    --workers 20 \
                ";
        done;
done;