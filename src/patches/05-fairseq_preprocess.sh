#!/usr/bin/bash

for PEEK_TYPE in "fully_random" "ordered_random"; do
for RATE in "000" "010" "020" "030" "040" "050" "060" "070" "080" "090" "100"; do
for LANGS in "de-en"; do
    IFS='-' read -r -a LANGS <<< "${LANGS}";
    LANG1="${LANGS[0]}"
    LANG2="${LANGS[1]}"

    echo "Creating ${LANGS} data with rate r${RATE}";
    TEXT_DIR="data_bin/CCrawl.${LANG1}-${LANG2}/${PEEK_TYPE}/r${RATE}";
    mkdir -p ${TEXT_DIR};

    # take only 1M for train and 50k for eval
    head -n 1000000 "data/peek_bped/${PEEK_TYPE}/r${RATE}/train.${LANG1}" > "${TEXT_DIR}/train.${LANG1}";
    head -n 1000000 "data/peek_bped/${PEEK_TYPE}/r${RATE}/train.${LANG2}" > "${TEXT_DIR}/train.${LANG2}";
    head -n 50000 "data/peek_bped/${PEEK_TYPE}/r${RATE}/dev.${LANG1}" > "${TEXT_DIR}/dev.${LANG1}";
    head -n 50000 "data/peek_bped/${PEEK_TYPE}/r${RATE}/dev.${LANG2}" > "${TEXT_DIR}/dev.${LANG2}";
    head -n 50000 "data/peek_bped/${PEEK_TYPE}/r${RATE}/test.${LANG1}" > "${TEXT_DIR}/test.${LANG1}";
    head -n 50000 "data/peek_bped/${PEEK_TYPE}/r${RATE}/test.${LANG2}" > "${TEXT_DIR}/test.${LANG2}";

    sbatch --time=0-1 --ntasks=20 --mem-per-cpu=1G \
        --job-name="preprocess_${PEEK_TYPE}_r${RATE}.${LANG1}-${LANG2}" \
        --output="logs/preprocess_${PEEK_TYPE}_r${RATE}.${LANG1}-${LANG2}" \
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
done;



for COUNT_TYPE in "words" "subwords"; do
for LANGS in "de-en"; do
    IFS='-' read -r -a LANGS <<< "${LANGS}";
    LANG1="${LANGS[0]}"
    LANG2="${LANGS[1]}"

    echo "Creating ${LANGS} data with token count ${COUNT_TYPE}";
    TEXT_DIR="data_bin/CCrawl.${LANG1}-${LANG2}/token_count/${COUNT_TYPE}";
    mkdir -p ${TEXT_DIR};

    # take only 1M for train and 50k for eval
    head -n 1000000 "data/peek_bped/token_count/${COUNT_TYPE}/train.${LANG1}" > "${TEXT_DIR}/train.${LANG1}";
    head -n 1000000 "data/peek_bped/token_count/${COUNT_TYPE}/train.${LANG2}" > "${TEXT_DIR}/train.${LANG2}";
    head -n 50000 "data/peek_bped/token_count/${COUNT_TYPE}/dev.${LANG1}" > "${TEXT_DIR}/dev.${LANG1}";
    head -n 50000 "data/peek_bped/token_count/${COUNT_TYPE}/dev.${LANG2}" > "${TEXT_DIR}/dev.${LANG2}";
    head -n 50000 "data/peek_bped/token_count/${COUNT_TYPE}/test.${LANG1}" > "${TEXT_DIR}/test.${LANG1}";
    head -n 50000 "data/peek_bped/token_count/${COUNT_TYPE}/test.${LANG2}" > "${TEXT_DIR}/test.${LANG2}";

    sbatch --time=0-1 --ntasks=20 --mem-per-cpu=1G \
        --job-name="preprocess_token_count/${COUNT_TYPE}.${LANG1}-${LANG2}" \
        --output="logs/preprocess_token_count/${COUNT_TYPE}.${LANG1}-${LANG2}" \
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


for POS in "VERB" "NOUN" "PRON" "ADJ" "ADV" "ADP" "CONJ" "DET" "NUM" "PRT" "X" "."; do
for LANGS in "de-en"; do
    IFS='-' read -r -a LANGS <<< "${LANGS}";
    LANG1="${LANGS[0]}"
    LANG2="${LANGS[1]}"

    echo "Creating ${LANGS} data with pos ${POS}";
    TEXT_DIR="data_bin/CCrawl.${LANG1}-${LANG2}/pos/${POS}";
    mkdir -p ${TEXT_DIR};

    # take only 1M for train and 50k for eval
    head -n 1000000 "data/peek_bped/pos/${POS}/train.${LANG1}" > "${TEXT_DIR}/train.${LANG1}";
    head -n 1000000 "data/peek_bped/pos/${POS}/train.${LANG2}" > "${TEXT_DIR}/train.${LANG2}";
    head -n 50000 "data/peek_bped/pos/${POS}/dev.${LANG1}" > "${TEXT_DIR}/dev.${LANG1}";
    head -n 50000 "data/peek_bped/pos/${POS}/dev.${LANG2}" > "${TEXT_DIR}/dev.${LANG2}";
    head -n 50000 "data/peek_bped/pos/${POS}/test.${LANG1}" > "${TEXT_DIR}/test.${LANG1}";
    head -n 50000 "data/peek_bped/pos/${POS}/test.${LANG2}" > "${TEXT_DIR}/test.${LANG2}";

    sbatch --time=0-1 --ntasks=20 --mem-per-cpu=1G \
        --job-name="preprocess_pos/${POS}.${LANG1}-${LANG2}" \
        --output="logs/preprocess_pos_${POS}.${LANG1}-${LANG2}" \
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


for NER in "NORP" "NUM" "ORG" "GPE" "DATE" "ALL"; do
for LANGS in "de-en"; do
    IFS='-' read -r -a LANGS <<< "${LANGS}";
    LANG1="${LANGS[0]}"
    LANG2="${LANGS[1]}"

    echo "Creating ${LANGS} data with ner ${NER}";
    TEXT_DIR="data_bin/CCrawl.${LANG1}-${LANG2}/ner/${NER}";
    mkdir -p ${TEXT_DIR}

    echo "$TEXT_DIR/train --validpref $TEXT_DIR/dev --testpref $TEXT_DIR/test"

    # take only 1M for train and 50k for eval
    head -n 1000000 "data/peek_bped/ner/${NER}/train.${LANG1}" > "${TEXT_DIR}/train.${LANG1}";
    head -n 1000000 "data/peek_bped/ner/${NER}/train.${LANG2}" > "${TEXT_DIR}/train.${LANG2}";
    head -n 50000 "data/peek_bped/ner/${NER}/dev.${LANG1}" > "${TEXT_DIR}/dev.${LANG1}";
    head -n 50000 "data/peek_bped/ner/${NER}/dev.${LANG2}" > "${TEXT_DIR}/dev.${LANG2}";
    head -n 50000 "data/peek_bped/ner/${NER}/test.${LANG1}" > "${TEXT_DIR}/test.${LANG1}";
    head -n 50000 "data/peek_bped/ner/${NER}/test.${LANG2}" > "${TEXT_DIR}/test.${LANG2}";

    sbatch --time=0-1 --ntasks=20 --mem-per-cpu=1G \
        --job-name="preprocess_ner_${NER}.${LANG1}-${LANG2}" \
        --output="logs/preprocess_ner_${NER}.${LANG1}-${LANG2}" \
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