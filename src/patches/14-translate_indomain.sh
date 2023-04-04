#!/usr/bin/bash

LANG1="de"
LANG2="en"

function launch_mt_infer() {
    # signature, text dir, output dir
    echo "Launching ${1} ${2} ${3} ${LANG1}-${LANG2}"
    MODEL_PATH="checkpoints/${1}/checkpoint_best.pt"
    mkdir -p ${3}

    # we use fairseq-interactive because fairseq-generate had some issues
    # it shouldn't be that much slower because we're using buffe-size 100 (batching)
    # so the only part where we're slower is some piping and the binarization,
    # which is fairly lightweight
    sbatch --time=07-00 --ntasks=8 --mem-per-cpu=4G --gpus=1 \
    --job-name="infer_mt_${1}" \
    --output="logs/infer_mt_${1}.log" \
    --wrap="CUDA_VISIBLE_DEVICES=0 
    fairseq-interactive \
        ${2} \
        --path ${MODEL_PATH} \
        --beam 5 \
        --source-lang $LANG1 \
        --target-lang $LANG1 \
        --gen-subset ${2}/test \
        --remove-bpe \
        --max-tokens 4096 \
        --tokenizer space \
        --seed 0 \
        --buffer-size 100 \
        --input ${2}/test.de \
        > ${3}/test_out.en \
    "
}

fairseq-interactive \
        data_bin/CCrawl.de-en/token_count/words \
        --path checkpoints/ende_s0_token_count_words/checkpoint_best.pt \
        --beam 5 \
        --source-lang de \
        --target-lang en \
        --remove-bpe \
        --max-tokens 4096 \
        --tokenizer space

# --max-len-a 1.2 \
# --max-len-b 10 \
# --bpe fastbpe

for PEEK_TYPE in "fully_random" "ordered_random"; do
for RATE in "000" "010" "020" "030" "040" "050" "060" "070" "080" "090" "100"; do
    TEXT_DIR="data_bin/CCrawl.${LANG1}-${LANG2}/${PEEK_TYPE}/r${RATE}"
    SIGNATURE="ende_s0_${PEEK_TYPE}_r${RATE}"
    OUT_DIR="data_out/CCrawl.${LANG1}-${LANG2}/${PEEK_TYPE}/r${RATE}"
    
    launch_mt_infer $SIGNATURE $TEXT_DIR $OUT_DIR
done;
done;


for COUNT_TYPE in "words" "subwords"; do
    TEXT_DIR="data_bin/CCrawl.${LANG1}-${LANG2}/token_count/${COUNT_TYPE}"
    SIGNATURE="ende_s0_token_count_${COUNT_TYPE}"
    OUT_DIR="data_out/CCrawl.${LANG1}-${LANG2}/token_count/${COUNT_TYPE}"
    
    launch_mt_infer $SIGNATURE $TEXT_DIR $OUT_DIR
done;


for POS in "VERB" "NOUN" "PRON" "ADJ" "ADV" "ADP" "CONJ" "DET" "NUM" "PRT" "X" "."; do
    TEXT_DIR="data_bin/CCrawl.${LANG1}-${LANG2}/pos/${POS}";
    SIGNATURE="ende_s0_pos_${POS}"
    OUT_DIR="data_out/CCrawl.${LANG1}-${LANG2}/pos/${POS}";
    
    launch_mt_infer $SIGNATURE $TEXT_DIR $OUT_DIR
done;

for NER in "NORP" "NUM" "ORG" "GPE" "DATE" "ALL"; do
    TEXT_DIR="data_bin/CCrawl.${LANG1}-${LANG2}/ner/${NER}";
    SIGNATURE="ende_s0_ner_${NER}"
    OUT_DIR="data_out/CCrawl.${LANG1}-${LANG2}/ner/${NER}";
    
    launch_mt_infer $SIGNATURE $TEXT_DIR $OUT_DIR
done;


function launch_mt_infer_2() {
    # signature, text dir, output dir
    echo "Launching ${1} ${2} ${3} ${LANG1}-${LANG2}"
    MODEL_PATH="checkpoints/${1}/checkpoint_best.pt"
    mkdir -p ${3}

    # we use fairseq-interactive because fairseq-generate had some issues
    # it shouldn't be that much slower because we're using buffe-size 100 (batching)
    # so the only part where we're slower is some piping and the binarization,
    # which is fairly lightweight
    sbatch --time=07-00 --ntasks=8 --mem-per-cpu=4G --gpus=1 \
    --job-name="infer_mt_${4}" \
    --output="logs/infer_mt_${4}.log" \
    --wrap="CUDA_VISIBLE_DEVICES=0 
    fairseq-interactive \
        ${2} \
        --path ${MODEL_PATH} \
        --beam 5 \
        --source-lang $LANG1 \
        --target-lang $LANG1 \
        --gen-subset ${2}/test \
        --remove-bpe \
        --max-tokens 4096 \
        --tokenizer space \
        --seed 0 \
        --buffer-size 100 \
        --input ${2}/test.de \
        > ${3}/test_out.en \
    "
}

for ADV in "same" "syn" "rand"; do
    TEXT_DIR="data_bin/CCrawl.${LANG1}-${LANG2}/adversarial/${ADV}";
    SIGNATURE="ende_s0_fully_random_r030"
    SIGNATURE2="ende_s0_adversarial_${ADV}"
    OUT_DIR="data_out/CCrawl.${LANG1}-${LANG2}/adversarial/${ADV}";
    
    launch_mt_infer_2 $SIGNATURE $TEXT_DIR $OUT_DIR $SIGNATURE2
done;