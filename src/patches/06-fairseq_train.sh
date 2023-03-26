#!/usr/bin/bash

LANG1="de"
LANG2="en"

for PEEK_TYPE in "fully_random" "ordered_random"; do
for RATE in "000" "010" "020" "030" "040" "050" "060" "070" "080" "090" "100"; do
    TEXT_DIR="data_bin/CCrawl.${LANG1}-${LANG2}/${PEEK_TYPE}/r${RATE}";
    SIGNATURE="ende_s0_${PEEK_TYPE}_r${RATE}"
    CHECKPOINT_DIR="checkpoints/${SIGNATURE}"
    mkdir -p ${CHECKPOINT_DIR}

    sbatch --time=07-00 --ntasks=8 --mem-per-cpu=4G --gpus=1 \
    --job-name="train_mt_${SIGNATURE}" \
    --output="logs/train_mt_${SIGNATURE}.log" \
    --wrap="CUDA_VISIBLE_DEVICES=0 fairseq-train \
        ${TEXT_DIR} \
        --arch transformer_wmt_en_de --share-all-embeddings \
        --dropout 0.3 --weight-decay 0.0 \
        --log-interval 2000 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.5 \
        --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
        --max-tokens 4096 \
        --eval-bleu \
        --patience 10 \
        --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' \
        --eval-bleu-detok moses \
        --eval-bleu-remove-bpe \
        --bpe fastbpe \
        --eval-bleu-print-samples \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
        --seed 0 \
        --save-dir $CHECKPOINT_DIR \
    "
done;
done;

    # --amp \