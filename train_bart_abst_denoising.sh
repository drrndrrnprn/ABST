#!/bin/bash

DATA_PATH='/home/drrndrrnprn/nlp/ABST/datasets'
PREFIX='semeval-pengb'
DOMAIN='analyzed'

INPUT_PATH="$DATA_PATH/$PREFIX/$DOMAIN"
OUTPUT_PATH="$INPUT_PATH"

# Add this parameter to the conmmand below
# to support the wandb backend, make sure you 
# install and setup your account first
# --wandb-project <wandb-project-name> 

fairseq-train "$INPUT_PATH/data-bin" \
    --log-interval=10 \
    --no-epoch-checkpoints \
    --no-progress-bar \
    --seed=42 \
    --criterion=cross_entropy \
    --optimizer=adam \
    --lr-scheduler=polynomial_decay \
    --task=denoising  \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens=8192 \
    --required-batch-size-multiple=1 \
    --train-subset=train \
    --valid-subset=dev \
    --max-tokens-valid=12288 \
    --validate-interval 1 \
    --bucket-cap-mb=25 \
    --arch=bart_abst \
    --max-update=500000 \
    --clip-norm=0.1 \
    --update-freq=1 \
    --lr 3e-5 \
    --stop-min-lr -1 \
    --keep-last-epochs=10 \
    --best-checkpoint-metric=loss \
    --adam-betas="(0.9, 0.98)" \
    --adam-eps=1e-06 \
    --weight-decay=0.01 \
    --warmup-updates=500 \
    --save-interval-updates 5000 \
    --validate-interval-updates 5000 \
    --power=1 \
    --tokens-per-sample=512 \
    --sample-break-mode=eos \
    --mask=0.3 \
    --mask-random=0.1 \
    --poisson-lambda=3.5 \
    --permute-sentences=1 \
    --mask-length=span-poisson \
    --replace-length=1 \
    --total-num-update 20000 \
    --dropout=0.3 \
    --attention-dropout=0.1 \
    --share-all-embeddings \
    --layernorm-embedding \
    --fp16 \
    --activation-fn=gelu \
    --restore-file "bart.base/model.pt" \
    --reset-optimizer \
    --reset-meters \
    --reset-dataloader \
    --reset-lr-scheduler \
    --pooler-activation-fn=tanh \
    --tensorboard-logdir="$OUTPUT_PATH/tensorboard" \
    --save-dir="$OUTPUT_PATH/checkpoints" | tee "$OUTPUT_PATH/train.log"