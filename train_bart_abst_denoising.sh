#!/bin/bash
USER_DIR='/home/drrndrrnprn/nlp/ABST/bartabst/'
DATA_PATH='/home/drrndrrnprn/nlp/ABST/datasets'
RESTORE_DIR='bartabst/checkpoints/bart.mlm/dev'
SAVE_DIR='bartabst/checkpoints/bart.abst/dev'
PREFIX='semeval-pengb'
DOMAIN='analyzed'
INPUT_PATH="$DATA_PATH/$PREFIX/$DOMAIN"
OUTPUT_PATH="$INPUT_PATH"

fairseq-train "$INPUT_PATH/data-raw" \
    --log-interval=10 \
    --no-epoch-checkpoints \
    --no-progress-bar \
    --seed=42 \
    --criterion=cross_entropy \
    --optimizer=adam \
    --lr-scheduler=polynomial_decay \
    --task=aspect_base_denoising  \
    --insert=0.0 \
    --mask=0.0 \
    --mask-length='subword' \
    --mask-random=0.1 \
    --permute=0.0 \
    --permute-sentences=0.0 \
    --poisson-lambda=0.0 \
    --replace-length=1 \
    --rotate=0.0 \
    --warmup_epoch 15 \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens=8192 \
    --required-batch-size-multiple=1 \
    --train-subset=train \
    --valid-subset=valid \
    --max-tokens-valid=12288 \
    --validate-interval 1 \
    --bucket-cap-mb=25 \
    --arch=bart_abst \
    --max-update=500000 \
    --clip-norm=0.1 \
    --update-freq=1 \
    --lr 3e-5 \
    --stop-min-lr -1 \
    --patience 2 \
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
    --total-num-update 20000 \
    --dropout=0.3 \
    --attention-dropout=0.1 \
    --batch-size 32 \
    --share-all-embeddings \
    --layernorm-embedding \
    --fp16 \
    --activation-fn=gelu \
    --restore-file "$RESTORE_DIR/checkpoint_best.pt" \
    --reset-optimizer \
    --reset-meters \
    --reset-dataloader \
    --reset-lr-scheduler \
    --pooler-activation-fn=tanh \
    --tensorboard-logdir="$OUTPUT_PATH/tensorboard" \
    --user-dir "$USER_DIR" \
    --save-dir="$SAVE_DIR" | tee "$OUTPUT_PATH/train.log"


    #--finetune-from-model \