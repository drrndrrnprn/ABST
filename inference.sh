#!/bin/bash

USER_DIR='/home/drrndrrnprn/nlp/ABST'
DATA_PATH='/home/drrndrrnprn/nlp/ABST/datasets'
PREFIX='semeval-pengb'
DOMAIN='analyzed'

INPUT_PATH="$DATA_PATH/$PREFIX/$DOMAIN"
OUTPUT_PATH="$INPUT_PATH"

mkdir -p $OUTPUT_PATH

cp $INPUT_PATH/data-bin/dict.txt  $USER_DIR/bartabst/checkpoints/bart_abst
python $USER_DIR/bartabst/inference.py "$INPUT_PATH/data-bin" \
    --model-dir checkpoints/bart_abst \
    --model-file checkpoint_best.pt \
    --task=aspect_base_denoising \
    --insert=0.0 \
    --mask=0.0 \
    --mask-length='subword' \
    --mask-random=0.0 \
    --permute=0.0 \
    --permute-sentences=0.0 \
    --poisson-lambda=0.0 \
    --replace-length=-1 \
    --rotate=0.0 \
    --user-dir "$USER_DIR" 
