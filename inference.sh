#!/bin/bash

USER_DIR='/home/drrndrrnprn/nlp/ABST/bartabst'
DATA_PATH='/home/drrndrrnprn/nlp/ABST/datasets'
PREFIX='semeval-pengb'
DOMAIN='analyzed'

INPUT_PATH="$DATA_PATH/$PREFIX/$DOMAIN"
OUTPUT_PATH="/home/drrndrrnprn/nlp/ABST/outputs/$PREFIX"

mkdir -p $OUTPUT_PATH

cp $INPUT_PATH/data-bin/dict.txt  $USER_DIR/checkpoints/bart.abst
python $USER_DIR/inference.py "$INPUT_PATH/data-raw" \
    --model-dir $USER_DIR/checkpoints/bart.abst \
    --model-file checkpoint_best.pt \
    --task=aspect_base_denoising \
    --arch=bart_abst \
    --output-dir $OUTPUT_PATH \
    --user-dir "$USER_DIR" 