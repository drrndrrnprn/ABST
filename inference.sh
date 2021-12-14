#!/bin/bash

USER_DIR='/home/drrndrrnprn/nlp/ABST/bartabst/'
DATA_PATH='/home/drrndrrnprn/nlp/ABST/datasets'
PREFIX='semeval-pengb'
DOMAIN='analyzed'

INPUT_PATH="$DATA_PATH/$PREFIX/$DOMAIN"
OUTPUT_PATH="$INPUT_PATH"

mkdir -p $OUTPUT_PATH

cp $INPUT_PATH/data-bin/dict.txt  $USER_DIR/checkpoints/
python $USER_DIR/inference.py \
    --model-dir checkpoints/bart_abst \
    --model-file checkpoint_best.pt \
