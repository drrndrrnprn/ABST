#!/bin/bash
USER_DIR='/home/drrndrrnprn/nlp/ABST/bartabst'
DATA_PATH='/home/drrndrrnprn/nlp/ABST/datasets'
MODEL_DIR="$USER_DIR/checkpoints/bart.abst/onlymask"
PREFIX='semeval-pengb'
DOMAIN='analyzed'
INPUT_PATH="$DATA_PATH/$PREFIX/$DOMAIN"
OUTPUT_PATH="/home/drrndrrnprn/nlp/ABST/outputs/$PREFIX"
AOS_FILE_NAME="data-raw/revf_test_asp.txt"

mkdir -p $OUTPUT_PATH

cp $USER_DIR/checkpoints/bart.base/dict.txt  $MODEL_DIR
python $USER_DIR/inference.py "$INPUT_PATH/data-raw" \
    --transfer_aos_path "$INPUT_PATH/$AOS_FILE_NAME" \
    --model-dir $MODEL_DIR \
    --model-file checkpoint_best.pt \
    --task=aspect_base_denoising \
    --arch=bart_abst \
    --output-dir $OUTPUT_PATH \
    --user-dir $USER_DIR