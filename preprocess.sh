#!/bin/bash

DATA_PATH='/home/drrndrrnprn/nlp/ABST/datasets'
PREFIX='semeval-pengb'
DOMAIN='analyzed'

INPUT_PATH="$DATA_PATH/$PREFIX/$DOMAIN"
OUTPUT_PATH="$DATA_PATH/$PREFIX/$DOMAIN"

CHECKPOINT_PATH="/home/drrndrrnprn/nlp/ABST/bartabst/checkpoints/bart.base"
mkdir -p "$OUTPUT_PATH"

for SPLIT in train dev
do
	python -m examples.roberta.multiprocessing_bpe_encoder \
	--encoder-json "$DATA_PATH/gpt2_bpe/encoder.json" \
	--vocab-bpe "$DATA_PATH/gpt2_bpe/vocab.bpe" \
	--inputs "$INPUT_PATH/$SPLIT.txt" \
	--outputs "$OUTPUT_PATH/$SPLIT.bpe" \
	--workers=60 \
	--keep-empty;
done

fairseq-preprocess \
	--only-source \
	--trainpref "$OUTPUT_PATH/train.bpe" \
	--validpref "$OUTPUT_PATH/dev.bpe" \
	--destdir "$OUTPUT_PATH/data-bin/" \
	--dataset-impl=raw \
	--workers=60 \
	--srcdict "$CHECKPOINT_PATH/dict.txt" \
	--tgtdict "$CHECKPOINT_PATH/dict.txt"
