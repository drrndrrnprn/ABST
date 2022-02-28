#!/bin/bash

MODEL_PATH='/home/drrndrrnprn/nlp/ABST/BARTABSA/peng/save_models/resbest_SequenceGeneratorModel_triple_f_2022-01-10-14-42-47-760814'
DATA_PATH='/home/drrndrrnprn/nlp/ABST/outputs/semeval-pengb/revfaos_single_20220228-07:36:01.json'

cd BARTABSA/peng || exit
python inference.py \
--model_path $MODEL_PATH \
--dataset_path $DATA_PATH \

cd ../../