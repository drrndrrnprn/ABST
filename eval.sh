#!/bin/bash

MODEL_PATH='/home/drrndrrnprn/nlp/ABST/BARTABSA/peng/save_models/resbest_SequenceGeneratorModel_triple_f_2022-01-10-14-42-47-760814'
DATA_PATH='/home/drrndrrnprn/nlp/ABST/outputs/semeval-pengb/singleaos_20220110-13:04:11.json'

cd BARTABSA/peng || exit
python inference.py \
--model_path $MODEL_PATH \
--dataset_path $DATA_PATH 
cd ../../