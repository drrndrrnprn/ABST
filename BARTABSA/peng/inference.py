import sys
sys.path.append('../')
import os
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
import itertools

import numpy as np
import torch

from data.pipe import BartBPEABSAPipe
from peng.model.predictor import Predictor
from peng.model.utils import process_aos, output_json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--bart_name', default='facebook/bart-base', type=str)
parser.add_argument('--dataset_path', default='/home/drrndrrnprn/nlp/ABST/BARTABSA/data/pengb/res', type=str)
parser.add_argument('--opinion_first', action='store_true', default=False)
parser.add_argument('--use_input_polarity', action='store_true', default=False)

args= parser.parse_args()

model_path = args.model_path
dataset_path = args.dataset_path

bart_name = args.bart_name
opinion_first = args.opinion_first
use_input_polarity = args.use_input_polarity

def get_data():
    pipe = BartBPEABSAPipe(tokenizer=bart_name, opinion_first=opinion_first)
    data_bundle = pipe.process_from_file(dataset_path)
    return data_bundle, pipe.tokenizer, pipe.mapping2id, pipe.mapping

data_bundle, tokenizer, mapping2id, mapping = get_data()

device = torch.device("cuda")
#model.load_state_dict(torch.load(model_path))
model = torch.load(model_path)
model.to(device)
pred = Predictor(model)

data = data_bundle.get_dataset('test')
output = pred.predict(data)

output = output['pred']
output = list(itertools.chain.from_iterable(output))

if use_input_polarity:
    aos = process_aos(output, mapping, data)
else:
    aos = process_aos(output, mapping)
    
output_path = ('/home/drrndrrnprn/nlp/ABST/BARTABSA/output/' + dataset_name + '/output_' + os.path.basename(model_path) + '.json')
output_json(data, aos, output_path)
