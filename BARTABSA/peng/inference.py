import sys
sys.path.append('../')
import os
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
import pprint
import itertools

import numpy as np
import torch

from data.pipe import BartBPEABSAPipe
from peng.model.predictor import Predictor

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--dataset_name', default='pengb/16res', type=str)
parser.add_argument('--opinion_first', action='store_true', default=False)
parser.add_argument('--bart_name', default='facebook/bart-base', type=str)

args= parser.parse_args()

model_path = args.model_path
dataset_name = args.dataset_name

bart_name = args.bart_name
opinion_first = args.opinion_first

def get_data():
    pipe = BartBPEABSAPipe(tokenizer=bart_name, opinion_first=opinion_first)
    data_bundle = pipe.process_from_file(f'../data/{dataset_name}')
    return data_bundle, pipe.tokenizer, pipe.mapping2id, pipe.mapping

data_bundle, tokenizer, mapping2id, mapping = get_data()

device = torch.device("cuda")
#model.load_state_dict(torch.load(model_path))
model = torch.load(model_path)
model.to(device)
pred = Predictor(model)

offset = len(mapping) + 3 #わからん bos+eos+?
mapping = list(mapping.keys())

output = pred.predict(data_bundle.get_dataset('test'))
print(data_bundle.get_dataset('test'))

#data_bundle.get_dataset('train') class fastNLP.core.dataset.DataSet
#pprint.pprint(output)
#exclude outputs with format error
# i.e. array([[ 0, 13, 13, 12, 12,  2, 15, 15, 12, 12,  2, 18, 18, 18, 12, 12, 2, 1]])
output = output['pred']
output = list(itertools.chain.from_iterable(output))
aos = list()
for line in output:
    p_line = list()
    line = line[1:-1]
    if len(line) % 5 != 0:
        p_line.append('error')
    else:
        for i in range(len(line)//5):
            l = line[i*5:i*5+5].tolist()
            sent = l[-1] - 2
            l = l[0:-1]
            l = list(map(lambda x: x-offset, l))
            l[1] += 1
            l[3] += 1
            l.append(mapping[sent])
            p_line.append(l)
            
    aos.append(p_line)
target_span = list(data_bundle.get_dataset('test')['target_span'])
tgt_s = list()
for t in target_span:
    s_line = [l[-1] for l in t] 
    tgt_s.append(s_line)

pprint.pprint(list(zip(aos,tgt_s)))
