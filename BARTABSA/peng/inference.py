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
from peng.model.bart_absa import BartSeq2SeqModel
from peng.model.generator import SequenceGeneratorModel
from peng.model.predictor import Predictor

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--dataset_name', default='pengb/16res', type=str)
parser.add_argument('--num_beams', default=4, type=int)
parser.add_argument('--opinion_first', action='store_true', default=False)
parser.add_argument('--decoder_type', default='avg_score', type=str, choices=['None', 'avg_score'])
parser.add_argument('--length_penalty', default=1.0, type=float)
parser.add_argument('--bart_name', default='facebook/bart-base', type=str)
parser.add_argument('--use_encoder_mlp', type=int, default=1)

args= parser.parse_args()

model_path = args.model_path
num_beams = args.num_beams
dataset_name = args.dataset_name
opinion_first = args.opinion_first
length_penalty = args.length_penalty
if isinstance(args.decoder_type, str) and args.decoder_type.lower() == 'none':
    args.decoder_type = None
decoder_type = args.decoder_type
bart_name = args.bart_name
use_encoder_mlp = args.use_encoder_mlp

def get_data():
    pipe = BartBPEABSAPipe(tokenizer=bart_name, opinion_first=opinion_first)
    data_bundle = pipe.process_from_file(f'../data/{dataset_name}')
    return data_bundle, pipe.tokenizer, pipe.mapping2id

data_bundle, tokenizer, mapping2id = get_data()
max_len = 10
max_len_a = {
    'penga/14lap': 0.9,
    'penga/14res': 1,
    'penga/15res': 1.2,
    'penga/16res': 0.9,
    'pengb/14lap': 1.1,
    'pengb/14res': 1.2,
    'pengb/15res': 0.9,
    'pengb/16res': 1.2
}[dataset_name]

bos_token_id = 0  #
eos_token_id = 1  #
label_ids = list(mapping2id.values())
model = BartSeq2SeqModel.build_model(bart_name, tokenizer, label_ids=label_ids, decoder_type=decoder_type,
                                     copy_gate=False, use_encoder_mlp=use_encoder_mlp, use_recur_pos=False)

model = SequenceGeneratorModel(model, bos_token_id=bos_token_id,
                               eos_token_id=eos_token_id,
                               max_length=max_len, max_len_a=max_len_a,num_beams=num_beams, do_sample=False,
                               repetition_penalty=1, length_penalty=length_penalty, pad_token_id=eos_token_id,
                               restricter=None)

device = torch.device("cuda")
#model.load_state_dict(torch.load(model_path))
model = torch.load(model_path)
model.to(device)
pred = Predictor(model)

output = pred.predict(data_bundle.get_dataset('test'))
print(data_bundle.get_dataset('test'))
print(output['pred'][0])
print(len(data_bundle.get_dataset('test')))

#data_bundle.get_dataset('train') class fastNLP.core.dataset.DataSet
#pprint.pprint(output)
#exclude outputs with format error
# i.e. array([[ 0, 13, 13, 12, 12,  2, 15, 15, 12, 12,  2, 18, 18, 18, 12, 12, 2, 1]])
output = output['pred']
output = list(itertools.chain.from_iterable(output))
for line in output:
    line = line[1:-1]
    if len(line) % 5 != 0:
        line = 'error'
    else:
        line = [line[i*5:i*5+5].tolist() for i in range(len(line)//5)]
    
        
    print(line)
