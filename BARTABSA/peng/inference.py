import sys
sys.path.append('../')
import os
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
import itertools
import json

from fastNLP.core.tester import Tester
import numpy as np
import torch

from data.pipe import BartBPEABSAPipe
from peng.model.predictor import Predictor
from peng.model.metrics import Seq2SeqSpanMetric
from peng.model.utils import process_aos, output_json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--bart_name', default='facebook/bart-base', type=str)
parser.add_argument('--dataset_path', default='/home/drrndrrnprn/nlp/ABST/BARTABSA/data/pengb/res', type=str)
parser.add_argument('--opinion_first', action='store_true', default=False)
parser.add_argument('--mode_select', choices=["preprocess", "eval"], default="eval")

args= parser.parse_args()

model_path = args.model_path
dataset_path = args.dataset_path

bart_name = args.bart_name
opinion_first = args.opinion_first
op_mode = args.mode_select
output_path = ('/home/drrndrrnprn/nlp/ABST/BARTABSA/output/' + str(dataset_path.split('/')[-2]))

def get_data():
    pipe = BartBPEABSAPipe(tokenizer=bart_name, opinion_first=opinion_first)
    data_bundle = pipe.process_from_file(dataset_path)
    return data_bundle, pipe.tokenizer, pipe.mapping2id, pipe.mapping

data_bundle, tokenizer, mapping2id, mapping = get_data()
bos_token_id = 0  
eos_token_id = 1  
label_ids = list(mapping2id.values())

device = torch.device("cuda")
#model.load_state_dict(torch.load(model_path))
model = torch.load(model_path)
model.to(device)
predictor = Predictor(model)

def get_aos(data, mapping, op_mode):
    output = predictor.predict(data)
    output = output['pred']
    output = list(itertools.chain.from_iterable(output))
    aos = process_aos(output, mapping) if op_mode=='eval' else process_aos(output, mapping, data)
    return aos

data = dict()
aos= dict()
if op_mode == 'eval':
    assert data_bundle.num_dataset == 1
    data['eval'] = data_bundle.get_dataset('train')
    aos['eval'] = get_aos(data['eval'], mapping, op_mode)
    metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids), opinion_first=opinion_first)
    tester = Tester(model=model,data=data['eval'],metrics=metric,batch_size=16,use_cuda=True,verbose=0)
    score = tester.test()
    # metric.evaluate(target_span, pred, tgt_tokens)
    # res = metric.get_metric()
    output_json(data['eval'], aos['eval'], output_path +  '/output_' + os.path.basename(model_path) + '.json')
    with open(output_path + '/f_rec_pre.json', 'w') as f:
        json.dump(score['Seq2SeqSpanMetric'], f, indent=2)

else:
    for sep in ['train','dev','test']:
        data[sep] = data_bundle.get_dataset(sep)
        aos[sep] = get_aos(data[sep], mapping, op_mode)        

