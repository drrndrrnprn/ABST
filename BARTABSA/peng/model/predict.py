import torch

from data.pipe import BartBPEABSAPipe
from peng.model.bart_absa import BartSeq2SeqModel
from peng.model.generator import SequenceGeneratorModel
import predictor 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='pengb/14lap', type=str)
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_beams', default=4, type=int)
parser.add_argument('--opinion_first', action='store_true', default=False)
parser.add_argument('--n_epochs', default=50, type=int)
parser.add_argument('--decoder_type', default='avg_score', type=str, choices=['None', 'avg_score'])
parser.add_argument('--length_penalty', default=1.0, type=float)
parser.add_argument('--bart_name', default='facebook/bart-base', type=str)
parser.add_argument('--use_encoder_mlp', type=int, default=1)
parser.add_argument('--save_model', type=int, default=0)

args= parser.parse_args()

lr = args.lr
n_epochs = args.n_epochs
batch_size = args.batch_size
num_beams = args.num_beams
dataset_name = args.dataset_name
opinion_first = args.opinion_first
length_penalty = args.length_penalty
if isinstance(args.decoder_type, str) and args.decoder_type.lower() == 'none':
    args.decoder_type = None
decoder_type = args.decoder_type
bart_name = args.bart_name
use_encoder_mlp = args.use_encoder_mlp
save_model = args.save_model
demo = False

def get_data():
    pipe = BartBPEABSAPipe(tokenizer=bart_name, opinion_first=opinion_first)
    data_bundle = pipe.process_from_file(f'../data/{dataset_name}', demo=demo)
    return data_bundle, pipe.tokenizer, pipe.mapping2id

data_bundle, tokenizer, mapping2id = get_data()

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
model.load_state_dict(torch.load(model_path))
model.to(device)

pred = predictor(model)

print(pred(data_bundle))



