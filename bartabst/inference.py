from collections import OrderedDict
import datetime
import json
import os

import torch
from fairseq import tasks, options, utils
from sacremoses import MosesTokenizer, MosesDetokenizer

def main():
    """
    Usage::

        python examples/bart/summarize.py \
            --model-dir $HOME/bart.large.cnn \
            --model-file model.pt \
            --src $HOME/data-bin/cnn_dm/test.source
    """
    parser = options.get_parser(None)
    parser.add_argument(
        "--model-dir",
        required=True,
        type=str,
        help="path containing model file and src_dict.txt",
    )
    parser.add_argument(
        "--model-file",
        default="checkpoint_best.pt",
        help="where in model_dir are weights saved",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="bart_abst",
    )
    parser
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
    ) 
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="path savedir for output",
    )
    parser.add_argument(
        "--transfer_aos_path",
        default=None,
        type=str,
        help="path to transfer aos file",
    )
    args = options.parse_args_and_arch(parser)
    
    args.mask = 0.0
    args.mask_random = 0.0
    args.insert = 0.0
    args.permute = 0.0
    args.rotate = 0.0
    args.poisson_lambda = 0.0
    args.permute_sentences = 0.0
    args.mask_length = 'subword'
    args.replace_length = 0.0
    
    md = MosesDetokenizer(lang='en')
    dt_now = datetime.datetime.now()
    cur_t = dt_now.strftime('%Y%m%d-%H:%M:%S')
    
    task = tasks.setup_task(args)
    
    model = task.build_model(args)
    model = model.from_pretrained(
            args.model_dir,
            checkpoint_file=args.model_file,
            data_name_or_path=args.model_dir,
        )
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda().half()

    dataset = task.mask_dataset_for_inference(task.args)
    itr = task.get_batch_iterator(dataset).next_epoch_itr(shuffle=False)
    for sample in itr:
        id = sample['id']
        id, idx = id.sort()
        src_tokens = sample['net_input']['src_tokens']
        src_ttt = torch.stack(tuple(src_tokens), dim=0)
        inputs_lengths = sample['net_input']['src_lengths']

        src_tokens = src_ttt.index_select(0, idx)
        inputs_lengths = inputs_lengths.index_select(0, idx)
        src_tokens_l = [src for src in src_tokens]
        inputs_lengths_l = [l for l in inputs_lengths]
        src_tokens = [src[:length].long() for src, length in zip(src_tokens_l, inputs_lengths_l)]
        outputs = model.abst(src_tokens, match_source_len=False)
        
        trans = [o[0][0] for o in outputs]
    aos_list = dataset.ob_raw_aos_list
    output_path = args.output_dir + "/" + cur_t +".json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # output json file
    output_json(trans, aos_list, output_path)

def output_json(data, aos, output_path):
    json_out = list()
    for d_ins, aos_ins in zip(data, aos):
        raw_words = d_ins.split(' ')
        f = False
        if aos_ins == None:
            f = True
        for i in range(len(aos_ins)):
            if int(aos_ins[i][1]) > len(raw_words) or int(aos_ins[i][3]) > len(raw_words):
                f = True #何も入ってないやつしたで消して
        if f:
            continue
        aspects = list()
        opinions = list()
        for i in range(len(aos_ins)):
            aspect =  OrderedDict()
            aspect['index'] = i
            aspect['from'] = int(aos_ins[i][0])
            aspect['to'] = int(aos_ins[i][1])
            aspect['polarity'] = aos_ins[i][4]
            aspect['term'] = raw_words[int(aos_ins[i][0]):int(aos_ins[i][1])]
            opinion = OrderedDict()
            opinion['index'] = i
            opinion['from'] = int(aos_ins[i][2])
            opinion['to'] = int(aos_ins[i][3])
            opinion['term'] = raw_words[int(aos_ins[i][2]):int(aos_ins[i][3])]
            aspects.append(aspect)
            opinions.append(opinion)
        json_ins = OrderedDict()
        json_ins['raw_words'] = ' '.join(raw_words)
        json_ins['words'] = raw_words
        json_ins['aspects'] = aspects
        json_ins['opinions'] = opinions
        json_out.append(json_ins)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode='w') as f:
        json.dump(json_out, f, sort_keys=False, indent=4)
        
if __name__ == "__main__":
    main()
