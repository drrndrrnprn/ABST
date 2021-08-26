import json
import os

import numpy as np

def get_max_len_max_len_a(data_bundle, max_len=10):
    """
    :param data_bundle:
    :param max_len:
    :return:
    """
    max_len_a = -1
    for name, ds in data_bundle.iter_datasets():
        if name=='train':continue
        src_seq_len = np.array(ds.get_field('src_seq_len').content)
        tgt_seq_len = np.array(ds.get_field('tgt_seq_len').content)
        _len_a = round(max(np.maximum(tgt_seq_len - max_len+2, 0)/src_seq_len), 1)

        if _len_a>max_len_a:
            max_len_a = _len_a

    return max_len, max_len_a


def get_num_parameters(model):
    num_param = 0
    for name, param in model.named_parameters():
        num_param += np.prod(param.size())
    print(f"The number of parameters is {num_param}")
    

def process_aos(output, mapping, data=None):
    aos = list()
    offset = len(mapping) + 3
    mapping = list(mapping.keys())
    
    for line in output:
        p_line = list()
        line = list(line)
        line = line[1:-1]
        if len(line) % 5 != 0:
            p_line.append('error')
        else:
            for i in range(len(line)//5):
                l = line[i*5:i*5+5]
                sent = l[-1] - 2
                l = l[0:-1]
                l = list(map(lambda x: x-offset, l))
                l[1] += 1
                l[3] += 1
                l.append(mapping[sent])
                p_line.append(l)     
        aos.append(p_line)
        if data:
            target_span = list(data['target_span'])
            tgt_s = list()
            new_aos = list()
            for t in target_span:
                s = [l[-1] for l in t] 
            tgt_s.append(s)
            for line, s in zip(aos, tgt_s):
                line[-1] = s
                new_aos.append(line)
            aos = new_aos
            
    return aos


def output_json(data, aos, output_path):
    json_out = list()
    for d_ins, aos_ins in zip(data, aos):# 極性のつけ方どうしよ
        raw_words = d_ins['raw_words']
        if aos_ins[0] == 'error':
            continue
        for i in range(len(aos_ins)):
            if aos_ins[i][1] > len(raw_words) or aos_ins[i][3] > len(raw_words):
                continue
        aspects = list()
        opinions = list()
        for i in range(len(aos_ins)):
            aspect =  \
                {'index' : i,
                'from' : int(aos_ins[i][0]),
                'to' : int(aos_ins[i][1]),
                'polarity' : aos_ins[i][4],
                'term' : raw_words[aos_ins[i][0]:aos_ins[i][1]]}
            opinion = \
                {'index' : i,
                'from' : int(aos_ins[i][2]),
                'to' : int(aos_ins[i][3]),
                'term' : raw_words[aos_ins[i][2]:aos_ins[i][3]]}
            aspects.append(aspect)
            opinions.append(opinion)
        json_ins = \
            {'raw_words': ' '.join(raw_words),
            'words': raw_words,
            'aspects': aspects,
            'opinions': opinions}
        json_out.append(json_ins)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode='w') as f:
        json.dump(json_out, f, sort_keys=True, indent=4)
