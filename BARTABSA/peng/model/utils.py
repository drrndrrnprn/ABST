from collections import OrderedDict
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
    print('process_aos')
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
        new_l_aos = list()
        for l_aos, t in zip(aos, target_span):
            ll_aos_ins = list()
            for ll_aos, l_t in zip(l_aos, t):
                if ll_aos == 'error':
                    continue           
                ll_aos[-1] = mapping[l_t[-1]-2]
                ll_aos_ins.append(ll_aos)
            new_l_aos.append(ll_aos_ins)
        return new_l_aos
    else:
        return aos


def output_json(data, aos, output_path):
    json_out = list()
    for d_ins, aos_ins in zip(data, aos):
        raw_words = d_ins['raw_words']
        f = False
        if aos_ins == None:
            f = True
        for i in range(len(aos_ins)):
            if aos_ins[i][1] > len(raw_words) or aos_ins[i][3] > len(raw_words):
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
            aspect['term'] = raw_words[aos_ins[i][0]:aos_ins[i][1]]
            opinion = OrderedDict()
            opinion['index'] = i
            opinion['from'] = int(aos_ins[i][2])
            opinion['to'] = int(aos_ins[i][3])
            opinion['term'] = raw_words[aos_ins[i][2]:aos_ins[i][3]]
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
