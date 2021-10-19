import argparse
import os
import json

#analyzed data(.json) --> raw text(.txt), aspect infomation(.txt)
def extract_aspect(data):
    raw_texts = list()
    aspect_info = list()
    for dic in data:
        raw_words = dic['raw_words']
        aspects = list()
        for a, o in zip(dic['aspects'], dic['opinions']):
            aspect = list()
            aspect.append(a['from'])
            aspect.append(a['to'])
            aspect.append(o['from'])
            aspect.append(o['to'])
            aspect.append(a['polarity'])        
            aspects.append(aspect)
        raw_texts.append(raw_words)
        aspect_info.append(aspects)
    
    return raw_texts, aspect_info
        
#raw text(.txt) and aspect infomation(.txt) --> train(.txt), dev(.txt), test(.txt) 
def separate_data(lines, n_dev=2000, n_test=500):
    n_lines = float(len(lines))
    sep1, sep2 = 0, 0
    if n_lines > n_test*10:
        sep2 = n_lines - n_test
        sep1 = sep2 - n_dev
    else:
        sep2 = n_lines - n_lines*0.1
        sep1 = sep1 = n_lines*0.2
        
    test = lines[sep2:]
    dev = lines[sep1:]
    train = lines[:sep1]
    return [train, dev, test]

def output_file(filename, text, aspect):
    output_path = filename + '.txt'
    with open(output_path, 'w') as f:
        #f.write('\n'.join(text))
        pass
    
    output_path = filename + '_asp.txt'
    with open(output_path, 'w') as f:
        #f.write('\n'.join(aspect))
        pass    

def concat_jsonfiles(self):
    pass

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('--only_extract', action='store_true', default=False)
    args= parser.parse_args()
    
    data_dir = os.path.dirname(args.data_path)
    with open(args.data_path, 'r') as f:
        data = json.load(f)
        text, aspect = extract_aspect(data)

    if args.only_extract:
        output_path = data_dir + '/' + os.path.splitext(os.path.basename(args.data_path))[0]
        output_file(output_path, text, aspect)
    else:  
        separated_text = separate_data(text)
        separated_aspect = separate_data(aspect)
        for st, sa in zip(separated_text, separated_aspect):
            output_path = data_dir + '/' + str(st)
            output_file(output_path, st, sa)


if __name__ == "__main__":                
    main() 