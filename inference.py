import argparse

import torch
from fairseq import tasks, options

from bartabst.data import data_utils
from bartabst.models.model import BARTMLModel
from bartabst.tasks.aspect_base_denoising import AspectBaseDenoisingTask


def main():
    """
    Usage::

         python examples/bart/summarize.py \
            --model-dir $HOME/bart.large.cnn \
            --model-file model.pt \
            --src $HOME/data-bin/cnn_dm/test.source
    """
    # parser = argparse.ArgumentParser()
    parser = options.get_generation_parser(default_task='aspect_base_denoising')
    parser.add_argument(
        "--model-dir",
        required=True,
        type=str,
        default="bart.large.cnn/",
        help="path containing model file and src_dict.txt",
    )
    parser.add_argument(
        "--model-file",
        default="checkpoint_best.pt",
        help="where in model_dir are weights saved",
    )
    parser.add_argument(
        "--src", default="test.source", help="text to summarize", type=str
    )
    parser.add_argument(
        "--out", default="test.hypo", help="where to save summaries", type=str
    )
    parser.add_argument("--bsz", default=32, help="where to save summaries", type=int)
    parser.add_argument(
        "--n", default=None, help="how many examples to summarize", type=int
    )
    # args = parser.parse_args()
    args = options.parse_args_and_arch(parser)

    task = tasks.setup_task(args)
    
    model = BARTMLModel.from_pretrained(
            args.model_dir,
            checkpoint_file=args.model_file,
            data_name_or_path=args.model_dir,
        )
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda().half()

    
    dataset = task.build_dataset_for_inference(src_tokens, src_lengths)
    
    # a_s, a_e, o_s, o_e, p = self.aos_list[index]
    # a_s, a_e, o_s, o_e = a_s + 1, a_e + 1, o_s + 1, o_e +1
    # asp_mask = np.full(sz, False)
    # opn_mask = np.full(sz, False)
    # asp_mask_idc = np.asarray([a_s+i for i in range(a_e-a_s)])
    # opn_mask_idc = np.asarray([o_s+i for i in range(o_e-o_s)])
    # asp_mask[asp_mask_idc] = True
    # opn_mask[opn_mask_idc] = True
    
    # new_item = np.copy(item)
    # new_item[mask] = self.mask_idx['mask']  
    # new_item[asp_mask] = self.mask_idx['asp_mask']
    # dic_mask = {'POS':'pos_mask', 'NEU':'neu_mask','NEG':'neg_mask'}
    # new_item[opn_mask] = self.mask_idx[dic_mask[p]]        
    
    
    masked_sentences = 
    outputs = model.fill_mask(masked_sentences)


if __name__ == "__main__":
    main()
