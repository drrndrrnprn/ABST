import argparse

import torch
from fairseq import tasks, options, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
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
        default="bart.large.cnn/",
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
    args = options.parse_args_and_arch(parser)
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--mask",
    #     default=0.0,
    #     type=float,
    #     help="fraction of words/subwords that will be masked",
    # )
    # parser.add_argument(
    #     "--mask-random",
    #     default=0.0,
    #     type=float,
    #     help="instead of using [MASK], use random token this often",
    # )
    # parser.add_argument(
    #     "--insert",
    #     default=0.0,
    #     type=float,
    #     help="insert this percentage of additional random tokens",
    # )
    # parser.add_argument(
    #     "--permute",
    #     default=0.0,
    #     type=float,
    #     help="take this proportion of subwords and permute them",
    # )
    # parser.add_argument(
    #     "--rotate",
    #     default=0.0,
    #     type=float,
    #     help="rotate this proportion of inputs",
    # )
    # parser.add_argument(
    #     "--poisson-lambda",
    #     default=0.0,
    #     type=float,
    #     help="randomly shuffle sentences for this proportion of inputs",
    # )
    # parser.add_argument(
    #     "--permute-sentences",
    #     default=0.0,
    #     type=float,
    #     help="shuffle this proportion of sentences in all inputs",
    # )
    # parser.add_argument(
    #     "--mask-length",
    #     default="subword",
    #     type=str,
    #     choices=["subword", "word", "span-poisson"],
    #     help="mask length to choose",
    # )
    # parser.add_argument(
    #     "--replace-length",
    #     default=-1,
    #     type=int,
    #     help="when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)",
    # )
    #new_args = parser.parse_args()
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

    # utils.import_user_module(args)
    task = tasks.setup_task(args)
    
    cfg = convert_namespace_to_omegaconf(args)
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
        print(trans)
        
if __name__ == "__main__":
    main()
