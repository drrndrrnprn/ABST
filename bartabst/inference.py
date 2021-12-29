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
    parser = options.get_parser(None)
    # parser = options.get_generation_parser(default_task='aspect_base_denoising')
    parser.add_argument("data", help="path to data directory")
    parser.add_argument(
        "--task",
        type=str,
        default="aspect_base_denoising",
        help="path containing model file and src_dict.txt",
    )
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
    args = options.parse_args_and_arch(parser)
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mask",
        default=0.0,
        type=float,
        help="fraction of words/subwords that will be masked",
    )
    parser.add_argument(
        "--mask-random",
        default=0.0,
        type=float,
        help="instead of using [MASK], use random token this often",
    )
    parser.add_argument(
        "--insert",
        default=0.0,
        type=float,
        help="insert this percentage of additional random tokens",
    )
    parser.add_argument(
        "--permute",
        default=0.0,
        type=float,
        help="take this proportion of subwords and permute them",
    )
    parser.add_argument(
        "--rotate",
        default=0.0,
        type=float,
        help="rotate this proportion of inputs",
    )
    parser.add_argument(
        "--poisson-lambda",
        default=0.0,
        type=float,
        help="randomly shuffle sentences for this proportion of inputs",
    )
    parser.add_argument(
        "--permute-sentences",
        default=0.0,
        type=float,
        help="shuffle this proportion of sentences in all inputs",
    )
    parser.add_argument(
        "--mask-length",
        default="subword",
        type=str,
        choices=["subword", "word", "span-poisson"],
        help="mask length to choose",
    )
    parser.add_argument(
        "--replace-length",
        default=-1,
        type=int,
        help="when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)",
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
    )    
    #args = parser.parse_args()

    task = tasks.setup_task(args)
    
    model = BARTMLModel.from_pretrained(
            args.model_dir,
            checkpoint_file=args.model_file,
            data_name_or_path=args.model_dir,
        )
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda().half()

    dataset = task.build_dataset_for_inference(task.args)
    itr = task.get_batch_iterator(dataset).next_epoch_itr(shuffle=False)
    outputs = model.fill_mask(**itr['net_input'], match_source_len=False)


if __name__ == "__main__":
    main()
