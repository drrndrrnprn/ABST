# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from fairseq import utils
from fairseq.data import (
    AppendTokenDataset,
    DenoisingDataset,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
)
from fairseq.data.encoders.gpt2_bpe import GPT2BPE as bpe
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import LegacyFairseqTask, register_task
from bartabst.data.aspect_base_denoising_dataset import AspectBaseDenoisingDataset
from bartabst.data import data_utils
import numpy as np


logger = logging.getLogger(__name__)
@register_task("aspect_base_denoising")
class AspectBaseDenoisingTask(LegacyFairseqTask):
    """
    Denoising task for applying sequence to sequence denoising. (ie. BART)
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="path to data directory")
        parser.add_argument(
            "--tokens-per-sample",
            default=512,
            type=int,
            help="max number of total tokens over all segments"
            " per sample for dataset",
        )
        parser.add_argument(
            "--sample-break-mode",
            default="complete_doc",
            type=str,
            help="mode for breaking sentence",
        )
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
            "--shorten-method",
            default="none",
            choices=["none", "truncate", "random_crop"],
            help="if not none, shorten sequences that exceed --tokens-per-sample",
        )
        parser.add_argument(
            "--shorten-data-split-list",
            default="",
            help="comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)',
        )


    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed

        # add mask token
        self.mask_idx = {}
        self.mask_idx['mask'] = self.dictionary.add_symbol("<mask>")
        self.mask_idx['asp_mask'] = self.dictionary.add_symbol("<asp_mask>")
        self.mask_idx['pos_mask'] = self.dictionary.add_symbol("<pos_mask>")
        self.mask_idx['neu_mask'] = self.dictionary.add_symbol("<neu_mask>")
        self.mask_idx['neg_mask'] = self.dictionary.add_symbol("<neg_mask>")

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task."""
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))
        if not hasattr(args, "shuffle_instance"):
            args.shuffle_instance = False
        return cls(args, dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        aos_list = dataset.aos_list
        
        dataset = StripTokenDataset(dataset, self.dictionary.eos())

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.tokens_per_sample,
            self.args.seed,
        )

        # # create continuous blocks of tokens
        # dataset = TokenBlockDataset(
        #     dataset,
        #     dataset.sizes,
        #     self.args.tokens_per_sample - 2,  # one less for <s> and one for </s>
        #     pad=self.dictionary.pad(),
        #     eos=self.dictionary.eos(),
        #     break_mode=self.args.sample_break_mode,
        #     document_sep_len=0,
        # )
        # logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())
        dataset = AppendTokenDataset(dataset, self.source_dictionary.eos())

        mask_whole_words = (
            get_whole_word_mask(self.args, self.source_dictionary)
            if self.args.mask_length != "subword"
            else None
        )

        self.datasets[split] = AspectBaseDenoisingDataset(
            dataset,
            dataset.sizes,
            self.dictionary,
            self.mask_idx,
            mask_whole_words,
            shuffle=self.args.shuffle_instance,
            seed=self.seed,
            args=self.args,
            aos_list=aos_list,
        )
        logger.info(
            "Split: {0}, Loaded {1} samples of denoising_dataset".format(
                split,
                len(self.datasets[split]),
            )
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        """
        Generate batches for inference. We assume that the input begins with a
        bos symbol (`<s>`) and ends with an eos symbol (`</s>`).
        """
        pad = self.source_dictionary.pad()
        eos = self.source_dictionary.eos()
        src_dataset = TokenBlockDataset(
            src_tokens,
            src_lengths,
            block_size=self.args.tokens_per_sample - 2,  # for <s> and </s>
            pad=pad,
            eos=eos,
            break_mode=self.args.sample_break_mode,
            document_sep_len=0,
        )
        prev_output_tokens = PrependTokenDataset(
            StripTokenDataset(src_dataset, eos), eos
        )
        src_dataset = PadDataset(src_dataset, pad_idx=pad, left_pad=False)
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": src_dataset,
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                    "prev_output_tokens": PadDataset(
                        prev_output_tokens, pad_idx=pad, left_pad=False
                    ),
                },
                "target": src_dataset,
            },
            sizes=[np.array(src_lengths)],
        )
        
    def mask_dataset_for_inference(self, args, combine=False, **kwargs):
        split = 'test'
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[0]
        split_path = os.path.join(data_path, split)
        
        src_dataset = data_utils.load_indexed_dataset(
            split_path,
            self.dictionary,
            'raw',
            combine=combine,
        )
        if src_dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )
        src_lengths = src_dataset.sizes
        aos_list = src_dataset.aos_list
        ob_raw_aos_list = src_dataset.ob_raw_aos_list
        
        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        src_dataset = AppendTokenDataset(src_dataset, self.source_dictionary.eos())
        
        return AspectBaseDenoisingDataset(
        src_dataset,
        src_dataset.sizes,
        self.dictionary,
        self.mask_idx,
        mask_whole_words=None,
        shuffle=False,
        seed=self.seed,
        args=args,
        aos_list=aos_list,
        ob_raw_aos_list=ob_raw_aos_list
        )
        
        # pad = self.source_dictionary.pad()
        # eos = self.source_dictionary.eos()
        # # src_dataset = TokenBlockDataset(
        # #     src_dataset,
        # #     src_dataset.sizes,
        # #     block_size=self.args.tokens_per_sample - 2,  # for <s> and </s>
        # #     pad=pad,
        # #     eos=eos,
        # #     break_mode=self.args.sample_break_mode,
        # #     document_sep_len=0,
        # # )
        # # #prev_output_tokens = PrependTokenDataset(
        # #     StripTokenDataset(src_dataset, eos), eos
        # # )
        # src_dataset = PadDataset(src_dataset, pad_idx=pad, left_pad=False)
        # return NestedDictionaryDataset(
        #     {
        #         "id": IdDataset(),
        #         "net_input": {
        #             "src_tokens": src_dataset,
        #             "src_lengths": NumelDataset(src_dataset, reduce=False),
        #             "prev_output_tokens": PadDataset(
        #                 src_dataset, pad_idx=pad, left_pad=False
        #             ),
        #         },
        #         "target": src_dataset,
        #     },
        #     sizes=[np.array(src_lengths)],
        # )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.dictionary