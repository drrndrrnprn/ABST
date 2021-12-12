# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import logging
import os

from omegaconf import MISSING, II, OmegaConf

import numpy as np
from fairseq import utils
from fairseq.data import (
    Dictionary,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.encoders.gpt2_bpe import GPT2BPE
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

from fairseq.tasks.language_modeling import SAMPLE_BREAK_MODE_CHOICES, SHORTEN_METHOD_CHOICES
import torch
from bartabst.data.aspect_base_mask_token_dataset import AspectBaseMaskTokensDataset
logger = logging.getLogger(__name__)


@dataclass
class MaskedLMConfig(FairseqDataclass):
    data: str = field(
        default=MISSING,
        metadata={
            "help": "colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner"
        },
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    mask_prob: float = field(
        default=0.15,
        metadata={"help": "probability of replacing a token with mask"},
    )
    leave_unmasked_prob: float = field(
        default=0.1,
        metadata={"help": "probability that a masked token is unmasked"},
    )
    random_token_prob: float = field(
        default=0.1,
        metadata={"help": "probability of replacing a token with a random token"},
    )
    freq_weighted_replacement: bool = field(
        default=False,
        metadata={"help": "sample random replacement words based on word frequencies"},
    )
    mask_whole_words: bool = field(
        default=False,
        metadata={"help": "mask whole words; you may also want to set --bpe"},
    )
    mask_multiple_length: int = field(
        default=1,
        metadata={"help": "repeat the mask indices multiple times"},
    )
    mask_stdev: float = field(
        default=0.0,
        metadata={"help": "stdev of the mask length"},
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    dataset_implem: str = field(
        default="raw",
        metadata={
            "help": "select dataset implementation"
            'e.g., raw, mmap ...'
        },
    )
    gpt2_encoder_json: str = field(
        default='dummy', metadata={"help": "path to encoder.json"}
    )
    gpt2_vocab_bpe: str = field(
        default='dummy', metadata={"help": "path to vocab.bpe"}
    )
    seed: int = II("common.seed")


@register_task("bart_e_mlm", dataclass=MaskedLMConfig)
class BARTEncoderMLMTask(FairseqTask):

    cfg: MaskedLMConfig

    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, cfg: MaskedLMConfig, dictionary):
        super().__init__(cfg)
        self.dictionary = dictionary

        # add mask token
        self.mask_idx = {}
        self.mask_idx['mask'] = dictionary.add_symbol("<mask>")
        self.mask_idx['asp_mask'] = self.dictionary.add_symbol("<asp_mask>")
        self.mask_idx['pos_mask'] = self.dictionary.add_symbol("<pos_mask>")
        self.mask_idx['neu_mask'] = self.dictionary.add_symbol("<neu_mask>")
        self.mask_idx['neg_mask'] = self.dictionary.add_symbol("<neg_mask>")
        
    @classmethod
    def setup_task(cls, cfg: MaskedLMConfig, **kwargs):
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(cfg, dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            combine=combine,
            dataset_impl=self.cfg.dataset_implem
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        aos_list = self.get_aos(split_path, dataset.lines, self.cfg)
                
        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.cfg.shorten_data_split_list,
            self.cfg.shorten_method,
            self.cfg.tokens_per_sample,
            self.cfg.seed,
        )

        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.cfg.tokens_per_sample - 1,  # one less for <s>
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode=self.cfg.sample_break_mode,
        )
        logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))
        
        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

        # create masked input and targets
        mask_whole_words = (
            get_whole_word_mask(self.args, self.source_dictionary)
            if self.cfg.mask_whole_words
            else None
        )

        src_dataset, tgt_dataset = AspectBaseMaskTokensDataset.apply_mask(
            dataset,
            self.source_dictionary,
            pad_idx=self.source_dictionary.pad(),
            mask_idx=self.mask_idx,
            seed=self.cfg.seed,
            mask_prob=self.cfg.mask_prob,
            leave_unmasked_prob=self.cfg.leave_unmasked_prob,
            random_token_prob=self.cfg.random_token_prob,
            freq_weighted_replacement=self.cfg.freq_weighted_replacement,
            mask_whole_words=mask_whole_words,
            mask_multiple_length=self.cfg.mask_multiple_length,
            mask_stdev=self.cfg.mask_stdev,
        )

        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(src_dataset))

        self.datasets[split] = SortDataset(
            NestedDictionaryDataset(
                {
                    "id": IdDataset(),
                    "net_input": {
                        "src_tokens": RightPadDataset(
                            src_dataset,
                            pad_idx=self.source_dictionary.pad(),
                        ),
                        "src_lengths": NumelDataset(src_dataset, reduce=False),
                    },
                    "target": RightPadDataset(
                        tgt_dataset,
                        pad_idx=self.source_dictionary.pad(),
                    ),
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(src_dataset, reduce=True),
                },
                sizes=[src_dataset.sizes],
            ),
            sort_order=[
                shuffle,
                src_dataset.sizes,
            ],
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        src_dataset = RightPadDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                self.cfg.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            pad_idx=self.source_dictionary.pad(),
        )
        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        src_dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": src_dataset,
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

    def get_aos(self, path, lines, cfg):
        cfg.gpt2_encoder_json = os.path.join(self.cfg.data, 'gpt2_bpe/encoder.json')
        cfg.gpt2_vocab_bpe = os.path.join(self.cfg.data, 'gpt2_bpe/vocab.bpe')
        bpe = GPT2BPE(cfg)
        aos_path = path + '_asp.txt'
        aos_list = []
        raw_aos_list = []
        with open(aos_path, "r", encoding="utf-8") as f:
            for aos_line in f:
                new_aos_line = [aos.split(',') for aos in aos_line.strip("\n").split('\t')]
                raw_aos_list.append(new_aos_line)
        
        for line, raw_aos in zip(lines, raw_aos_list):
            raw_aos = raw_aos[0]
            a_s, a_e, o_s, o_e = int(raw_aos[0]), int(raw_aos[1]), int(raw_aos[2]), int(raw_aos[3])
            # assert a_s < a_e, 'a_s >= a_e'
            # assert o_s < o_e, 'o_s >= o_e'
            # assert a_e-1 < o_s or o_e-1 < a_s, 'aos overlapping '
            if not ((a_s < a_e) and (o_s < o_e) and (a_e-1 < o_s or o_e-1 < a_s)):
                continue

            raw_line = bpe.decode(line).split(' ')
            if a_s < o_s:
                sep_raw_line = [raw_line[:a_s],
                                raw_line[a_s:a_e],
                                raw_line[a_e:o_s],
                                raw_line[o_s:o_e],
                                raw_line[o_e:]]
            else:
                sep_raw_line = [raw_line[:o_s],
                                raw_line[o_s:o_e],
                                raw_line[o_e:a_s],
                                raw_line[a_s:a_e],
                                raw_line[a_e:]]
            for i in range(1,len(sep_raw_line)):
                if sep_raw_line[i] != [] and not (i==1 and sep_raw_line[0] == []):
                    sep_raw_line[i] = (' ' + '#$%'.join(sep_raw_line[i])).split('#$%')
                if sep_raw_line[0] == []:
                    sep_raw_line[1][0].strip(' ')
            #sep_raw_line = '#$%'.join(sep_raw_line).strip().split('#$%')
            cumsum_length = 0
            aos = []
            encoded_line = np.empty(1)
            for sep in sep_raw_line:
                encoded = self.dictionary.encode_line(bpe.encode(' '.join(sep)))
                encoded = encoded[:-1]
                cumsum_length += len(encoded)
                aos.append(cumsum_length)
                encoded_line = np.append(encoded_line, encoded.cpu().numpy())
            
            encoded_line = encoded_line[1:]
            #encoded_line = np.append(encoded_line, np.array([2]))
            encoded_line = list(map(int,encoded_line.tolist()))
            sentence = bpe.decode(self.dictionary.string(encoded_line))

            if a_s > o_s:
                buf_s, buf_e = aos[0], aos[1]
                aos[0], aos[1] = aos[2], aos[3]
                aos[2], aos[3] = buf_s, buf_e
            aos = aos[:-1]
            # aos[1] += 1
            # aos[3] += 1
            aos.append(raw_aos[4])
            aos_list.append(aos)
            raw_sentence = ' '.join(raw_line)
            assert sentence == raw_sentence, 'bpe encodeing error'
            
            a_t = bpe.decode(self.dictionary.string(encoded_line[aos[0]:aos[1]]))
            o_p = bpe.decode(self.dictionary.string(encoded_line[aos[2]:aos[3]]))
            r_a_t = ' '.join(raw_line[a_s:a_e])
            r_o_p = ' '.join(raw_line[o_s:o_e])
            assert a_t.strip() == r_a_t, 'bpe encoding error'
            assert o_p.strip() == r_o_p, 'bpe encoding error'
        return aos_list
                
    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
