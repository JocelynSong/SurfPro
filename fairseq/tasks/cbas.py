# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
from dataclasses import dataclass, field
import logging
import os
import torch
import json
import torch.nn.functional as F

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
from fairseq.file_io import PathManager
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from fairseq.tasks import FairseqTask, register_task
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.data import FairseqDataset

from .language_modeling import SAMPLE_BREAK_MODE_CHOICES, SHORTEN_METHOD_CHOICES
from fairseq.cbas_utils import ESM1b_Landscape


logger = logging.getLogger(__name__)


class SampleDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.

    Original lines are also kept in memory"""

    def __init__(self, samples, dictionary, append_eos=True, reverse_order=False):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(samples, dictionary)
        self.size = len(self.tokens_list)

    def read_data(self, samples, dictionary):
        for line in samples:
            self.lines.append(line)

            tokens = dictionary.encode_line(
                line,
                add_if_not_exist=False,
                append_eos=self.append_eos,
                reverse_order=self.reverse_order,
            ).long()
            self.tokens_list.append(tokens)
            self.sizes.append(len(tokens))

        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


@dataclass
class CBASConfig(FairseqDataclass):
    data: str = field(
        default=MISSING,
        metadata={
            "help": "colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner"
        },
    )
    dataset_impl: ChoiceEnum(get_available_dataset_impl()) = field(
        default="raw",
        metadata={"help": "data saving format"}
    )
    protein_task: str = field(
        default="avGFP",
        metadata={"help": "protein dataset"},
    )
    number_round_samples: int = field(
        default=10000,
        metadata={"help": "number of round samples"}
    )
    number_round_optimization: int = field(
        default=20,
        metadata={"help": "number of cbas round optimization"}
    )
    generation_round: int = field(
        default=10,
        metadata={"help": "number optimized generation in inference"}
    )
    generation_round_sample: int = field(
        default=1000,
        metadata={"help": "number of generated samples in each optimization round"}
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
    seed: int = II("common.seed")


@register_task("cbas", dataclass=CBASConfig)
class CBASTask(FairseqTask):

    cfg: CBASConfig

    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, cfg: CBASConfig, dictionary, starting_sequence):
        super().__init__(cfg)
        self.dictionary = dictionary
        self.mask_idx = dictionary.add_symbol("<mask>")
        self.landscape = ESM1b_Landscape(cfg)
        self.starting_sequence = starting_sequence
        self.number_round_samples = cfg.number_round_samples

    @classmethod
    def setup_task(cls, cfg: CBASConfig, **kwargs):
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))

        data_path = os.path.join(paths[0], cfg.protein_task)
        with open(os.path.join(data_path, 'starting_sequence.json')) as f:
            starting_sequence = json.load(f)

        return cls(cfg, dictionary, starting_sequence)

    def load_round_dataset(self, split, model):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        inference_dataset = self.build_dataset_for_inference([self.starting_sequence], [len(self.starting_sequence)])
        # print(inference_dataset[0]["net_input.src_tokens"])

        samples = self.inference_get_round_samples(model, inference_dataset[0])  # [number, length]
        fitness_values = self.landscape.get_fitness(samples)

        indexes = np.argsort(-np.array(fitness_values))[: int(self.number_round_samples/2)]
        final_samples = [samples[index] for index in indexes]
        dataset = SampleDataset(final_samples, self.dictionary)

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.cfg.shorten_data_split_list,
            self.cfg.shorten_method,
            self.cfg.tokens_per_sample,
            self.cfg.seed,
        )

        logger.info("generated {} samples".format(len(dataset)))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

        # create masked input and targets
        mask_whole_words = (
            get_whole_word_mask(self.args, self.source_dictionary)
            if self.cfg.mask_whole_words
            else None
        )

        src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
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

        self.datasets["train"] = SortDataset(
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

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, "{}.{}.txt".format(split, self.cfg.protein_task))

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.cfg.shorten_data_split_list,
            self.cfg.shorten_method,
            self.cfg.tokens_per_sample,
            self.cfg.seed,
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

        src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
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
        src_dataset = SampleDataset(src_tokens, self.dictionary)

        src_dataset = maybe_shorten_dataset(
            src_dataset,
            "test",
            self.cfg.shorten_data_split_list,
            self.cfg.shorten_method,
            self.cfg.tokens_per_sample,
            self.cfg.seed,
        )

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
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
        return src_dataset

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    def inference_get_round_samples(self, model, sample):
        masked_tokens = None
        src_tokens = sample["net_input.src_tokens"]
        src_tokens = src_tokens.reshape(1, -1)
        src_tokens = src_tokens.to("cuda")
        with torch.no_grad():
            logits = model(src_tokens, masked_tokens=masked_tokens)[0][0][1: -1].cpu()  # [T, C]
            logits = logits[:, 8: -1]
            probs = logits / torch.sum(logits, dim=1, keepdim=True)
            for i in range(len(probs)):
                probs[i][-1] = 1 - torch.sum(probs[i][: -1])
            probs = probs.numpy()

            original_sample = sample["net_input.src_tokens"]
            sequence = self.dictionary.string(original_sample, separator="")
            hypos = [sequence]
            for i in range(self.cfg.number_round_samples):
                hypo = [aa for aa in sequence]
                num = random.choice(range(1, 11))  # [1, 10]
                inds = np.random.choice(len(sequence), num)
                for ind in inds:
                    index = np.random.choice(20, 1, p=probs[ind], replace=False)[0]
                    hypo[ind] = self.dictionary[index+8]
                hypo = "".join(hypo)
                hypos.append(hypo)    # [number, length]
        return hypos

    def inference_protein_one_step(self, model, sample):
        masked_tokens = None
        src_tokens = sample["net_input.src_tokens"]
        src_tokens = src_tokens.reshape(1, -1)
        src_tokens = src_tokens.to("cuda")
        with torch.no_grad():
            logits = model(src_tokens, masked_tokens=masked_tokens)[0][0][1: -1].cpu()  # [T, C]
            logits = logits[:, 8: -1]
            probs = logits / torch.sum(logits, dim=1, keepdim=True)
            for i in range(len(probs)):
                probs[i][-1] = 1 - torch.sum(probs[i][: -1])
            probs = probs.numpy()

            original_sample = sample["net_input.src_tokens"]
            sequence = self.dictionary.string(original_sample, separator="")
            hypos = [sequence]
            for i in range(self.cfg.generation_round_sample):
                hypo = [aa for aa in sequence]
                num = random.choice(range(1, 21))  # [1, 20]
                inds = np.random.choice(len(sequence), num)
                for ind in inds:
                    index = np.random.choice(20, 1, p=probs[ind], replace=False)[0]
                    hypo[ind] = self.dictionary[index+8]
                hypo = "".join(hypo)
                hypos.append(hypo)    # [number, length]
        return hypos

    def inference_protein(self, model):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        seqs = [self.starting_sequence]
        final_seqs = []
        final_fits = []
        for i in range(self.cfg.generation_round):
            inference_dataset = self.build_dataset_for_inference(seqs, [len(seqs[0])])

            samples = self.inference_protein_one_step(model, inference_dataset[0])  # [number, length]
            fitness_values = self.landscape.get_fitness(samples)
            indexes = np.argsort(-np.array(fitness_values))
            print("round = %d, max fitness=%f, sample=%s" % (i+1, fitness_values[indexes[0]], samples[indexes[0]]))

            # final_seqs = [samples[index] for index in indexes[: 10]]
            # final_fits = [fitness_values[index] for index in indexes[: 10]]
            seqs = [samples[indexes[0]]]
            final_seqs.extend(samples)
            final_fits.extend(fitness_values)

        final_indexes = np.argsort(-np.array(final_fits))
        generate_seqs = [final_seqs[index] for index in final_indexes[: 128]]
        generate_fits = [final_fits[index] for index in final_indexes[: 128]]
        return generate_seqs, generate_fits



