# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import itertools
import json
import logging
import os
from typing import Optional
from argparse import Namespace
from omegaconf import II
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from fairseq import metrics, utils
from fairseq.data import (
    AppendTokenDataset,
    FragmentProteinDataset,
    SurfaceProteinDataset,
    ControlledSurfaceProteinDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.models.esm_modules import Alphabet
from fairseq.models.geometric_protein_model import get_edges_batch


EVAL_BLEU_ORDER = 4
device = torch.device("cuda")


logger = logging.getLogger(__name__)


def load_protein_dataset(
    data_path,
    split,
    src_protein,
    src_dict,
    dataset_impl_source,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    truncate_source=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
):
    def split_exists(split, dataset_impl):
        filename = os.path.join(data_path, "{}.seq.txt".format(split))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    # src_datasets = []
    # tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, dataset_impl_source):
            prefix = os.path.join(data_path, "{}.".format(split_k))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        coor_dataset = data_utils.load_indexed_dataset(
            prefix + "coor.txt", dataset_impl='surf_coor', source=True
        )
        # atom_dataset = data_utils.load_indexed_dataset(
        #     prefix + "atom.txt", dataset_impl='surf_atom', source=True
        # )
        aa_dataset = data_utils.load_indexed_dataset(
            prefix + "atom.txt", dataset_impl='surf_aa', source=True
        )
        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + "seq.txt", src_dict, "ss", source=False
        )
        # label_dataset = data_utils.load_indexed_dataset(
        #     prefix + "label.txt", src_dict, "label", source=False
        # )
        assert len(coor_dataset) == len(tgt_dataset)

        logger.info(
            "{} {} {} {} examples".format(
                data_path, split_k, src_protein, len(tgt_dataset)
            )
        )

    return SurfaceProteinDataset(
        src_dict,
        aa_dataset,
        aa_dataset.sizes,
        coor_dataset,
        coor_dataset.sizes,
        tgt_dataset,
        tgt_dataset.sizes,
        label_dataset=None,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=None,
        eos=src_dict.eos_idx,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


@dataclass
class FragmentProteinDesignConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    protein_task: str = field(
        default="avGFP",
        metadata={"help": "protein task name"}
    )
    source_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "source language",
            "argparse_alias": "-s",
        },
    )
    target_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "target language",
            "argparse_alias": "-t",
        },
    )
    load_alignments: bool = field(
        default=False, metadata={"help": "load the binarized alignments"}
    )
    left_pad_source: bool = field(
        default=False, metadata={"help": "pad the source on the left"}
    )
    left_pad_target: bool = field(
        default=False, metadata={"help": "pad the target on the left"}
    )
    max_source_positions: int = field(
        default=10240, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    upsample_primary: int = field(
        default=-1, metadata={"help": "the amount of upsample primary dataset"}
    )
    truncate_source: bool = field(
        default=False, metadata={"help": "truncate source to max-source-positions"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={
            "help": "if >0, then bucket source and target lengths into "
            "N buckets and pad accordingly; this is useful on TPUs to minimize the number of compilations"
        },
    )
    train_subset: str = II("dataset.train_subset")
    dataset_impl_source: Optional[ChoiceEnum(get_available_dataset_impl())] = field(
        default="raw", metadata={"help": "data format of source data"}
    )
    dataset_impl_target: Optional[ChoiceEnum(get_available_dataset_impl())] = field(
        default="coor", metadata={"help": "data format of target data"}
    )
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")

    # options for reporting BLEU during validation
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_bleu_args: Optional[str] = field(
        default="{}",
        metadata={
            "help": 'generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_bleu_detok: str = field(
        default="space",
        metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; "
            "use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_bleu_detok_args: Optional[str] = field(
        default="{}",
        metadata={"help": "args for building the tokenizer, if needed, as JSON string"},
    )
    eval_tokenized_bleu: bool = field(
        default=False, metadata={"help": "compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_remove_bpe: Optional[str] = field(
        default=None,
        metadata={
            "help": "remove BPE before computing BLEU",
            "argparse_const": "@@ ",
        },
    )
    eval_bleu_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )
    eval_aa_recovery: bool = field(
        default=False, metadata={
            "help": "evaluate amino acid recovery or not"
        }
    )


@register_task("fragment_protein_design", dataclass=FragmentProteinDesignConfig)
class FragmentProteinDesignTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: FragmentProteinDesignConfig

    def __init__(self, cfg: FragmentProteinDesignConfig, src_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = src_dict
        self.mask_idx = self.src_dict.mask_idx

    @classmethod
    def setup_task(cls, cfg: FragmentProteinDesignConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).
        the dictionary is composed of amino acids

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0

        # load dictionaries
        alphabet = Alphabet.from_architecture("ESM-1b")
        src_dict = alphabet
        return cls(cfg, src_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]
        protein_task = self.cfg.protein_task

        # infer langcode
        # src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_protein_dataset(
            data_path,
            split,
            protein_task,
            self.src_dict,
            dataset_impl_source=self.cfg.dataset_impl_source,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constrains=None):
        return FragmentProteinDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            constraints=constrains
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        if self.cfg.eval_bleu:
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        if self.cfg.eval_aa_recovery:
            # greedy decoding
            indexes = model.forward_inference(**sample["net_input"])   # [batch, length]
            srcs = [self.tgt_dict.string(sample["target"]["aa"][i]) for i in range(sample["target"]["aa"].size(0))]
            strings = [self.tgt_dict.string(indexes[i]) for i in range(len(indexes))]

            # sampling
            # indexes = model.forward_inference_sampling(**sample["net_input"]) # [10, batch, length]
            # srcs = [self.tgt_dict.string(sample["target"]["aa"][i]) for i in range(sample["target"]["aa"].size(0))]
            # strings = [[self.tgt_dict.string(indexes[i][j]) for j in range (len(indexes[i]))] for i in range(len(indexes))]
            return loss, sample_size, logging_output, strings, srcs
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.eval_bleu:

            def sum_logs(key):
                import torch
                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    try:
                        from sacrebleu.metrics import BLEU
                        comp_bleu = BLEU.compute_bleu
                    except ImportError:
                        # compatibility API for sacrebleu 1.x
                        import sacrebleu
                        comp_bleu = sacrebleu.compute_bleu

                    fn_sig = inspect.getfullargspec(comp_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = comp_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (10240, 1024)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def build_generator(self, models, args, **unused):
        # add models input to match the API for SequenceGenerator
        from fairseq.iterative_refinement_generator import IterativeRefinementGenerator

        return IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
            max_iter=getattr(args, "iter_decode_max_iter", 10),
            beam_size=getattr(args, "iter_decode_with_beam", 1),
            reranking=getattr(args, "iter_decode_with_external_reranker", False),
            decoding_format=getattr(args, "decoding_format", None),
            adaptive=not getattr(args, "iter_decode_force_max_iter", False),
            retain_history=getattr(args, "retain_iter_history", False),
        )

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])
