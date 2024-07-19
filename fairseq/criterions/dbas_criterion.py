# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import math
import numpy as np
from omegaconf import II

import torch
import torch.nn.functional as F
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class DBASCriterionConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")


@register_criterion("dbas_criterion", dataclass=DBASCriterionConfig)
class MaskedLmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, cfg: DBASCriterionConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        targets = model.get_targets(sample, None)   # [B, T]
        masked_tokens = targets.ne(self.padding_idx)
        sample_size = masked_tokens.int().sum()

        if self.tpu:
            masked_tokens = None  # always project all tokens on TPU
        elif masked_tokens.device == torch.device("cpu"):
            if not masked_tokens.any():
                masked_tokens = None
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )

        logits = model(**sample["net_input"], masked_tokens=masked_tokens)[0]
        logits = torch.log(logits)   # [B, T, C]
        # if weights is not None:
        #     logits = weights * logits

        # if masked_tokens is not None:
        #     targets = targets[masked_tokens]

        # loss = modules.cross_entropy(
        #     logits.view(-1, logits.size(-1)),
        #     targets.view(-1),
        #     reduction="sum",
        #     ignore_index=self.padding_idx,
        # )
        loss = F.nll_loss(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.padding_idx,
            reduction="sum",
        )

        logging_output = {
            "loss": loss if self.tpu else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
