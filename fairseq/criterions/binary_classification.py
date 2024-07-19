# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class BinaryClassificationCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("binary_classification", dataclass=BinaryClassificationCriterionConfig)
class BinaryCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.criterion = nn.BCELoss()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        decoder_out = model(**sample["net_input"])
        # loss = self.criterion(decoder_out.squeeze(-1), sample["target"]["labels"].float())

        loss = self.compute_loss(decoder_out, sample, reduce=reduce)
        sample_size = sample["target"]["labels"].size(0)

        logging_output = {
            "loss": loss.data,
            "nsentences": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, decoder_out, sample, reduce=True):
        lprobs = decoder_out
        target = sample["target"]["labels"]
        nll_loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=3,
            reduction="sum" if reduce else "none",
        )

        # loss = (1.0 - 0.01 - eps_i) * nll_loss + eps_i * smooth_loss
        return nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("nsentences", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
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
