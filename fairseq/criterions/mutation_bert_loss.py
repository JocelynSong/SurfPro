# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import math
from numpy.lib.utils import source
from omegaconf import II

import torch
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

import torch.nn.functional as F


@dataclass
class MutationBertConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")
    importance: float = field(
        default=1.0,
        metadata={"help": "the importance of the position prediction weight"}
    )


@register_criterion("mutation_bert_loss", dataclass=MutationBertConfig)
class MutationBertLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, cfg: MutationBertConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.importance = cfg.importance

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # masked_tokens = None

        outputs = model(**sample["net_input"], masked_tokens=None)
        # logits = F.log_softmax(outputs[0], dim=-1) #[B, T, C]
        logits = torch.log(outputs[0])  # [B, T, C]
        position_logits = F.log_softmax(outputs[1]["position_logits"], dim=-1) # [B, T, 2]

        targets = model.get_targets(sample, [logits])
        sources = model.get_sources(sample)
        targets = torch.cat((targets[:, 0].unsqueeze(1), targets[:, 2: ]), 1)
        sources = torch.cat((sources[:, 0].unsqueeze(1), sources[:, 2: ]), 1)

        masked_tokens = targets.ne(self.padding_idx)
        sample_size = masked_tokens.int().sum()

        if masked_tokens is not None:
            targets = targets[masked_tokens].reshape(targets.size(0), -1)
            sources = sources[masked_tokens].reshape(targets.size(0), -1)
            logits = logits[masked_tokens].reshape(logits.size(0), -1, logits.size(-1))
            position_logits = position_logits[masked_tokens].reshape(position_logits.size(0), -1, position_logits.size(-1))

        mutated_pos = (sources != targets)  # True for mutation, and False for not
        mutated_pos = mutated_pos.long().unsqueeze(-1) # [B, T, 1]

        if targets.dim() == logits.dim() -1:
            targets = targets.unsqueeze(-1)

        loss_pos = -position_logits.gather(dim=-1, index=mutated_pos).sum()
        loss_token = (-logits.gather(dim=-1, index=targets) * mutated_pos).sum()

        loss = loss_token + self.importance * loss_pos

        logging_output = {
            "loss": loss if self.tpu else loss.data,
            "loss_token": loss_token if self.tpu else loss_token.data,
            "loss_pos": loss_pos if self.tpu else loss_pos.data,
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
