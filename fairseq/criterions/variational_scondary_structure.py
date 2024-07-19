# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
import numpy as np

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy_vae import VAEProteinDesignCriterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class VAESSCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    kl_factor: float = field(
        default=0.5,
        metadata={"help": "importance factor for kl divergence"}
    )
    final_kl: float = field(
        default=0.5,
        metadata={"help": "importance factor for kl divergence"}
    )


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):

    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("variational_secondary_structure_criterion", dataclass=VAESSCriterionConfig)
class VariationalSSProteinDesignCriterion(VAEProteinDesignCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        kl_factor=0.5,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy, kl_factor)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        decoder_out_ss, z_mean_ss, z_std_ss, decoder_out_seq, z_mean_seq, z_std_seq = model(**sample["net_input"])

        ss_target = sample["output"]["ss_output"]
        loss_ss, nll_loss_ss = self.compute_loss(model, decoder_out_ss, ss_target, reduce=reduce)

        seq_target = sample["output"]["seq_output"]
        loss_seq, nll_loss_seq = self.compute_loss(model, decoder_out_seq, seq_target, reduce=reduce)

        kl_loss_ss = - torch.sum(1 + torch.log(torch.square(z_std_ss)) - torch.square(z_mean_ss) - torch.square(z_std_ss)) * 0.5
        kl_loss_seq = - torch.sum(
            1 + torch.log(torch.square(z_std_seq)) - torch.square(z_mean_seq) - torch.square(z_std_seq))

        final_loss = loss_ss + loss_seq + self.kl_factor * (kl_loss_ss + kl_loss_seq)

        sample_size = (
            sample["output"]["ss_output"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": final_loss.data,
            "ss_entropy_loss": loss_ss.data,
            "seq_entropy_loss": loss_seq.data,
            "ss_kl_loss": kl_loss_ss.data,
            "seq_kl_loss": kl_loss_seq.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["output"]["ss_output"].size(0),
            "sample_size": sample_size,
        }
        return final_loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, target, reduce=True):
        lprobs = F.log_softmax(net_output, dim=-1, dtype=torch.float32)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce
        )
        return loss, nll_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ss_entropy_loss = sum(log.get("ss_entropy_loss", 0) for log in logging_outputs)
        seq_entropy_loss = sum(log.get("seq_entropy_loss", 0) for log in logging_outputs)
        ss_kl_loss = sum(log.get("ss_kl_loss", 0) for log in logging_outputs)
        seq_kl_loss = sum(log.get("seq_kl_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        weights = sum(log.get("average_weights", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ss_entropy_loss", ss_entropy_loss / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "seq_entropy_loss", seq_entropy_loss / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "ss_kl_loss", ss_kl_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "seq_kl_loss", seq_kl_loss / sample_size / math.log(2), sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
