# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import math
from omegaconf import II
import numpy as np

import torch
import torch.nn.functional as F
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class GeometricProteinDesignConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")
    encoder_factor: float = field(
        default=1.0,
        metadata={"help": "the importance of the encoder loss"}
    )
    decoder_factor: float = field(
        default=1.0,
        metadata={"help": "the importance of the decoder loss"}
    )


@register_criterion("geometric_protein_loss", dataclass=GeometricProteinDesignConfig)
class GeometricProteinLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, cfg: GeometricProteinDesignConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.encoder_factor = cfg.encoder_factor
        self.decoder_factor = cfg.decoder_factor

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        source_input = sample["source_input"]
        target_input = sample["target_input"]
        motif = sample["motif"]
        output_mask = motif["output"]
        sample_size = output_mask.int().sum()

        encoder_out, decoder_out, _ = model(source_input["src_tokens"],
                                            source_input["src_lengths"],
                                            target_input["target_coor"],
                                            motif)

        # encoder output should be logits
        loss_encoder = -torch.log(encoder_out.gather(dim=-1, index=source_input["src_tokens"].unsqueeze(-1)).squeeze(-1))
        loss_encoder = torch.mean(torch.sum(loss_encoder * output_mask, dim=-1))

        # decoder output should be the directly predicted mse loss
        target_coor = target_input["target_coor"]
        decoder_out = decoder_out.reshape(target_coor.size(0), -1, decoder_out.size(-1))
        loss_decoder = torch.mean(torch.sum(torch.sum(torch.square(decoder_out - target_coor), dim=-1) * output_mask, dim=-1))
        #loss_decoder = torch.sum(torch.sqrt(torch.sum(torch.square(decoder_out - target_coor), dim=-1)) * output_mask) / output_mask.int().sum()
        # loss = self.encoder_factor * loss_encoder + self.decoder_factor * loss_decoder
        loss = self.decoder_factor * loss_decoder
        # loss = loss_decoder
        # loss = loss_encoder * self.encoder_factor

        logging_output = {
            "loss": loss.data,
            "loss_encoder": loss_encoder.data,
            "loss_decoder": loss_decoder.data,
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_mean = np.mean([log.get("loss", 0).cpu() for log in logging_outputs])
        loss_encoder_mean = np.mean([log.get("loss_encoder").cpu() for log in logging_outputs])
        loss_decoder_mean = np.mean([log.get("loss_decoder").cpu() for log in logging_outputs])
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_mean, round=8
        )
        metrics.log_scalar(
            "sequence loss", loss_encoder_mean, round=8
        )
        metrics.log_scalar(
            "coordinate loss", loss_decoder_mean, round=8
        )
        metrics.log_scalar(
            "sample size", sample_size)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
