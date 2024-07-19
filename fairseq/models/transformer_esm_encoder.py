# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os, re
from typing import Any, Dict, Optional, List
from pathlib import Path
import urllib
import warnings
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import checkpoint_utils
from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
    base_architecture as transformer_base_architecture,
)
from fairseq.models.esm import ESM2
from fairseq.models.esm_modules import Alphabet


device = torch.device("cuda")


def load_hub_workaround(url):
    try:
        data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
    except RuntimeError:
        # Pytorch version issue - see https://github.com/pytorch/pytorch/issues/43106
        fn = Path(url).name
        data = torch.load(
            f"{torch.hub.get_dir()}/checkpoints/{fn}",
            map_location="cpu",
        )
    except urllib.error.HTTPError as e:
        raise Exception(f"Could not load {url}, check if you specified a correct model name?")
    return data


def _has_regression_weights(model_name):
    """Return whether we expect / require regression weights;
    Right now that is all models except ESM-1v and ESM-IF"""
    return not ("esm1v" in model_name or "esm_if" in model_name)


def load_regression_hub(model_name):
    url = f"https://dl.fbaipublicfiles.com/fair-esm/regression/{model_name}-contact-regression.pt"
    regression_data = load_hub_workaround(url)
    return regression_data


def _load_model_and_alphabet_core_v2(model_data):
    def upgrade_state_dict(state_dict):
        """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
        prefixes = ["encoder.sentence_encoder.", "encoder."]
        pattern = re.compile("^" + "|".join(prefixes))
        state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
        return state_dict

    cfg = model_data["cfg"]["model"]
    state_dict = model_data["model"]
    state_dict = upgrade_state_dict(state_dict)
    alphabet = Alphabet.from_architecture("ESM-1b")
    model = ESM2(
        num_layers=cfg.encoder_layers,
        embed_dim=cfg.encoder_embed_dim,
        attention_heads=cfg.encoder_attention_heads,
        alphabet=alphabet,
        token_dropout=cfg.token_dropout,
    )
    return model, alphabet, state_dict


def load_from_pretrained_models(pretrained_model_name):
    def _download_model_and_regression_data(model_name):
        url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
        model_data = load_hub_workaround(url)
        if _has_regression_weights(model_name):
            regression_data = load_regression_hub(model_name)
        else:
            regression_data = None
        return model_data, regression_data

    model_data, regression_data = _download_model_and_regression_data(pretrained_model_name)

    if regression_data is not None:
        model_data["model"].update(regression_data["model"])

    # if pretrained_model_name.startswith("esm2"):
    #     model, alphabet, model_state = _load_model_and_alphabet_core_v2(model_data)
    # else:
    #     model, alphabet, model_state = _load_model_and_alphabet_core_v1(model_data)
    model, alphabet, model_state = _load_model_and_alphabet_core_v2(model_data)

    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())

    if regression_data is None:
        expected_missing = {"contact_head.regression.weight", "contact_head.regression.bias"}
        error_msgs = []
        missing = (expected_keys - found_keys) - expected_missing
        if missing:
            error_msgs.append(f"Missing key(s) in state_dict: {missing}.")
        unexpected = found_keys - expected_keys
        if unexpected:
            error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected}.")

        if error_msgs:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        if expected_missing - found_keys:
            warnings.warn(
                "Regression weights not found, predicting contacts will not produce correct results."
            )
    model.load_state_dict(model_state, strict=regression_data is not None)
    return model, alphabet


@register_model("transformer_esm_encoder")
class TransformerESMModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        """
        Add model-specific arguments to the parser.
        """
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--pretrained-esm-model",
            type=str,
            metavar="STR",
            help="XLM model to use for initializing transformer encoder and/or decoder",
        )
        parser.add_argument(
            "--latent-dimension",
            type=int,
            default=128,
            help="dimension of latent variables"
        )
        parser.add_argument(
            "--latent-sample-size",
            type=int,
            default=1,
            help="latent variable sample size for computing expectation"
        )
        parser.add_argument(
            "--memory-dimension",
            type=int,
            default=0,
            help="dimension of additional memory vector for latent variable"
        )
        parser.add_argument(
            "--max-length",
            type=int,
            default=237,
            help="max sequence length"
        )
        parser.add_argument(
            "--single-energy-file",
            type=str,
            default="",
            help="MRF single energy file"
        )
        parser.add_argument(
            "--pair-energy-file",
            type=str,
            default="",
            help="MRF pairwise energy file"
        )

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.encoder_layers = args.encoder_layers
        self.max_length = args.max_length

    @classmethod
    def build_model(self, args, task, cls_dictionary=MaskedLMDictionary):
        assert hasattr(args, "pretrained_esm_model"), (
            "You must specify a path for --pretrained-esm-model to use "
            "--arch transformer_from_pretrained_xlm"
        )
        return super().build_model(args, task)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        model, alphabet = load_from_pretrained_models(args.pretrained_esm_model)
        for param in model.parameters():
            param.requires_grad = False
        return model

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_outs = self.encoder(src_tokens, repr_layers=[self.encoder_layers])
        encoder_out = encoder_outs["representations"][self.encoder_layers]  # [batch, src_length, dim]
        encoder_outs["encoder_out"] = [encoder_out.transpose(0, 1)]
        # encoder_outs["encoder_out"] = [new_encoder_out]
        encoder_outs["encoder_padding_mask"] = []

        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_outs,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens
        )
        return decoder_out


@register_model_architecture("transformer_esm_encoder", "transformer_esm")
def base_architecture(args):
    transformer_base_architecture(args)


@register_model_architecture("transformer_esm_encoder", "transformer_esm_base")
def transformer_vae_base(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 512)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    base_architecture(args)


@register_model_architecture("transformer_esm_encoder", "transformer_esm_tiny")
def transformer_vae_tiny(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 512)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    base_architecture(args)


@register_model_architecture("transformer_esm_encoder", "transformer_esm_esm")
def transformer_vae_esm(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 320)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1280)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 320)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1280)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    base_architecture(args)

