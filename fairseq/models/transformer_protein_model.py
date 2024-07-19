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
import torch.nn.functional as F

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


class TransformerProteinDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, output_projection=None):
        self.args = args
        super().__init__(
            args,
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            latent_vectors: [length, batch, dim]

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()

        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                    enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_tokens(prev_output_tokens)
        x = self.embed_scale * x

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        extra = {"attn": [attn], "inner_states ": inner_states}

        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def inference_forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            latent_vectors: [length, batch, dim]

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()

        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                    enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_tokens(prev_output_tokens)
        x = self.embed_scale * x
        # x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        extra = {"attn": [attn], "inner_states": inner_states}

        if not features_only:
            x = self.output_layer(x)
        return x, extra


@register_model("transformer_protein_model")
class TransformerProteinModel(TransformerModel):
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

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.decoder.embed_tokens = self.encoder.embed_tokens

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
        return model

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerProteinDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=args.no_cross_attention,
        )

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
        # sequence representations
        encoder_outs = self.encoder(src_tokens, repr_layers=[self.encoder_layers])
        encoder_out = encoder_outs["representations"][self.encoder_layers]  # [batch, src_length, dim]

        # decoding secondary structure
        encoder_outs["encoder_out"] = [encoder_out.transpose(0, 1)]
        encoder_outs["encoder_padding_mask"] = []

        decoder_out, _ = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_outs,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )  # logits before softmax
        decoder_out = F.log_softmax(decoder_out, dim=-1)
        return decoder_out

    def forward_inference(
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
        # sequence representations
        encoder_outs = self.encoder(src_tokens, repr_layers=[self.encoder_layers])
        encoder_out = encoder_outs["representations"][self.encoder_layers]  # [batch, src_length, dim]

        # decoding secondary structure
        encoder_outs["encoder_out"] = [encoder_out.transpose(0, 1)]
        encoder_outs["encoder_padding_mask"] = []

        max_length = max(src_lengths)
        prev_output_tokens = prev_output_tokens[:, : 1]
        for i in range(max_length-2):
            decoder_out, _ = self.decoder.inference_forward(
                prev_output_tokens,
                encoder_out=encoder_outs,
                features_only=features_only,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
                src_lengths=src_lengths,
                return_all_hiddens=return_all_hiddens
            )  # before softmax
            this_step = decoder_out[:, -1, :]
            # _, top_indices = torch.topk(this_step, k=5, dim=-1)
            # index_selects = torch.tensor(np.random.randint(low=0, high=5, size=(this_step.size(0)))).to(device).unsqueeze(-1)
            # indexes = top_indices.gather(index=index_selects, dim=-1).view(-1, 1)
            indexes = torch.argmax(this_step, dim=-1).view(-1, 1)  # [batch]
            prev_output_tokens = torch.cat((prev_output_tokens, indexes), 1)

        return prev_output_tokens


@register_model_architecture("transformer_protein_model", "transformer_protein_model")
def base_architecture(args):
    transformer_base_architecture(args)


@register_model_architecture("transformer_protein_model", "transformer_protein_model_base")
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


@register_model_architecture("transformer_protein_model", "transformer_protein_model_tiny")
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


@register_model_architecture("transformer_protein_model", "transformer_protein_model_esm")
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



