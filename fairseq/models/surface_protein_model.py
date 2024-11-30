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
import math

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.distributions as dist

from sklearn.neighbors import NearestNeighbors

from fairseq import checkpoint_utils
from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
    base_architecture as transformer_base_architecture
)
from fairseq.models.egnn import EGNN
from fairseq.models.FA_encoder import FAEncoder


device = torch.device("cuda")


# def read_atom_dict():
#     atom_list = ["C", "CA", "N", "O", "CB", "<unk>"]
#     atom2index = dict()
#     index2atom = []
#     for atom in atom_list:
#         atom2index[atom] = len(atom2index)
#         index2atom.append(atom)
#     return atom2index, index2atom


def read_atom_dict():
    lines = open(
        "dict.txt",
        "r", encoding="utf-8").readlines()
    atom2index = dict()
    index2atom = []
    for line in lines:
        phrases = line.strip().split()
        atom = phrases[0].strip()

        if atom not in atom2index:
            atom2index[atom] = len(atom2index)
            index2atom.append(atom)
    return atom2index, index2atom


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def get_surface_aa_feature():
    HYDROPATHY = {"I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8, "W": -0.9, "G": -0.4,
                  "T": -0.7, "S": -0.8, "Y": -1.3, "P": -1.6, "H": -3.2, "N": -3.5, "D": -3.5, "Q": -3.5, "E": -3.5,
                  "K": -3.9, "R": -4.5}  # *
    # VOLUME = {'#': 0, "G": 60.1, "A": 88.6, "S": 89.0, "C": 108.5, "D": 111.1, "P": 112.7, "N": 114.1, "T": 116.1,
    #           "E": 138.4, "V": 140.0, "Q": 143.8, "H": 153.2, "M": 162.9, "I": 166.7, "L": 166.7, "K": 168.6,
    #           "R": 173.4, "F": 189.9, "Y": 193.6, "W": 227.8}
    CHARGE = {**{'R': 1, 'K': 1, 'D': -1, 'E': -1, 'H': 0.1}, **{x: 0 for x in 'ABCFGIJLMNOPQSTUVWXYZ'}} # *
    POLARITY = {**{x: 1 for x in 'RNDQEHKSTY'}, **{x: 0 for x in "ACGILMFPWV"}}
    ACCEPTOR = {**{x: 1 for x in 'DENQHSTY'}, **{x: 0 for x in "RKWACGILMFPV"}}
    DONOR = {**{x: 1 for x in 'RKWNQHSTY'}, **{x: 0 for x in "DEACGILMFPV"}}
    # PMAP = lambda x: [HYDROPATHY[x] / 5, CHARGE[x], POLARITY[x], ACCEPTOR[x], DONOR[x]]
    PMAP = lambda x: [HYDROPATHY[x] / 5, CHARGE[x]]

    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    aa_features = []
    for aa in alphabet:
        aa_features.append(PMAP(aa))
    return torch.Tensor(np.array(aa_features)).to(device)  # [20, 2]


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
            aa_identity_features: identity about if a residue is on the surface or not; size [batch, length, 2]
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


def get_edges(n_nodes, k, indices):
    rows, cols = [], []

    for i in range(n_nodes):
        for j in range(k):
            rows.append(i)
            cols.append(indices[i][j+1])

    edges = [rows, cols]   # L * 30
    return edges


def get_edges_batch(n_nodes, batch_size, coords, k=30):
    rows, cols = [], []
    # batch = torch.tensor(range(batch_size)).reshape(-1, 1).expand(-1, n_nodes).reshape(-1).to(device)
    # edges = knn_graph(coords, k=k, batch=batch, loop=False)
    # edges = edges[[1, 0]]

    for i in range(batch_size):
        # k = min(k, len(coords[i]))
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords[i])
        distances, indices = nbrs.kneighbors(coords[i])  # [N, 30]
        edges = get_edges(n_nodes, k, indices)  # [[N*N], [N*N]]
        edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
        rows.append(edges[0] + n_nodes * i)  # every sample in batch has its own graph
        cols.append(edges[1] + n_nodes * i)
    edges = [torch.cat(rows).to(device), torch.cat(cols).to(device)]  # B * L * 30
    return edges


@register_model("surface_protein_model")
class SurfaceProteinModel(TransformerModel):
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
            "--memory-dimension",
            type=int,
            default=0,
            help="dimension of additional memory vector for latent variable"
        )
        parser.add_argument(
            "--surface-representation-dimension",
            type=int,
            default=128
        )
        parser.add_argument(
            "--surface-hidden-layer",
            type=int,
            default=3
        )
        parser.add_argument(
            "--surface-hidden-dimension",
            type=int,
            default=128
        )
        parser.add_argument(
            "--gnn-layer",
            default=3,
            type=int)
        parser.add_argument(
            "--knn",
            type=int,
            default=30,
            help="number of k nearest neighbors",
        )

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.encoder_layers = args.encoder_layers
        self.surf_aa_features = get_surface_aa_feature()  # [20, 5]
        self.dictionary = self.decoder.dictionary
        self.mask_id = self.dictionary.mask()
        self.bos_id = self.dictionary.bos()
        self.eos_id = self.dictionary.eos()
        self.k = args.knn
        # self.pooling_encoder = FAEncoder(3, args.encoder_embed_dim, 2, 4, 0.1,
        #                                  bidirectional=True, encoder_type='sru')
        self.pooling_encoder = FAEncoder(args.encoder_embed_dim+3, args.encoder_embed_dim, 2, 4, 0.1,
                                         bidirectional=True, encoder_type='sru')

    @classmethod
    def build_model(self, args, task, cls_dictionary=MaskedLMDictionary):
        return super().build_model(args, task)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = EGNN(in_node_nf=2, hidden_nf=args.encoder_embed_dim, out_node_nf=3,
                       in_edge_nf=0, device=device, n_layers=args.decoder_layers, attention=True)
        # encoder = EGNN(in_node_nf=5, hidden_nf=args.encoder_embed_dim, out_node_nf=3,
        #                in_edge_nf=0, device=device, n_layers=args.decoder_layers, attention=True)
        return encoder
        # return FAEncoder(5+3, 128, 2, 4, 0.1, bidirectional=True, encoder_type='sru')

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
        coor_features,
        aa_features,
        src_lengths,
        prev_output_tokens,
        tgt_lengths,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.
        aa_identity_features: [B, L]
        coor_features: [B, L, 4, 3]
        prev_output_tokens: [B, L, N+1]
        attention_score: [B, tgt len, src len]

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        # get surface chemical feature
        bs, length = prev_output_tokens.size(0), prev_output_tokens.size(1)
        surf_aa_embs = torch.index_select(self.surf_aa_features, 0, aa_features.reshape(-1)).reshape(bs, -1, 2)
        coor_features = coor_features.view(coor_features.size(0), -1, 3)

        # sequence representations
        k = min(self.k, min(src_lengths).item()-1)

        edges = get_edges_batch(coor_features.size()[1], bs, coor_features.detach().cpu(), k)

        h, coor_features = self.encoder(surf_aa_embs, coor_features, edges, None, k)
        h = h.reshape(bs, -1, h.size()[-1])
        coor_features = coor_features.reshape(bs, -1, 3)
        h = self.pooling_encoder((None, coor_features, None, h))  # [B, L, H]

        # feature = torch.max(h, dim=1)[0]
        # decoder_out = torch.tanh(self.binder_indicator(feature))
        # decoder_out = F.log_softmax(output, dim=-1)  # [batch, 2]

        encoder_outs = {}
        # decoding secondary structure
        encoder_outs["encoder_out"] = [h.transpose(0, 1)]
        encoder_outs["encoder_padding_mask"] = []

        decoder_out, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_outs,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )  # logits before softmax
        decoder_out = F.log_softmax(decoder_out, dim=-1)  # [bs, length-1, vocab]
        return decoder_out

    def forward_inference(
            self,
            coor_features,
            aa_features,
            src_lengths,
            prev_output_tokens,
            tgt_lengths,
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
        bs, length = prev_output_tokens.size(0), prev_output_tokens.size(1)
        surf_aa_embs = torch.index_select(self.surf_aa_features, 0, aa_features.reshape(-1)).reshape(bs, -1, 2)
        surf_aa_embs = torch.zeros(surf_aa_embs.size()[0], surf_aa_embs.size()[1], surf_aa_embs.size()[2]).to(device)
        coor_features = coor_features.view(coor_features.size(0), -1, 3)

        # sequence representations
        k = min(self.k, min(src_lengths).item() - 1)

        edges = get_edges_batch(coor_features.size()[1], bs, coor_features.detach().cpu(), k)

        h, coor_features = self.encoder(surf_aa_embs, coor_features, edges, None, k)
        h = h.reshape(bs, -1, h.size()[-1])
        coor_features = coor_features.reshape(bs, -1, 3)
        h = self.pooling_encoder((None, coor_features, None, h))  # [B, L, H]

        encoder_outs = {}
        # decoding secondary structure
        encoder_outs["encoder_out"] = [h.transpose(0, 1)]
        encoder_outs["encoder_padding_mask"] = []

        max_length = prev_output_tokens.size(1)   # L + 1
        prev_output_tokens = prev_output_tokens[:, : 1]

        for i in range(max_length-1):
            decoder_out, extra = self.decoder.inference_forward(
                prev_output_tokens,
                encoder_out=encoder_outs,
                features_only=features_only,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
                src_lengths=src_lengths,
                return_all_hiddens=return_all_hiddens
            )  # before softmax
            this_step = decoder_out[:, -1, :]  # [batch, vocab]
            # _, top_indices = torch.topk(this_step, k=3, dim=-1)
            # # indexes = top_indices[:, -1].view(-1, 1)
            # index_selects = torch.tensor(np.random.randint(low=0, high=3, size=(this_step.size(0)))).to(device).unsqueeze(-1)
            # indexes = top_indices.gather(index=index_selects, dim=-1).view(-1, 1)
            indexes = torch.argmax(this_step, dim=-1).view(-1, 1)  # [batch]
            prev_output_tokens = torch.cat((prev_output_tokens, indexes), 1)

            # features = extra["features"]  # [batch, length, dim]
            # weight = F.gumbel_softmax(decoder_out, tau=0.1, hard=False)  # [batch, length, vocab, 1]
            # next_token_features = torch.sum(weight.unsqueeze(-1) * self.decoder.embed_tokens.weight.view(1, 1, 25, -1),
            #                                 dim=2)  # [vocab, dim]
            # surf_out = F.log_softmax(self.surf_mlp(torch.cat((features, next_token_features), -1)), dim=-1)
            # this_step_surf = surf_out[:, -1, :]
            # surf_indexes = torch.argmax(this_step_surf, dim=-1).view(-1, 1)
            # aa_identity_features = torch.cat((aa_identity_features, surf_indexes), 1)
        return prev_output_tokens


    def forward_inference_sampling(
            self,
            coor_features,
            aa_features,
            src_lengths,
            prev_output_tokens,
            tgt_lengths,
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
        bs, length = prev_output_tokens.size(0), prev_output_tokens.size(1)
        surf_aa_embs = torch.index_select(self.surf_aa_features, 0, aa_features.reshape(-1)).reshape(bs, -1, 2)
        coor_features = coor_features.view(coor_features.size(0), -1, 3)

        # sequence representations
        k = min(self.k, min(src_lengths).item() - 1)

        edges = get_edges_batch(coor_features.size()[1], bs, coor_features.detach().cpu(), k)

        h, coor_features = self.encoder(surf_aa_embs, coor_features, edges, None, k)
        h = h.reshape(bs, -1, h.size()[-1])
        coor_features = coor_features.reshape(bs, -1, 3)
        h = self.pooling_encoder((None, coor_features, None, h))  # [B, L, H]

        encoder_outs = {}
        # decoding secondary structure
        encoder_outs["encoder_out"] = [h.transpose(0, 1)]
        encoder_outs["encoder_padding_mask"] = []

        max_length = prev_output_tokens.size(1)   # L + 1
        # prev_output_tokens = prev_output_tokens[:, : 1]
        target_tokens = prev_output_tokens
        gens = []
        for _ in range(10):
            prev_output_tokens = target_tokens[:, : 1]
            for i in range(max_length-1):
                decoder_out, _ = self.decoder.inference_forward(
                    prev_output_tokens,
                    encoder_out=encoder_outs,
                    features_only=features_only,
                    alignment_layer=alignment_layer,
                    alignment_heads=alignment_heads,
                    src_lengths=src_lengths,
                    return_all_hiddens=return_all_hiddens
                )  # before softmax
                this_step = decoder_out[:, -1, :]  # [batch, vocab]
                # indexes = torch.argmax(this_step, dim=-1).unsqueeze(0)
                probs = torch.softmax(this_step/0.1, dim=-1)
                m = dist.Categorical(probs)
                indexes = m.sample((1,))
                # indexes = torch.multinomial(probs, 1)   # [batch, 1]
                prev_output_tokens = torch.cat((prev_output_tokens, indexes), 1)
            gens.append(prev_output_tokens) # [10, batch, length]
        return gens

    def max_positions(self):
        """Maximum length supported by the model."""
        return (10240, 1024)


@register_model_architecture("surface_protein_model", "surface_protein_model")
def base_architecture(args):
    transformer_base_architecture(args)


@register_model_architecture("surface_protein_model", "surface_protein_model_base")
def surface_model_base(args):
    args.surface_representation_layer = getattr(args, "surface_representation_layer", 3)
    args.surface_representation_dimension = getattr(args, "surface_representation_dimension", 128)
    args.surface_hidden_layer = getattr(args, "surface_hidden_layer", 3)
    args.surface_hidden_dimension = getattr(args, "surface_hidden_dimension", 128)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 128)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 256)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    base_architecture(args)






