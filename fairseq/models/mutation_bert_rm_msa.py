# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
    SinusoidalPositionalEmbedding,
    TransformerSentenceEncoder,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params


logger = logging.getLogger(__name__)


@register_model("mutation_bert_wo_msa")
class MutationBertNoMSA(FairseqEncoderModel):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        # if specified then apply bert initialization on the model. We need
        # to explictly call this to make sure that the output embeddings
        # and projection layers are also correctly initialized
        if getattr(args, "apply_bert_init", False):
            self.apply(init_bert_params)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # Arguments related to dropout
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for" " attention weights",
        )
        parser.add_argument(
            "--act-dropout",
            type=float,
            metavar="D",
            help="dropout probability after" " activation in FFN",
        )

        # Arguments related to hidden states and self-attention
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )

        # Arguments related to input and output embeddings
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--share-encoder-input-output-embed",
            action="store_true",
            help="share encoder input" " and output embeddings",
        )
        parser.add_argument(
            "--encoder-learned-pos",
            action="store_true",
            help="use learned positional embeddings in the encoder",
        )
        parser.add_argument(
            "--no-token-positional-embeddings",
            action="store_true",
            help="if set, disables positional embeddings" " (outside self attention)",
        )
        parser.add_argument(
            "--num-segment", type=int, metavar="N", help="num segment in the input"
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn"
        )

        # Arguments related to sentence level prediction
        parser.add_argument(
            "--sentence-class-num",
            type=int,
            metavar="N",
            help="number of classes for sentence task",
        )
        parser.add_argument(
            "--sent-loss",
            action="store_true",
            help="if set," " calculate sentence level predictions",
        )

        # Arguments related to parameter initialization
        parser.add_argument(
            "--apply-bert-init",
            action="store_true",
            help="use custom param initialization for BERT",
        )

        # misc params
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="Which activation function to use for pooler layer.",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument("--gumbel-temperature", type=float, default=0.1,
                            help="the temperature in gumbel softmax computation")

    def forward(self, src_tokens, segment_labels=None, **kwargs):
        """
        Returns:
            - a tuple of the following:
                - logits for predictions in format B x T x C to be used in
                  softmax afterwards
                - a dictionary of additional data, where 'pooled_output' contains
                  the representation for classification_token and 'inner_states'
                  is a list of internal model states used to compute the
                  predictions (similar in ELMO). 'sentence_logits'
                  is the prediction logit for NSP task and is only computed if
                  this is specified in the input arguments.
        """
        return self.encoder(src_tokens, segment_labels=segment_labels, **kwargs)

    def max_positions(self):
        return self.encoder.max_positions

    def get_sources(self, sample):
        """Get targets from either the sample or the net's output."""
        return sample["net_input"]["src_tokens"]

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        logger.info(args)

        encoder = MutationBertNoMSAEncoder(args, task.dictionary)
        return cls(args, encoder)


class MutationBertNoMSAEncoder(FairseqEncoder):
    """
    Encoder for Masked Language Modelling.
    """

    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        self.padding_idx = dictionary.pad()
        self.mask_index = dictionary.unk_index
        self.vocab_size = dictionary.__len__()
        self.max_positions = args.max_positions

        self.sentence_encoder = TransformerSentenceEncoder(
            padding_idx=self.padding_idx,
            vocab_size=self.vocab_size,
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            max_seq_len=self.max_positions,
            num_segments=args.num_segment,
            use_position_embeddings=not args.no_token_positional_embeddings,
            encoder_normalize_before=args.encoder_normalize_before,
            apply_bert_init=args.apply_bert_init,
            activation_fn=args.activation_fn,
            learned_pos_embedding=args.encoder_learned_pos,
        )

        self.share_input_output_embed = args.share_encoder_input_output_embed
        self.embed_out = None
        self.sentence_projection_layer = None
        self.sentence_out_dim = args.sentence_class_num
        self.lm_output_learned_bias = None

        # Remove head is set to true during fine-tuning
        self.load_softmax = not getattr(args, "remove_head", False)

        self.lm_head_transform_weight = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.activation_fn = utils.get_activation_fn(args.activation_fn)
        self.layer_norm = LayerNorm(args.encoder_embed_dim)

        self.lm_output_learned_bias = None
        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(self.vocab_size))

            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(
                    args.encoder_embed_dim, self.vocab_size, bias=False
                )

        self.mutation_pos = nn.Linear(args.encoder_embed_dim, 2, bias=False)  # [if a amino acid at a position would change]
        self.gumbel_temp = args.gumbel_temperature
        self.fit_embeds = nn.Embedding(2, args.encoder_embed_dim)   # [2, dim]

    def forward(self, src_tokens, segment_labels=None, masked_tokens=None, **unused):
        """
        Forward pass for Masked LM encoder. This first computes the token
        embedding using the token embedding matrix, position embeddings (if
        specified) and segment embeddings (if specified).

        Here we assume that the sentence representation corresponds to the
        output of the classification_token (see bert_task or cross_lingual_lm
        task for more details).
        Args:
            - src_tokens: B x T matrix representing sentences
            - segment_labels: B x T matrix representing segment label for tokens
        Returns:
            - a tuple of the following:
                - logits for predictions in format B x T x C to be used in
                  softmax afterwards
                - a dictionary of additional data, where 'pooled_output' contains
                  the representation for classification_token and 'inner_states'
                  is a list of internal model states used to compute the
                  predictions (similar in ELMO). 'sentence_logits'
                  is the prediction logit for NSP task and is only computed if
                  this is specified in the input arguments.
        """
        fitness_labels = src_tokens[:, 1]  # [batch]
        fitness_labels = -fitness_labels + 7

        src_tokens = torch.cat((src_tokens[:, 0].unsqueeze(1), src_tokens[:, 2: ]), 1)

        fits = self.fit_embeds(fitness_labels).unsqueeze(1)  # [length, 1, dim]
        inner_states, sentence_rep = self.sentence_encoder(
            src_tokens,
            segment_labels=segment_labels,
            fits_embeddings=fits
        )

        x = inner_states[-1].transpose(0, 1)  # [B, T, C]
        # project masked tokens only
        if masked_tokens is not None:
            x = x[masked_tokens, :]
        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))


        hidden_states = x

        pos_logits = self.mutation_pos(x)  # [B, T, 2]
        # pos_prob = nn.functional.softmax(pos_logits, dim=-1)  # [B, T, 2] to apply gumble-softmax
        # gumbel_prob = utils.apply_gumvel_softmax(pos_prob, self.gumbel_temp)  # [B, T, 2]
        gumbel_prob = F.gumbel_softmax(pos_logits, tau=self.gumbel_temp)
        mask_index = torch.tensor([[self.mask_index]], device=x.device)  # [[1]]
        new_x = gumbel_prob[:, :, 1].unsqueeze(2) * self.sentence_encoder.embed_tokens(mask_index) + gumbel_prob[:, :, 0].unsqueeze(2) * hidden_states
        x = new_x

        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(
            self.sentence_encoder.embed_tokens, "weight"
        ):
            x = F.linear(x, self.sentence_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)
        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias

        final_output = F.softmax(x, dim=-1)
        return final_output, {
            "inner_states": inner_states,
            "position_logits": pos_logits
        }

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        if isinstance(
            self.sentence_encoder.embed_positions, SinusoidalPositionalEmbedding
        ):
            state_dict[
                name + ".sentence_encoder.embed_positions._float_tensor"
            ] = torch.FloatTensor(1)
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if (
                    "embed_out.weight" in k
                    or "sentence_projection_layer.weight" in k
                    or "lm_output_learned_bias" in k
                ):
                    del state_dict[k]
        return state_dict


@register_model_architecture("mutation_bert_wo_msa", "mutation_bert_wo_msa_base")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.act_dropout = getattr(args, "act_dropout", 0.0)

    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", True
    )
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.num_segment = getattr(args, "num_segment", 2)

    args.sentence_class_num = getattr(args, "sentence_class_num", 2)
    args.sent_loss = getattr(args, "sent_loss", False)

    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)


@register_model_architecture("mutation_bert_wo_msa", "mutation_bert_wo_msa_tiny")
def mutation_bert_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", True
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.num_segment = getattr(args, "num_segment", 2)

    args.encoder_layers = getattr(args, "encoder_layers", 6)

    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)

    args.sentence_class_num = getattr(args, "sentence_class_num", 2)
    args.sent_loss = getattr(args, "sent_loss", False)

    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    base_architecture(args)


@register_model_architecture("mutation_bert_wo_msa", "mutation_bert_wo_msa_base_base")
def mutation_bert_base_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", True
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.num_segment = getattr(args, "num_segment", 2)

    args.encoder_layers = getattr(args, "encoder_layers", 12)

    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)

    args.sentence_class_num = getattr(args, "sentence_class_num", 2)
    args.sent_loss = getattr(args, "sent_loss", True)

    args.apply_bert_init = getattr(args, "apply_bert_init", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    base_architecture(args)


@register_model_architecture("mutation_bert_wo_msa", "mutation_bert_wo_msa_large")
def mutation_bert_large_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    mutation_bert_base_architecture(args)


@register_model_architecture("mutation_bert_wo_msa", "mutation_bert_wo_msa_xlm_base")
def mutation_xlm_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", True
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.num_segment = getattr(args, "num_segment", 1)

    args.encoder_layers = getattr(args, "encoder_layers", 6)

    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)

    args.sent_loss = getattr(args, "sent_loss", False)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.apply_bert_init = getattr(args, "apply_bert_init", True)
    base_architecture(args)
