import random

from sru import SRUpp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models.transformer import TransformerEncoder


class FrameAveraging(nn.Module):
    def __init__(self):
        super(FrameAveraging, self).__init__()
        self.ops = torch.tensor([
            [i, j, k] for i in [-1, 1] for j in [-1, 1] for k in [-1, 1]
        ])

    def create_frame(self, X, mask):
        mask = mask.unsqueeze(-1)
        center = (X * mask).sum(dim=1) / mask.sum(dim=1)
        X = X - center.unsqueeze(1) * mask  # [B, N, 3]
        C = torch.bmm(X.transpose(1, 2), X)  # [B, 3, 3] (Cov)
        L, V = torch.linalg.eigh(C.float().detach(), UPLO='U')  # [B,3,3]
        # _, V = torch.symeig(C.detach(), True)  # [B,3,3]
        F_ops = self.ops.unsqueeze(1).unsqueeze(0).to(X.device) * V.unsqueeze(1)  # [1,8,1,3] x [B,1,3,3] -> [B,8,3,3]
        h = torch.einsum('boij,bpj->bopi', F_ops.transpose(2, 3), X)  # transpose is inverse [B,8,N,3]
        h = h.view(X.size(0) * 8, X.size(1), 3)
        return h, F_ops.detach(), center

    def invert_frame(self, X, mask, F_ops, center):
        X = torch.einsum('boij,bopj->bopi', F_ops, X)
        X = X.mean(dim=1)  # frame averaging
        X = X + center.unsqueeze(1)
        return X * mask.unsqueeze(-1)


class FAEncoder(FrameAveraging):
    def __init__(self, input_size, hidden_dim, n_layers, n_heads, dropout, bidirectional=True, encoder_type='sru'):
        super(FAEncoder, self).__init__()

        self.encoder_type = encoder_type

        if encoder_type == 'sru':
            self.encoder = SRUpp(
                input_size=input_size,
                hidden_size=hidden_dim // 2,
                proj_size=hidden_dim // 2,
                num_layers=n_layers,
                dropout=dropout,
                bidirectional=bidirectional,
            ).float().cuda()  # .cuda() is required...no idea why
        elif encoder_type == 'transformer':
            self.encoder = TransformerEncoder(
                input_size=input_size,
                hidden_size=hidden_dim,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
            )
        else:
            raise NotImplementedError

        self.encoder.reset_parameters()

    def forward(self, input):
        # X: [B, N, 14, 3]
        S, X, A, h_S = input

        B, N = X.shape[0], X.shape[1]
        mask = X.sum(dim=-1) != 0
        if len(X.shape) == 4:
            X = X[:, :, 0]  # [B, N, 3]

        h, _, _ = self.create_frame(X, mask)  # [B*8, N, 3]
        mask = mask.unsqueeze(1).expand(-1, 8, -1).reshape(B * 8, N)
        # mask = mask.unsqueeze(1).expand(-1, 1, -1).reshape(B * 1, N)
        if h_S is not None:
            h_S = h_S.unsqueeze(1).expand(-1, 8, -1, -1).reshape(B * 8, N, -1)
            # h_S = h_S.unsqueeze(1).expand(-1, 1, -1, -1).reshape(B * 1, N, -1)
            h = torch.cat([h, h_S], dim=-1)

        if self.encoder_type == 'sru':
            h, _, _ = self.encoder(
                h.transpose(0, 1),
                mask_pad=(~mask.transpose(0, 1).bool())
            )
            h = h.transpose(0, 1)
        elif self.encoder_type == 'transformer':
            h, _ = self.encoder(
                h.float(),
                input_masks=mask.bool(),
            )

        return h.view(B, 8, N, -1).mean(dim=1)  # frame averaging
        # return h.view(B, 1, N, -1).mean(dim=1)  # frame averaging


# if __name__ == "__main__":
#     B, L = 3, 16
#     coords = torch.rand(B, L, 1, 3).cuda()320
#     esm_embeddings = torch.rand(B, L, 512).cuda()
#     prot_seq, A = None, None
#
#     model = FAEncoder(512 + 3, 128, 2, 4, 0.1, bidirectional=True, encoder_type='sru')
#     out = model((None, coords, A, esm_embeddings))  # [B, L, H]
#     print(out.size())