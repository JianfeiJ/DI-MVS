import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
import math


class DepthHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, scale=False):
        super(DepthHead, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x_d, act_fn=torch.tanh):
        out = self.conv2(self.relu(self.conv1(x_d)))
        return act_fn(out)


class ConvGRU_new(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU_new, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, h, cz, cr, cq, *x_list):
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)) + cq)

        h = (1 - z) * h + z * q
        return h


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class ProjectionInputDepth(nn.Module):
    def __init__(self, hidden_dim, out_chs):
        super().__init__()
        self.out_chs = out_chs

        self.convd1 = nn.Conv2d(1, hidden_dim, 7, padding=3)
        self.convd2 = nn.Conv2d(hidden_dim, 64, 3, padding=1)

        self.convd = nn.Conv2d(64 + hidden_dim, out_chs - 1, 3, padding=1)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, depth, cor):

        dfm = F.relu(self.convd1(depth))
        dfm = F.relu(self.convd2(dfm))
        cor_dfm = torch.cat([cor, dfm], dim=1)

        out_d = F.relu(self.convd(cor_dfm))
        if self.training and self.dropout is not None:
            out_d = self.dropout(out_d)
        return torch.cat([out_d, depth], dim=1)


class UpMaskNet(nn.Module):
    def __init__(self, hidden_dim=128, ratio=8):
        super(UpMaskNet, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, ratio * ratio * 9, 1, padding=0))

    def forward(self, feat):
        # scale mask to balence gradients
        mask = .25 * self.mask(feat)
        return mask


class BasicUpdateBlockDepth(nn.Module):
    def __init__(self, hidden_dim=128, cost_dim=256, ratio=8, context_dim=64, UpMask=False):
        super(BasicUpdateBlockDepth, self).__init__()

        self.encoder = ProjectionInputDepth(hidden_dim=hidden_dim, out_chs=hidden_dim)
        self.depth_gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=hidden_dim + cost_dim)
        self.depth_head = DepthHead(hidden_dim, hidden_dim=hidden_dim, scale=False)
        self.UpMask = UpMask
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, ratio * ratio * 9, 1, padding=0))
        self.attn = MiniVit(context_dim, n_query_channels=context_dim)
        self.dilated = DilatedConv(dim=context_dim, k=3, dilation=1)

    def forward(self, net, depth_cost_func, inv_depth, context, seq_len=4, scale_inv_depth=None):
        inv_depth_list = []
        mask_list = []
        for i in range(seq_len):

            # TODO detach()
            inv_depth = inv_depth.detach()  # (B, 1, H, W)

            input_features = self.encoder(inv_depth, context)  # (B, context_dim, H, W)

            dilated_feature = self.dilated(input_features)

            attn_context = self.attn(dilated_feature)  # (B, context_dim, H, W)

            inp_i = torch.cat([attn_context, depth_cost_func(scale_inv_depth(inv_depth)[1])],
                              dim=1)  # (B, context_dim+cost_dim, H, W)

            net = self.depth_gru(net, inp_i)

            delta_inv_depth = self.depth_head(net)

            inv_depth = inv_depth + delta_inv_depth
            inv_depth_list.append(inv_depth)
            if self.UpMask and i == seq_len - 1:
                mask = .25 * self.mask(net)
                mask_list.append(mask)
            else:
                mask_list.append(inv_depth)
        return net, mask_list, inv_depth_list


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.0,
                 maxlen: int = 10000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2, dtype=torch.float) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen, dtype=torch.float).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size), dtype=torch.float)
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, embedding):
        return self.dropout(embedding + self.pos_embedding[:embedding.size(0), :])


class MiniVit(nn.Module):
    def __init__(self, in_channels, n_query_channels=32, patch_size=8,
                 embedding_dim=32, num_heads=4, num_layers=2, norm='linear'):
        super(MiniVit, self).__init__()
        self.norm = norm
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.n_query_channels = n_query_channels

        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.embedding_convPxP = nn.Conv2d(in_channels, self.embedding_dim,
                                           kernel_size=patch_size, stride=patch_size, padding=0)

        self.positional_encodings = PositionalEncoding(self.embedding_dim)

        encoder_layers = nn.TransformerEncoderLayer(self.embedding_dim, num_heads,
                                                    dim_feedforward=2 * self.embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, x):
        n, _, h, w = x.shape  # c = self.in_channels

        embeddings = self.embedding_convPxP(x)
        embeddings = embeddings.flatten(2)
        embeddings = embeddings.permute(2, 0, 1)
        embeddings = self.positional_encodings(embeddings)

        tgt = self.transformer_encoder(embeddings)

        queries = tgt[:self.n_query_channels, ...]

        feat = self.conv3x3(x)
        attn_feat = torch.matmul(feat.view(n, self.embedding_dim, h * w).permute(0, 2, 1), queries.permute(1, 2, 0))
        attn_feat = attn_feat.permute(0, 2, 1).view(n, self.n_query_channels, h, w)
        return attn_feat
