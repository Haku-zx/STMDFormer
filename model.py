import torch
import torch.nn as nn
from net_component import Transformer
from time import sleep
import torch.nn.functional as F
import math

class SpatialTemporalEmbeddingLayer(nn.Module):

    def __init__(self, edim=32, num_nodes=1, input_dim=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = edim
        self.input_len = 12
        self.input_dim = input_dim
        self.embed_dim = edim
        self.temp_dim_tid = edim
        self.temp_dim_diw = edim

        self.if_time_in_day = True
        self.if_day_in_week = True
        self.if_spatial = True

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)

        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(288, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(7, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)


    def forward(self, x):
        input_data = x[..., range(self.input_dim)]

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        # spatial embeddings
        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))

        tem_emb = []
        if time_in_day_emb is not None:

            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        h = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)

        return h


class TransformerLayer(nn.Module):
    def __init__(self, in_dim, layers=1, dropout=.1, heads=8, num_nodes=1, batch_size=32):
        super().__init__()
        self.transformer = Transformer.Transformer(in_dim, heads, layers, layers, in_dim * 4, dropout=dropout,
                                                   num_nodes=num_nodes, batch_size=batch_size)

    def forward(self, input):
        x = input.permute(1, 0, 2)
        x = self.transformer(x, x)
        return x.permute(1, 0, 2)
class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time


        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_emb = x[..., 1]
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        week_emb = x[..., 2]

        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        tem_emb = time_day + time_week
        return tem_emb

class Conv(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(features, features, (1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
class SpatialAttention(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1):
        super(SpatialAttention, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.d_model = d_model
        self.q = Conv(d_model)
        self.v = Conv(d_model)
        self.concat = Conv(d_model)

        self.memory = nn.Parameter(torch.randn(head, seq_length, num_nodes, self.d_k))
        nn.init.xavier_uniform_(self.memory)

        self.weight = nn.Parameter(torch.ones(d_model, num_nodes, seq_length))
        self.bias = nn.Parameter(torch.zeros(d_model, num_nodes, seq_length))

        apt_size = 10
        nodevecs = torch.randn(num_nodes, apt_size), torch.randn(apt_size, num_nodes)
        self.nodevec1, self.nodevec2 = [
            nn.Parameter(n.to(device), requires_grad=True) for n in nodevecs
        ]

    def forward(self, input, adj_list=None):
        query, value = self.q(input), self.v(input)
        query = query.view(
            query.shape[0], -1, self.d_k, query.shape[2], self.seq_length
        ).permute(0, 1, 4, 3, 2)
        value = value.view(
            value.shape[0], -1, self.d_k, value.shape[2], self.seq_length
        ).permute(
            0, 1, 4, 3, 2
        )

        key = torch.softmax(self.memory / math.sqrt(self.d_k), dim=-1)
        query = torch.softmax(query / math.sqrt(self.d_k), dim=-1)
        Aapt = torch.softmax(
            F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=-1
        )
        kv = torch.einsum("hlnx, bhlny->bhlxy", key, value)
        attn_qkv = torch.einsum("bhlnx, bhlxy->bhlny", query, kv)

        attn_dyn = torch.einsum("nm,bhlnc->bhlnc", Aapt, value)

        x = attn_qkv + attn_dyn

        x = self.concat(x)
        if self.num_nodes not in [170, 358,5]:
            x = x * self.weight + self.bias + x
        return x, self.weight, self.bias


class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, input):
        mean = input.mean(dim=(1, 2), keepdim=True)
        variance = input.var(dim=(1, 2), unbiased=False, keepdim=True)
        input = (input - mean) / torch.sqrt(variance + self.eps)
        if self.elementwise_affine:
            input = input * self.weight + self.bias
        return input

class Encoder(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1):
        "Take in model size and number of heads."
        super(Encoder, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.d_model = d_model
        self.attention = SpatialAttention(
            device, d_model, head, num_nodes, seq_length=seq_length
        )
        self.LayerNorm = LayerNorm(
            [d_model, num_nodes, seq_length], elementwise_affine=False
        )
        self.dropout1 = nn.Dropout(p=dropout)
        self.glu = GLU(d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, input, adj_list=None):
        # 64 64 170 12
        x, weight, bias = self.attention(input)
        x = x + input
        x = self.LayerNorm(x)
        x = self.dropout1(x)
        x = self.glu(x) + x
        x = x * weight + bias + x
        x = self.LayerNorm(x)
        x = self.dropout2(x)
        return x


class network(nn.Module):
    def __init__(self, dropout=0.1, edim=32, out_dim=12, hid_dim=64, layers=6, batch_size=32, num_nodes=1, input_dim=3):
        super(network, self).__init__()
        self.stelayer = SpatialTemporalEmbeddingLayer(edim=edim, num_nodes=num_nodes, input_dim=input_dim)
        self.conv = nn.Conv2d(in_channels=edim * 4, out_channels=hid_dim, kernel_size=(1, 1))
        self.translyear = TransformerLayer(in_dim=hid_dim, layers=layers, dropout=dropout, num_nodes=num_nodes,
                                           batch_size=batch_size)
        self.lin = nn.Linear(hid_dim, out_dim)

        if num_nodes == 170 or num_nodes == 307 or num_nodes == 358  or num_nodes == 883:
            time = 288
        elif num_nodes == 250 or num_nodes == 266:
            time = 48
        elif num_nodes>200:
            time = 96

        if num_nodes == 883:
            self.input_dim = 3

        self.Temb = TemporalEmbedding(time, channels)
        self.network_channel = channels * 2
        self.head = 1

        self.SpatialBlock = Encoder(
            device,
            d_model=self.network_channel,
            head=self.head,
            num_nodes=num_nodes,
            seq_length=1,
            dropout=dropout,
        )

        self.fc_st = nn.Conv2d(
            self.network_channel, self.network_channel, kernel_size=(1, 1)
        )
        self.conv2 = nn.Conv2d(256, 128, kernel_size=(1, 1))

    def forward(self, input):
        x = input.transpose(1, 3)
        x1 = input.transpose(1, 3)
        x = self.stelayer(x)
        tem_emb = self.Temb(x1)
        data_st = torch.cat([x] + [tem_emb], dim=1)
        x = self.SpatialBlock(data_st) + self.fc_st(data_st)
        x = self.conv2(x)
        x = self.conv(x)[..., -1]
        x = x.transpose(1, 2)
        x = self.translyear(x)
        x = self.lin(x)
        return x.transpose(1, 2).unsqueeze(-1)
