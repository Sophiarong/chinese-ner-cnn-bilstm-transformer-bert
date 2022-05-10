import torch
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer, TransformerEncoder

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ntag, d_model, nhead, d_hid, nlayers, pad_index, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.classier_1 = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()
        self.classier_2 = nn.Linear(d_model, ntag)
        self.cross_loss = nn.CrossEntropyLoss(ignore_index=pad_index)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.classier_1.bias.data.zero_()
        self.classier_1.weight.data.uniform_(-initrange, initrange)
        self.classier_2.bias.data.zero_()
        self.classier_2.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask, src_padding_mask, tag):
        """
        :param src:Tensor, shape[seq_len, batch_size]
        :param src_mask: Tensor, shape[seq_len, seq_len]
        :return: Tensor, shape[seq_len, batch_size, ntag]
        """
        src = src.t()
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_padding_mask)

        pre = self.classier_2(self.relu(self.classier_1(output)))
        pre = pre.permute(1, 0, 2)
        self.result = torch.argmax(pre, dim=-1)
        self.pre = torch.argmax(pre, dim=-1).reshape(-1)

        if tag is not None:
            loss = self.cross_loss(pre.reshape(-1, pre.shape[-1]), tag.reshape(-1))
            return loss


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position*div_term)
        pe[:, 0, 1::2] = torch.cos(position*div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: Tensor, shape[seq_len, batchsize, embedding_dim]
        :return:
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz)*float('-inf'), diagonal=1)