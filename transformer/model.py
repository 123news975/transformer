
import math

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.scale = d_k ** -0.5
    def forward(self, Q, K, V, mask: Optional[torch.Tensor]=None):
        # Q, K, V: [batch, heads, len, d_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch, heads, q_len, k_len]
        if mask is not None:
            # mask should be broadcastable to scores; we expect mask==1 for valid, 0 for masked
            scores = scores.masked_fill(mask == 0, float("-1e9"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # [batch, heads, q_len, d_k]
        return out, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask: Optional[torch.Tensor]=None):
        # query/key/value: [batch, seq_len, d_model]
        batch_size = query.size(0)

        # linear & split heads
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)  # [batch, heads, q_len, d_k]
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)

        # mask broadcasting: mask should be of shape [batch, 1, q_len, k_len] or [batch, 1, 1, k_len]
        if mask is not None:
            # ensure mask shape matches [batch, 1, q_len, k_len] or [batch, 1, 1, k_len]
            # attention expects 0 for masked positions.
            mask = mask.to(query.device)

        out, attn = self.attention(Q, K, V, mask=mask)
        # concat heads
        out = out.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)  # [batch, q_len, d_model]
        out = self.w_o(out)
        out = self.dropout(out)
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, src_mask=None):
        # self-attention
        _x = x
        attn_out = self.self_attn(x, x, x, mask=src_mask)  # [batch, seq, d_model]
        x = _x + self.dropout(attn_out)
        x = self.norm1(x)
        # ffn
        _x = x
        ffn_out = self.ffn(x)
        x = _x + self.dropout(ffn_out)
        x = self.norm2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_out, src_mask=None, trg_mask=None):
        # masked self-attention
        _x = x
        self_attn_out = self.self_attn(x, x, x, mask=trg_mask)
        x = _x + self.dropout(self_attn_out)
        x = self.norm1(x)
        # encoder-decoder attention
        _x = x
        cross_attn_out = self.cross_attn(x, enc_out, enc_out, mask=src_mask)
        x = _x + self.dropout(cross_attn_out)
        x = self.norm2(x)
        # ffn
        _x = x
        ffn_out = self.ffn(x)
        x = _x + self.dropout(ffn_out)
        x = self.norm3(x)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, d_ff, num_layers, dropout, max_pos=MAX_POSITION):
        super().__init__()
        self.tok_embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_pos)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    def forward(self, src, src_mask=None):
        x = self.tok_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, num_heads, d_ff, num_layers, dropout, max_pos=MAX_POSITION):
        super().__init__()
        self.tok_embedding = nn.Embedding(output_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_pos)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    def forward(self, trg, enc_out, src_mask=None, trg_mask=None):
        x = self.tok_embedding(trg) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, trg_mask)
        out = self.fc_out(x)
        return out

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, num_heads, d_ff, num_layers, dropout):
        super().__init__()

        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(trg_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.src_pad_idx = en_vocab["<pad>"]
        self.trg_pad_idx = zh_vocab["<pad>"]

    def make_src_mask(self, src):
        # src: [batch, src_len] -> mask: [batch, 1, 1, src_len], 1 indicates valid
        mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return mask  # dtype=bool



    def make_trg_mask(self, trg):
        # trg: [batch, trg_len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)  # [batch,1,1,trg_len]
        trg_len = trg.size(1)
        # subsequent mask: [1,1,trg_len,trg_len] with ones in lower triangle (valid)
        subsequent = torch.tril(torch.ones((trg_len, trg_len), dtype=torch.bool, device=trg.device)).unsqueeze(0).unsqueeze(0)
        mask = trg_pad_mask & subsequent  # broadcast to [batch,1,trg_len,trg_len]
        return mask

    def forward(self, src, trg):
        # src: [batch, src_len], trg: [batch, trg_len]
        src_mask = self.make_src_mask(src)  # [batch,1,1,src_len]
        trg_mask = self.make_trg_mask(trg)  # [batch,1,trg_len,trg_len]
        enc_out = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_out, src_mask, trg_mask)
        return out
