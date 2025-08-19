import numpy as np
import torch as t
import torch.nn as nn
import einops
import matplotlib.pyplot as plt
from dataclasses import dataclass

class AttentionHead(nn.Module):
    def __init__(self, d_k: int, d_v: int, d_model: int, mask: bool): 
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.mask = mask

        self.W_Q = nn.Parameter(t.zeros((self.d_k, self.d_model)))
        self.W_K = nn.Parameter(t.zeros((self.d_k, self.d_model)))
        self.W_V = nn.Parameter(t.zeros((self.d_v, self.d_model)))

        nn.init.normal_(self.W_Q, 0, 0.02)
        nn.init.normal_(self.W_K, 0, 0.02)
        nn.init.normal_(self.W_V, 0, 0.02)

    def forward(self, 
                q: ["batch", "seq_len", "d_model"], 
                k: ["batch", "seq_len", "d_model"],
                v: ["batch", "seq_len", "d_model"]):
        Q = einops.einsum(self.W_Q, q, "d_k d_model, b seq_len d_model -> b seq_len d_k")
        K = einops.einsum(self.W_K, k, "d_k d_model, b seq_len d_model -> b seq_len d_k")
        V = einops.einsum(self.W_V, v, "d_v d_model, b seq_len d_model -> b seq_len d_v")
        z = einops.einsum(Q, K, "b seq1 d_k, b seq2 d_k -> b seq1 seq2")

        if self.mask:
            mask = t.triu(t.ones(z.shape[1], z.shape[2]), diagonal = 1).bool().to(z.device)
            z = z.masked_fill(mask, -t.inf)

        z = nn.functional.softmax(z/(self.d_k**0.5), dim = -1)

        return einops.einsum(z, V, "b seq1 seq2, b seq2 d_v -> b seq1 d_v")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_k: int, d_v: int, d_model: int, n_heads: int):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.attention_heads = nn.ModuleList(
            [AttentionHead(d_k, d_v, d_model, True) for _ in range(n_heads)]
        )
        self.W_O = nn.Parameter(t.zeros((n_heads * d_v, d_model)))
        nn.init.normal_(self.W_O, 0, 0.02)

    def forward(self, 
                q : ["batch", "seq_len", "d_model"],
                k : ["batch", "seq_len", "d_model"],
                v : ["batch", "seq_len", "d_model"],):
        #norm_x = self.layer_norm1(x)
        head_outs = []
        for i in range(self.n_heads):
            head_outs.append(self.attention_heads[i](q,k,v))
        concat_head_outs = t.concat(head_outs, dim = -1)
        multi_head_outs = einops.einsum(self.W_O, concat_head_outs, "headsxd_v d_model, b seq_len headsxd_v -> b seq_len d_model")
        return multi_head_outs


class AttentionBlock(nn.Module):
    def __init__(self, d_k: int, d_v: int, d_model: int, n_heads: int):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.multihead = MultiHeadAttention(d_k, d_v, d_model, n_heads)
        self.layer_norm1 = nn.LayerNorm([self.d_model])
        self.layer_norm2 = nn.LayerNorm([self.d_model])
        self.feedforward = nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model),
            nn.ReLU(),
            nn.Linear(4*self.d_model, self.d_model),
            nn.Dropout(p = 0.1)
        )

    def forward(self, x : ["batch", "seq_len", "d_model"]):
        norm_x = self.layer_norm1(x)
        multi_out = x + self.multihead(norm_x, norm_x, norm_x)
        norm1_x = self.layer_norm2(multi_out)
        out_x = multi_out + self.feedforward(norm1_x)
        return out_x 

def positional_encoding(d_model: int, seq_len: int, device):
    base_vals = einops.repeat(t.arange(0, seq_len).to(device), "a -> b a", b = d_model)
    exps = einops.repeat(t.arange(0, d_model).to(device), "a -> a b", b = seq_len)
    pos_encoding_sin = t.sin(base_vals/t.pow(10000, exps/d_model))
    pos_encoding_cos = t.cos(base_vals/t.pow(10000, (exps - 1)/d_model))

    pos_encoding = t.zeros(d_model, seq_len).to(device)
    pos_encoding[::2, :] = pos_encoding_sin[::2, :]
    pos_encoding[1::2, :] = pos_encoding_cos[1::2, :]
    return pos_encoding.T

@dataclass
class DecoderConfig:
    n_layers: int
    d_model: int
    n_heads: int
    d_head: int
    alphabet_size: int

    def get_model(self):
        return Decoder(self)

class Decoder(nn.Module):

    def __init__(self, config: DecoderConfig):
        super().__init__()

        self.d_k = config.d_head
        self.d_v = config.d_head
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.alphabet_size = config.alphabet_size

        self.embed_linear = nn.Embedding(self.alphabet_size, self.d_model)

        self.attention_blocks = nn.ModuleList(
            [AttentionBlock(self.d_k, self.d_v, self.d_model, self.n_heads) for _ in range(self.n_layers)]
        )
        self.final_layer_norm = nn.LayerNorm([self.d_model])
        self.final_linear = nn.Parameter(t.zeros(self.d_model, self.alphabet_size))
        nn.init.normal_(self.final_linear, 0, 0.02)

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.02)

    def forward(self, x : ["batch", "seq_len"]):
        pos_encode = positional_encoding(self.d_model, x.shape[1], x.device)
        embed = self.embed_linear(x) + pos_encode
        for i in range(self.n_layers):
            embed = self.attention_blocks[i](embed)
        embed = self.final_layer_norm(embed)
        embed = einops.einsum(embed, self.final_linear, "batch seq_len d_model, d_model alphabet_size -> batch seq_len alphabet_size")
        #embed = t.softmax(embed, dim = -1)
        return embed
 

    """
    def __init__(self, d_k: int, d_v: int, d_model: int, n_heads: int, n_layers: int, alphabet_size: int, seq_len: int):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.pos_encoding = positional_encoding(d_model, seq_len)
        self.pos_encoding.to(device)

        self.embed_linear = nn.Embedding(alphabet_size, d_model)
        self.embed_linear.to(device)
        #nn.Parameter(t.zeros(alphabet_size, d_model))
        #nn.init.normal_(self.embed_linear, 0, 0.02)
        self.attention_blocks = [AttentionBlock(d_k, d_v, d_model, n_heads) for _ in range(n_layers)]
        self.final_layer_norm = nn.LayerNorm([self.d_model]).to(device)
        self.final_linear = nn.Parameter(t.zeros(d_model, alphabet_size)).to(device)
        nn.init.normal_(self.final_linear, 0, 0.02)

    def forward(self, x : ["batch", "seq_len"]):
        assert x.shape[1] == self.seq_len
        #print(x.long())
        #one_hot = nn.functional.one_hot(x.long(), num_classes = alphabet_size).float()
        #print(one_hot.shape)
        #print(self.embed_linear.shape)
        #embed = einops.einsum(one_hot, self.embed_linear, "batch seq_len alph_size, alph_size d_model -> batch seq_len d_model") + self.pos_encoding

        embed = self.embed_linear(x) + self.pos_encoding
        for i in range(self.n_layers):
            embed = self.attention_blocks[i](embed)
        embed = self.final_layer_norm(embed)
        embed = einops.einsum(embed, self.final_linear, "batch seq_len d_model, d_model alphabet_size -> batch seq_len alphabet_size")
        embed = t.softmax(embed, dim = -1)
        return embed
        #return einops.rearrange(embed, "batch seq_len alphabet_size -> batch alphabet_size seq_len")
    """