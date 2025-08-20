import torch as t
from torch import optim, nn, utils, Tensor
import einops
from load_data import *

class encode_block(nn.Module):
    def __init__(self, n_channels, n_groups):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.GroupNorm(n_groups, n_channels),
            nn.SiLU(),
        )
        self.dropout = nn.Dropout(p = 0.1)
        self.block2 = nn.Sequential(
            nn.Conv2d(n_channels, 2*n_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.GroupNorm(n_groups, 2*n_channels),
            nn.SiLU(),
        )
        self.block3 = nn.MaxPool2d(2)

    def forward(self, x):
        #print(x.shape)
        h1 = x + self.block1(x)
        #print(h1.shape)
        h1 = self.dropout(h1)
        h2 = self.block2(h1)
        return self.block3(h2)
    
class decode_block(nn.Module):
    def __init__(self, n_channels, n_groups):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.GroupNorm(n_groups, n_channels),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(n_channels, n_channels//2, kernel_size = 2, stride = 2),
            nn.GroupNorm(n_groups, n_channels//2),
            nn.SiLU(),
        )
        #self.block3 = nn.ConvTranspose2d(n_channels, n_channels//2, kernel_size = 3, stride = 2)

    def forward(self, x):
        #print("DECODE BLOCK")
        #print(x.shape)
        h1 = x + self.block1(x)
        #print(h1.shape)
        h2 = self.block2(h1)
        #print(h2.shape)
        return h2

class AttentionHead(nn.Module):
    def __init__(self, n_channels: int, d_model: int):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model

        self.W_Q = nn.Linear(n_channels, d_model)
        self.W_K = nn.Linear(n_channels, d_model)
        self.W_V = nn.Linear(n_channels, d_model)

        self.W_O = nn.Linear(d_model, n_channels)

    def forward(self, x: ["batch", "channels", "h", "w"]):
        flattened_x = einops.rearrange(x, "b c h w -> b (h w) c")
        Q = self.W_Q(flattened_x)
        K = self.W_K(flattened_x)
        V = self.W_V(flattened_x)

        z = einops.einsum(Q, K, "b nfeats1 dm, b nfeats2 dm -> b nfeats1 nfeats2")
        z = nn.functional.softmax(z/(self.n_channels**0.5), dim = -1)

        attend_output = einops.einsum(z, V, "b nfeats1 nfeats2, b nfeats2 dm -> b nfeats1 dm")
        output = self.W_O(attend_output)

        return einops.rearrange(output, "b (h w) c-> b c h w", h = x.shape[2], w = x.shape[3])

def positional_encoding(d_model: int, t_vals: ["batch"]):
    expos = einops.repeat(2*t.arange(0, d_model)/d_model, "a -> b a", b = t_vals.shape[0]).to(t_vals.device)
    expanded_tvals = einops.repeat(t_vals, "b -> b a", a = d_model)
    #print(expanded_tvals.device, expos.device)
    pos_encoding_sin = t.sin(expanded_tvals/(t.pow(10000, expos/d_model)))
    pos_encoding_cos = t.cos(expanded_tvals/(t.pow(10000, (expos - 1)/d_model)))

    pos_encoding = t.zeros(t_vals.shape[0], d_model).to(t_vals.device)
    pos_encoding[:, ::2] = pos_encoding_sin[:, ::2]
    pos_encoding[:, 1::2] = pos_encoding_cos[:, 1::2]
    return pos_encoding.unsqueeze(-1).unsqueeze(-1)

class diffusion_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 1),
            nn.SiLU()
        )
        self.encode_blocks = nn.ModuleList([
            encode_block(64, 8),
            encode_block(128, 8),
            AttentionHead(256, 256),
            encode_block(256, 8),
            AttentionHead(512, 64),
        ]
        )
        self.decode_blocks = nn.ModuleList([
            decode_block(512, 8),
            AttentionHead(256, 256),
            decode_block(256, 8),
            decode_block(128,8),
        ]
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size = 1),
            nn.ReLU()
        )
        

    def forward(self, x_t: ["batch", "channels", "h", "w"], t_vals: ["batch"]):
        #print(x_t.shape)
        out1 = self.conv1(x_t) + positional_encoding(64, t_vals)

        encode1 = self.encode_blocks[0](out1) + positional_encoding(128, t_vals)
        encode2 = self.encode_blocks[1](encode1) + positional_encoding(256, t_vals)
        attn1 = self.encode_blocks[2](encode2)
        encode3 = self.encode_blocks[3](attn1) + positional_encoding(512, t_vals)
        attn2 = self.encode_blocks[4](encode3)

        decode1 = attn2 + encode3
        #print(self.decode_blocks[0](decode1).shape, encode2.shape)
        decode2 = self.decode_blocks[0](decode1) + encode2
        attn3 = self.decode_blocks[1](decode2)
        decode3 = self.decode_blocks[2](attn3) + encode1
        decode4 = self.decode_blocks[3](decode3)

        out2 = self.conv2(decode4)            
        return out2
    

if __name__ == "__main__":

    train_dataset = cifar_dataset(["/workspace/Paper-Implementations/ddpm/data/cifar-10-batches-py/data_batch_1"])

    model = diffusion_model()

    batched_data, labels = train_dataset[0:64]
    time_vals = t.randint(low = 1, high = 1000, size = (batched_data.shape[0],))
    print(model(batched_data, time_vals).shape)

    total_params = sum(param.numel() for param in model.parameters())
    print(f"Total number of parameters: {total_params}")
