import torch
from torch import nn
from deepblocks.layer import MultiHeadAttention, MultiHeadAttentionV2


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) 
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def test_forward_shape():
    for i in range(1, 20):
        mha = MultiHeadAttention(100, i)
        x = torch.rand(11, 21, 100)
        assert mha(x).shape == (11, 21, 100)


def test_optimization_against_vanilla():
    mha = MultiHeadAttentionV2(10, 1)
    vanilla = Attention(10, 1)
    x = torch.rand(11, 21, 10)
    y = torch.randn(11, 21, 10)

    mha_opt = torch.optim.AdamW(mha.parameters(), lr=1)
    vanilla_opt = torch.optim.AdamW(vanilla.parameters(), lr=1)

    def loss(model, opt):
        l = 1e10
        for _ in range(1000):
            y_hat = model(x)
            loss = (y_hat-y).pow(2).mean()
            if l > loss.item():
                l = loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        # return the best train loss
        return l

    mha_loss = loss(mha, mha_opt)
    vanilla_loss = loss(vanilla, vanilla_opt)

    assert mha_loss-vanilla_loss < 1e-1