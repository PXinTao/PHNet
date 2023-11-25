import torch
import torch.nn as nn
import torch.nn.functional as F



class MultiScaleTransformer(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, kernel_sizes=[3, 11, 19, 27], embed_dim=32, depth=12, n_heads=8, qkv_bias=True, attn_p=0, proj_p=0):
        super(MultiScaleTransformer, self).__init__()
        self.pixel_embed = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=1, bias=False)
        self.blocks = nn.ModuleList(
            [
                MultiScaleBlock(
                    in_channels=embed_dim, kernel_sizes=kernel_sizes, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=proj_p
                ) for _ in range(depth)
            ]
        )
        self.pred_head = nn.Conv2d(in_channels=embed_dim, out_channels=num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, image):
        x = self.pixel_embed(image)
        for b in self.blocks:
            x = b(x)
        x = self.pred_head(x)
        return x


class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, kernel_sizes, n_heads, qkv_bias=True, attn_p=0, proj_p=0):
        super(MultiScaleBlock, self).__init__()
        self.msconv = MultiScaleConv(in_channels=in_channels, kernel_sizes=kernel_sizes)
        self.attn = MultiScaleAttention(dim=in_channels, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=proj_p)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels=in_channels*len(kernel_sizes), out_channels=in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor):
        x = self.msconv(x)
        x = x + self.attn(x)
        x = x.flatten(1, 2)
        x = self.fuse(x)
        return x


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[3, 11, 19, 27], bias=True):
        super(MultiScaleConv, self).__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=k,
                    padding=(k-1)//2,
                    groups=in_channels,
                    bias=bias
                ) for k in kernel_sizes
            ]
        )
    def forward(self, x):
        outs = []
        for m in self.convs:
            outs.append(m(x))
        return torch.stack(outs, dim=1)


class MultiScaleAttention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super(MultiScaleAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, h, w, n_tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError
        # x = F.layer_norm(x, normalized_shape=(dim, h, w), eps=1e-6)

        # (n_samples, h, w, n_tokens, 3 * dim)
        qkv = self.qkv(x)

        # (n_samples, h, w, n_tokens, 3, n_heads, head_dim)
        qkv = qkv.reshape(n_samples, h, w, n_tokens, 3, self.n_heads, self.head_dim)

        # (3, n_samples, h, w, n_heads, n_tokens, head_dim)
        qkv = qkv.permute(4, 0, 1, 2, 5, 3, 6)

        q, k, v = qkv[0], qkv[1], qkv[2]

        # (n_samples, h, w, n_heads, n_tokens, n_tokens)
        dp = (q @ k.transpose(-2, -1)) * self.scale
        attn = dp.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # (n_samples, h, w, n_heads, n_tokens, head_dim)
        weighted_avg = attn @ v
        # (n_samples, h, w, n_tokens, n_heads, head_dim)
        weighted_avg = weighted_avg.transpose(3, 4)
        
        # (n_samples, h, w, n_tokens, dim)
        weighted_avg = weighted_avg.flatten(4)

        # (n_samples, h, w, n_tokens, dim)
        x = self.proj(weighted_avg)
        # (n_samples, h, w, n_tokens, dim)
        x = self.proj_drop(x)

        # (n_samples, n_tokens, dim, h, w)
        # return x.permute(0, 3, 4, 1, 2)

        # (n_samples, h, w, n_tokens, dim)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiScaleAttention(
                dim,
                n_heads=n_heads,
                qkv_bias=qkv_bias,
                attn_p=attn_p,
                proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
                in_features=dim,
                hidden_features=hidden_features,
                out_features=dim,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


if __name__ == "__main__":
    m = MultiScaleTransformer()
    x = torch.randn(5, 3, 200, 200)
    y = m(x)
    print(x.shape)