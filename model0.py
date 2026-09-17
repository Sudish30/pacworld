"""Model 0: action-conditioned conv UNet that predicts the next frame from K context frames.

Input: context frames concatenated on channels (B, K*3, H, W) and the K context
actions (B, K). Actions are embedded, concatenated and projected to a
conditioning vector that is added inside every residual block (the same slot a
diffusion UNet uses for its timestep embedding). With predict_residual the
network outputs a delta that is added to the last context frame.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, cin, cout, cond_dim, groups):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.cond = nn.Linear(cond_dim, cout)
        self.norm2 = nn.GroupNorm(groups, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.cond(cond)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, context, n_actions, base_channels, channel_mults, blocks_per_level,
                 cond_dim, action_embed_dim, groupnorm_groups, predict_residual):
        super().__init__()
        self.K = context
        self.predict_residual = predict_residual
        g = groupnorm_groups

        self.action_embed = nn.Embedding(n_actions, action_embed_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(context * action_embed_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))

        widths = [base_channels * m for m in channel_mults]
        self.in_conv = nn.Conv2d(context * 3, widths[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_channels = [widths[0]]
        c = widths[0]
        for level, w in enumerate(widths):
            blocks = nn.ModuleList()
            for _ in range(blocks_per_level):
                blocks.append(ResBlock(c, w, cond_dim, g))
                c = w
                skip_channels.append(c)
            self.down_blocks.append(blocks)
            if level < len(widths) - 1:
                self.downsamples.append(Downsample(c))
                skip_channels.append(c)
            else:
                self.downsamples.append(nn.Identity())

        self.mid1 = ResBlock(c, c, cond_dim, g)
        self.mid2 = ResBlock(c, c, cond_dim, g)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level, w in reversed(list(enumerate(widths))):
            blocks = nn.ModuleList()
            for _ in range(blocks_per_level + 1):
                blocks.append(ResBlock(c + skip_channels.pop(), w, cond_dim, g))
                c = w
            self.up_blocks.append(blocks)
            self.upsamples.append(Upsample(c) if level > 0 else nn.Identity())

        self.out_norm = nn.GroupNorm(g, c)
        self.out_conv = nn.Conv2d(c, 3, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, ctx, actions):
        cond = self.cond_mlp(self.action_embed(actions).flatten(1))
        h = self.in_conv(ctx)
        skips = [h]
        for blocks, down in zip(self.down_blocks, self.downsamples):
            for block in blocks:
                h = block(h, cond)
                skips.append(h)
            if not isinstance(down, nn.Identity):
                h = down(h)
                skips.append(h)
        h = self.mid2(self.mid1(h, cond), cond)
        for blocks, up in zip(self.up_blocks, self.upsamples):
            for block in blocks:
                h = block(torch.cat([h, skips.pop()], dim=1), cond)
            h = up(h)
        out = self.out_conv(F.silu(self.out_norm(h)))
        if self.predict_residual:
            out = out + ctx[:, -3:]
        return out


def build_model(cfg):
    m = cfg["model"]
    return UNet(context=cfg["data"]["context"], n_actions=m["n_actions"], base_channels=m["base_channels"],
                channel_mults=m["channel_mults"], blocks_per_level=m["blocks_per_level"], cond_dim=m["cond_dim"],
                action_embed_dim=m["action_embed_dim"], groupnorm_groups=m["groupnorm_groups"],
                predict_residual=m["predict_residual"])


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    from common import load_config
    cfg = load_config("configs/model0.yaml")
    model = build_model(cfg)
    print(f"params: {count_params(model) / 1e6:.2f}M")
    x = torch.randn(2, cfg["data"]["context"] * 3, 64, 64)
    a = torch.randint(0, 9, (2, cfg["data"]["context"]))
    print("output:", tuple(model(x, a).shape))
