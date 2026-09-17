"""Model 1: DIAMOND-style pixel-space conditional diffusion world model.

UNet input: the noisy target frame (scaled by c_in) concatenated on channels with
the K context frames (15 channels for K=4). Conditioning = 4 action embeddings +
Fourier features of the noise level c_noise + Fourier features of the context
noise level, summed, passed through an MLP, and injected into every residual
block through adaptive GroupNorm. Self-attention at the low resolutions.

Denoiser follows Karras et al. 2022 (EDM), Table 1:
  c_skip = sd^2 / (s^2 + sd^2)      c_out = s * sd / sqrt(s^2 + sd^2)
  c_in   = 1 / sqrt(s^2 + sd^2)     c_noise = ln(s) / 4
  D(x; s) = c_skip * x + c_out * F(c_in * x, c_noise, ...)
Training loss is MSE between F and (x0 - c_skip * x) / c_out, which equals the
EDM-weighted denoising loss. Sampling uses a deterministic Euler sampler on the
Karras sigma schedule.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- blocks
class FourierFeatures(nn.Module):
    def __init__(self, dim, scale=16.0):
        super().__init__()
        self.register_buffer("freqs", torch.randn(dim // 2) * scale)

    def forward(self, x):                      # (B,) -> (B, dim)
        ang = x[:, None] * self.freqs[None] * 2 * math.pi
        return torch.cat([ang.sin(), ang.cos()], dim=1)


class AdaGN(nn.Module):
    """GroupNorm whose per-channel scale and shift come from the conditioning vector."""

    def __init__(self, channels, cond_dim, groups):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels, affine=False)
        self.proj = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        gamma, beta = self.proj(cond)[:, :, None, None].chunk(2, dim=1)
        return self.norm(x) * (1 + gamma) + beta


class ResBlock(nn.Module):
    def __init__(self, cin, cout, cond_dim, groups, dropout):
        super().__init__()
        self.norm1 = AdaGN(cin, cond_dim, groups)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = AdaGN(cout, cond_dim, groups)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x, cond)))
        h = self.conv2(self.drop(F.silu(self.norm2(h, cond))))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, channels, heads, groups):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, H * W).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.proj(out.transpose(-1, -2).reshape(B, C, H, W))


class Stage(nn.Module):
    """A residual block optionally followed by self-attention."""

    def __init__(self, cin, cout, cond_dim, groups, dropout, attn, heads):
        super().__init__()
        self.res = ResBlock(cin, cout, cond_dim, groups, dropout)
        self.attn = Attention(cout, heads, groups) if attn else None

    def forward(self, x, cond):
        x = self.res(x, cond)
        return self.attn(x) if self.attn is not None else x


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


# ----------------------------------------------------------------------------- UNet
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, context, n_actions, widths, blocks_per_level, attn_levels,
                 heads, cond_dim, action_embed_dim, groupnorm_groups, fourier_dim, dropout):
        super().__init__()
        g = groupnorm_groups
        self.action_embed = nn.Embedding(n_actions, action_embed_dim)
        self.action_proj = nn.Linear(context * action_embed_dim, cond_dim)
        self.noise_ff = FourierFeatures(fourier_dim)
        self.noise_proj = nn.Linear(fourier_dim, cond_dim)
        self.ctx_ff = FourierFeatures(fourier_dim)
        self.ctx_proj = nn.Linear(fourier_dim, cond_dim)
        self.cond_mlp = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))

        self.in_conv = nn.Conv2d(in_channels, widths[0], 3, padding=1)
        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        skips = [widths[0]]
        c = widths[0]
        for level, w in enumerate(widths):
            attn = level in attn_levels
            stages = nn.ModuleList()
            for _ in range(blocks_per_level):
                stages.append(Stage(c, w, cond_dim, g, dropout, attn, heads))
                c = w
                skips.append(c)
            self.down.append(stages)
            if level < len(widths) - 1:
                self.downsample.append(Downsample(c))
                skips.append(c)
            else:
                self.downsample.append(nn.Identity())

        self.mid = nn.ModuleList([Stage(c, c, cond_dim, g, dropout, True, heads), Stage(c, c, cond_dim, g, dropout, False, heads)])

        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for level, w in reversed(list(enumerate(widths))):
            attn = level in attn_levels
            stages = nn.ModuleList()
            for _ in range(blocks_per_level + 1):
                stages.append(Stage(c + skips.pop(), w, cond_dim, g, dropout, attn, heads))
                c = w
            self.up.append(stages)
            self.upsample.append(Upsample(c) if level > 0 else nn.Identity())
        assert not skips

        self.out_norm = nn.GroupNorm(g, c)
        self.out_conv = nn.Conv2d(c, out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, actions, c_noise, c_ctx_noise):
        cond = (self.action_proj(self.action_embed(actions).flatten(1))
                + self.noise_proj(self.noise_ff(c_noise))
                + self.ctx_proj(self.ctx_ff(c_ctx_noise)))
        cond = self.cond_mlp(cond)

        h = self.in_conv(x)
        skips = [h]
        for stages, down in zip(self.down, self.downsample):
            for stage in stages:
                h = stage(h, cond)
                skips.append(h)
            if not isinstance(down, nn.Identity):
                h = down(h)
                skips.append(h)
        for stage in self.mid:
            h = stage(h, cond)
        for stages, up in zip(self.up, self.upsample):
            for stage in stages:
                h = stage(torch.cat([h, skips.pop()], dim=1), cond)
            h = up(h)
        return self.out_conv(F.silu(self.out_norm(h)))


# ----------------------------------------------------------------------------- EDM denoiser
class Denoiser(nn.Module):
    def __init__(self, unet, sigma_data):
        super().__init__()
        self.unet = unet
        self.sigma_data = sigma_data

    def coefficients(self, sigma):
        sd = self.sigma_data
        s2 = sigma ** 2 + sd ** 2
        c_skip = sd ** 2 / s2
        c_out = sigma * sd / s2.sqrt()
        c_in = 1 / s2.sqrt()
        c_noise = sigma.log() / 4
        return c_skip[:, None, None, None], c_out[:, None, None, None], c_in[:, None, None, None], c_noise

    @staticmethod
    def ctx_noise_embedding(ctx_sigma):
        # same transform as c_noise; a clean context (sigma 0) maps to the smallest level
        return ctx_sigma.clamp(min=1e-3).log() / 4

    def raw(self, x_noisy, sigma, ctx, actions, ctx_sigma):
        """Network output F for noisy input x at noise level sigma."""
        _, _, c_in, c_noise = self.coefficients(sigma)
        inp = torch.cat([c_in * x_noisy, ctx], dim=1)
        return self.unet(inp, actions, c_noise, self.ctx_noise_embedding(ctx_sigma))

    def forward(self, x_noisy, sigma, ctx, actions, ctx_sigma):
        """Denoised estimate D(x; sigma)."""
        c_skip, c_out, _, _ = self.coefficients(sigma)
        return c_skip * x_noisy + c_out * self.raw(x_noisy, sigma, ctx, actions, ctx_sigma).float()

    def loss(self, x0, ctx, actions, sigma, ctx_sigma, noise=None):
        """EDM-weighted denoising loss (per-sample mean), expressed as MSE in F-space."""
        noise = torch.randn_like(x0) if noise is None else noise
        x_noisy = x0 + sigma[:, None, None, None] * noise
        c_skip, c_out, _, _ = self.coefficients(sigma)
        target = (x0 - c_skip * x_noisy) / c_out
        out = self.raw(x_noisy, sigma, ctx, actions, ctx_sigma).float()
        return F.mse_loss(out, target)


def sample_sigmas(p_mean, p_std, n, device, generator=None):
    return (torch.randn(n, device=device, generator=generator) * p_std + p_mean).exp()


def karras_schedule(n_steps, sigma_min, sigma_max, rho, device):
    if n_steps == 1:
        return torch.tensor([sigma_max, 0.0], device=device)
    i = torch.arange(n_steps, device=device, dtype=torch.float32)
    sig = (sigma_max ** (1 / rho) + i / (n_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([sig, sig.new_zeros(1)])


@torch.no_grad()
def euler_sample(denoiser, ctx, actions, ctx_sigma, n_steps, dcfg, generator=None):
    """Deterministic Euler sampler (Karras Alg. 1 with S_churn = 0)."""
    B, _, H, W = ctx.shape
    device = ctx.device
    sigmas = karras_schedule(n_steps, dcfg["sigma_min"], dcfg["sigma_max"], dcfg["rho"], device)
    x = torch.randn(B, 3, H, W, device=device, generator=generator) * sigmas[0]
    for i in range(n_steps):
        s, s_next = sigmas[i], sigmas[i + 1]
        d = (x - denoiser(x, s.expand(B), ctx, actions, ctx_sigma)) / s
        x = x + (s_next - s) * d
    return x.clamp(-1, 1)


def build_model(cfg):
    m, K = cfg["model"], cfg["data"]["context"]
    unet = UNet(in_channels=3 + 3 * K, out_channels=3, context=K, n_actions=m["n_actions"], widths=m["widths"],
                blocks_per_level=m["blocks_per_level"], attn_levels=m["attn_levels"], heads=m["heads"],
                cond_dim=m["cond_dim"], action_embed_dim=m["action_embed_dim"], groupnorm_groups=m["groupnorm_groups"],
                fourier_dim=m["fourier_dim"], dropout=m["dropout"])
    return Denoiser(unet, cfg["diffusion"]["sigma_data"])


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    from common import load_config
    cfg = load_config("configs/model1.yaml")
    model = build_model(cfg)
    print(f"params: {count_params(model) / 1e6:.2f}M")
    B, K = 2, cfg["data"]["context"]
    ctx = torch.randn(B, 3 * K, 64, 64)
    acts = torch.randint(0, 9, (B, K))
    x0 = torch.randn(B, 3, 64, 64)
    sigma = sample_sigmas(cfg["diffusion"]["p_mean"], cfg["diffusion"]["p_std"], B, "cpu")
    print("loss:", model.loss(x0, ctx, acts, sigma, torch.zeros(B)).item())
    out = euler_sample(model, ctx, acts, torch.zeros(B), 3, cfg["diffusion"])
    print("sample:", tuple(out.shape), f"range [{out.min():.2f}, {out.max():.2f}]")
