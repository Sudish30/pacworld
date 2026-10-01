"""Latent action model (LAM): infers a small discrete action vocabulary from frames alone (Genie-style).

For the transition frames[j] -> frames[j+1] (target i = j+1) the ENCODER sees the frames at offsets
lam.encoder_offsets = [-3, -2, -1, 0] around the target (j-2 .. j+1) and outputs a vector that is quantised to one
of lam.n_codes codes (EMA vector quantisation, straight-through gradient). The DECODER is model1's UNet body: it sees
the world model's own context (data.context_offsets, frames before the target only) and gets the quantised code in
the conditioning slot where the world model gets its actions; it predicts the target as a residual on the last
context frame. The code is the only path from the future to the decoder, and it holds at most log2(n_codes) bits.
codes[j] = the code of transition j -> j+1, i.e. a drop-in replacement for actions[j] (dataset.apply_action_source).

PRIOR: p(code of the next transition | the world model's context), a small classifier. At play time it says which
codes can happen here; resolve() turns held keys into a code with it (notes/latent_actions_design.md, section 7).

Nothing in this file reads actions or RAM. The Pac-Man positions used by arm B's loss weight come from the pixel
detector (tools/pac_positions.py).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import gather_context
from model1 import UNet


# ----------------------------------------------------------------------------- encoder / prior trunk
class ConvTrunk(nn.Module):
    """Stride-2 conv stages down to 4x4, then an MLP to out_dim (the LAM encoder and the prior share this shape)."""

    def __init__(self, in_channels, widths, hidden, out_dim, groups, size):
        super().__init__()
        layers, c = [], in_channels
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, stride=2, padding=1), nn.GroupNorm(groups, w), nn.SiLU(),
                       nn.Conv2d(w, w, 3, padding=1), nn.GroupNorm(groups, w), nn.SiLU()]
            c = w
        self.convs = nn.Sequential(*layers)
        side = size // 2 ** len(widths)
        self.mlp = nn.Sequential(nn.Flatten(), nn.Linear(c * side * side, hidden), nn.SiLU(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.mlp(self.convs(x))


# ----------------------------------------------------------------------------- vector quantiser
class VectorQuantizer(nn.Module):
    """EMA codebook (van den Oord et al. 2017, appendix), commitment loss, straight-through gradient, and a restart of
    codes that went unused for restart_after training steps (re-seeded from a random encoder output of the batch)."""

    def __init__(self, n_codes, dim, decay, commitment, restart_after, eps=1e-5, entropy=None):
        """entropy (LAM v2): {"weight", "temperature"} adds weight * (E_z[H(p(k|z))] - H(E_z[p(k|z)])), MAGVIT-v2's
        code-usage entropy objective, on L2-normalised z and codes (gradient to the encoder only); None = v1.
        weight "tied": the weight is set once by calibrate_entropy() from the first training batch (train_lam.py) so
        that |weight * term| equals the reconstruction loss there; it is then fixed and saved with the checkpoint."""
        super().__init__()
        self.n, self.decay, self.beta, self.restart_after, self.eps = n_codes, decay, commitment, restart_after, eps
        self.entropy = entropy
        self.last_entropy = {}
        self.entropy_raw = None                      # the unweighted term of the last training forward (with gradient)
        if entropy:
            w = entropy["weight"]
            self.register_buffer("entropy_weight", torch.tensor(float("nan") if w == "tied" else float(w)))
        self.register_buffer("embed", torch.zeros(n_codes, dim))
        self.register_buffer("embed_sum", torch.zeros(n_codes, dim))
        self.register_buffer("cluster_size", torch.zeros(n_codes))
        self.register_buffer("idle", torch.zeros(n_codes, dtype=torch.long))     # training steps since last used
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        self.restarts = 0

    def nearest(self, z):
        with torch.autocast(z.device.type, enabled=False):      # distances in fp32 even inside a bf16 region
            z = z.float()
            d = z.pow(2).sum(1, keepdim=True) - 2 * z @ self.embed.t() + self.embed.pow(2).sum(1)[None]
            return d.argmin(1)

    @torch.no_grad()
    def _update(self, z, idx):
        if not bool(self.initialized):
            pick = torch.randint(len(z), (self.n,), device=z.device)
            self.embed.copy_(z[pick])
            self.embed_sum.copy_(z[pick])
            self.cluster_size.fill_(1.0)
            self.initialized.fill_(True)
            idx = self.nearest(z)
        onehot = F.one_hot(idx, self.n).type_as(z)
        self.cluster_size.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
        self.embed_sum.mul_(self.decay).add_(onehot.t() @ z, alpha=1 - self.decay)
        total = self.cluster_size.sum()
        size = (self.cluster_size + self.eps) / (total + self.n * self.eps) * total
        self.embed.copy_(self.embed_sum / size[:, None])
        used = onehot.sum(0) > 0
        self.idle.add_(1).masked_fill_(used, 0)
        dead = self.idle > self.restart_after
        if dead.any():
            k = int(dead.sum())
            pick = torch.randint(len(z), (k,), device=z.device)
            self.embed[dead] = z[pick]
            self.embed_sum[dead] = z[pick]
            self.cluster_size[dead] = 1.0
            self.idle[dead] = 0
            self.restarts += k
        return idx

    def forward(self, z):
        """z (B, dim) float32 -> (quantised with straight-through gradient, code index, commitment loss)."""
        with torch.autocast(z.device.type, enabled=False):      # codebook statistics in fp32
            z = z.float()
            idx = self.nearest(z)
            if self.training:
                idx = self._update(z.detach(), idx)
            q = self.embed[idx]
            commit = self.beta * F.mse_loss(z, q.detach())
            if self.entropy and self.training:
                zn, en = F.normalize(z, dim=1), F.normalize(self.embed.detach(), dim=1)
                p = torch.softmax(-(zn[:, None] - en[None]).pow(2).sum(-1) / self.entropy["temperature"], 1)
                per_sample = -(p * p.clamp_min(1e-12).log()).sum(1).mean()          # low = confident assignments
                avg = p.mean(0)
                batch = -(avg * avg.clamp_min(1e-12).log()).sum()                   # high = codes spread over the batch
                self.entropy_raw = per_sample - batch
                if not torch.isnan(self.entropy_weight):                            # "tied" and not yet calibrated: the
                    commit = commit + self.entropy_weight * self.entropy_raw        # training loop adds the term itself
                self.last_entropy = {"per_sample": per_sample.item(), "batch": batch.item(), "weight": self.entropy_weight.item()}
            return z + (q - z).detach(), idx, commit

    def calibrate_entropy(self, recon_loss):
        """weight "tied", first training batch only: weight = recon_loss / |entropy term|. Returns the weighted term
        of this batch (to add to its loss), or None if the weight is already set."""
        if not self.entropy or not torch.isnan(self.entropy_weight):
            return None
        self.entropy_weight.fill_(float(recon_loss) / abs(self.entropy_raw.item()))
        return self.entropy_weight * self.entropy_raw


# ----------------------------------------------------------------------------- decoder
class CodeUNet(UNet):
    """model1's UNet with the code vector in the action slot: cond = MLP(Linear(code)). No noise-level inputs (the
    decoder is a deterministic regression), so those embeddings are dropped."""

    def __init__(self, context_frames, code_dim, m, code_input=False):
        """code_input (LAM v2): the code vector is also broadcast over the image and concatenated to the input frames,
        so it enters the first conv (default init, non-zero) as code_dim extra channels."""
        self.code_input = code_input
        super().__init__(in_channels=3 * context_frames + (code_dim if code_input else 0), out_channels=3, context=1, n_actions=1, widths=m["widths"],
                         blocks_per_level=m["blocks_per_level"], attn_levels=m["attn_levels"], heads=m["heads"],
                         cond_dim=m["cond_dim"], action_embed_dim=code_dim, groupnorm_groups=m["groupnorm_groups"],
                         fourier_dim=m["fourier_dim"], dropout=m["dropout"])
        del self.action_embed, self.noise_ff, self.noise_proj, self.ctx_ff, self.ctx_proj

    def forward(self, ctx, q):
        x = torch.cat([ctx, q[:, :, None, None].expand(-1, -1, *ctx.shape[-2:]).to(ctx.dtype)], 1) if self.code_input else ctx
        return self.run(x, self.cond_mlp(self.action_proj(q)))


class LAM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        L, d = cfg["lam"], cfg["data"]
        self.enc_offsets = list(L["encoder_offsets"])
        if self.enc_offsets[-1] != 0 or any(a >= b for a, b in zip(self.enc_offsets, self.enc_offsets[1:])):
            raise SystemExit("lam.encoder_offsets must be strictly increasing and end at 0 (the target frame)")
        self.encoder = ConvTrunk(3 * len(self.enc_offsets), L["encoder_widths"], L["encoder_hidden"], L["code_dim"],
                                 L["groupnorm_groups"], d["size"])
        self.vq = VectorQuantizer(L["n_codes"], L["code_dim"], L["vq_decay"], L["commitment"], L["restart_after"],
                                  entropy=L.get("entropy"))
        # LAM v2: lam.decoder_offsets = the trailing subset of data.context_offsets the decoder sees (None = all, v1)
        offs = list(d.get("context_offsets") or range(-d["context"], 0))
        dec = list(L.get("decoder_offsets") or offs)
        if dec != offs[len(offs) - len(dec):]:
            raise SystemExit(f"lam.decoder_offsets {dec} must be the trailing offsets of data.context_offsets {offs}")
        self.dec_frames = len(dec)
        self.decoder = CodeUNet(self.dec_frames, L["code_dim"], L["decoder"], code_input=bool(L.get("decoder_code_input")))
        if L.get("decoder_cond_init_std"):          # LAM v2: the code influences the output from step 0 (v1: zero init)
            for mod in self.decoder.modules():
                if hasattr(mod, "proj") and isinstance(mod.proj, nn.Linear) and mod.__class__.__name__ == "AdaGN":
                    nn.init.normal_(mod.proj.weight, std=L["decoder_cond_init_std"])
        if L.get("decoder_out_init_std"):           # LAM v2: the UNet's output conv is zero-initialised (as in model1), which
            nn.init.normal_(self.decoder.out_conv.weight, std=L["decoder_out_init_std"])   # alone blocks any code influence

    def encode(self, enc_frames):
        """(B, 3E, H, W) -> code index (B,). Inference only (eval mode: no codebook update)."""
        z = self.encoder(enc_frames).float()
        return self.vq.nearest(z)

    def forward(self, enc_frames, dec_ctx):
        """Returns (predicted target (B,3,H,W), code index (B,), commitment loss, quantised vector (B, code_dim))."""
        z = self.encoder(enc_frames).float()
        q, idx, commit = self.vq(z)
        return self.decode(dec_ctx, q), idx, commit, q

    def decode(self, dec_ctx, q):
        dec_ctx = dec_ctx[:, -3 * self.dec_frames:]          # v2: only the decoder's trailing frames (v1: all of them)
        return dec_ctx[:, -3:] + self.decoder(dec_ctx, q).float()


class Prior(nn.Module):
    """p(code of the transition that follows the newest context frame | the world model's context)."""

    def __init__(self, cfg):
        super().__init__()
        p, d = cfg["prior"], cfg["data"]
        self.net = ConvTrunk(3 * d["context"], p["widths"], p["hidden"], cfg["lam"]["n_codes"], p["groupnorm_groups"], d["size"])

    def forward(self, ctx):
        return self.net(ctx).float()


# ----------------------------------------------------------------------------- data helpers
def encoder_frames(ds, target_idx, offsets):
    """Raw bytes of the encoder's frames for these targets (a dataset.WindowDataset), through the one context rule:
    offsets reaching before the episode start read its first frame. Returns (B, E, H, W[, 3])."""
    off = torch.as_tensor(offsets, dtype=torch.long)
    f, _, _ = gather_context(ds.frames, ds.actions, target_idx, ds.first_idx(target_idx), off)
    return f


def player_weight(pac, size, lam_w, device):
    """Arm B's loss weight: 1 + lambda inside radius_px of Pac-Man in frame i-1 or frame i, else 1.
    pac (B, 2, 2) = (frame i-1, frame i) x (row, col) from the pixel detector, NaN where he was not found."""
    B = pac.shape[0]
    if not lam_w["enabled"]:
        return torch.ones(B, 1, size, size, device=device)
    rr = torch.arange(size, device=device, dtype=torch.float32)
    pac = pac.to(device)
    near = torch.zeros(B, size, size, dtype=torch.bool, device=device)
    for k in range(pac.shape[1]):
        r, c = pac[:, k, 0], pac[:, k, 1]
        d2 = (rr[None, :, None] - r[:, None, None]) ** 2 + (rr[None, None, :] - c[:, None, None]) ** 2
        near |= torch.nan_to_num(d2, nan=float("inf")) <= lam_w["radius_px"] ** 2
    return (1.0 + lam_w["lambda"] * near.float())[:, None]


def weighted_mse(pred, tgt, w):
    return ((pred - tgt) ** 2 * w).sum() / (w.sum() * pred.shape[1])


def perplexity(counts):
    p = counts.float() / counts.sum().clamp(min=1)
    return float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum()))


def cosine_lr(step, tr):
    if step < tr["warmup_steps"]:
        return tr["lr"] * (step + 1) / tr["warmup_steps"]
    t = (step - tr["warmup_steps"]) / max(1, tr["steps"] - tr["warmup_steps"])
    return tr["lr_min"] + 0.5 * (tr["lr"] - tr["lr_min"]) * (1 + math.cos(math.pi * min(1.0, t)))


# ----------------------------------------------------------------------------- play time
DIRECTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]      # the key names; keymap["dir"][c] is one of these or "NONE"


def resolve(keys, probs, c_prev, keymap):
    """Held keys -> the code for the next transition (design section 7.3).

    keys: set of held directions (subset of DIRECTIONS; two keys = a diagonal). probs (n_codes,) = the prior's
    p(code | context). c_prev: the LAM's code for the most recent transition. keymap: tools/lam_keymap.py output.
      1. the environment codes together are likely (>= tau_env)  -> the most likely of them (the world forces it)
      2. a held key's code is possible here (>= tau_legal)       -> the likeliest such code
      3. the last code is still possible                          -> keep it (the no-op: keep doing what you did)
      4. otherwise                                                 -> the likeliest code (corner / wall)
    """
    dirs, tau_legal, tau_env = keymap["dir"], keymap["tau_legal"], keymap["tau_env"]
    env = [c for c, d in enumerate(dirs) if d == "NONE"]
    if env and sum(probs[c] for c in env) >= tau_env:
        return max(env, key=lambda c: probs[c])
    cand = [c for c, d in enumerate(dirs) if d in keys]
    if cand:
        best = max(cand, key=lambda c: probs[c])
        if probs[best] >= tau_legal:
            return best
    if c_prev is not None and c_prev >= 0 and probs[c_prev] >= tau_legal:
        return int(c_prev)
    return int(max(range(len(probs)), key=lambda c: probs[c]))
