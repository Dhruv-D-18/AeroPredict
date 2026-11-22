"""
AeroPredict 2.0 — spatio-temporal models (pure PyTorch, no extra libraries)
==========================================================================
TemporalRNN  (ablation Model 3)  GRU or LSTM per airport — time only, no graph
STGNN        (ablation Model 4)  graph convolution (GCN or GAT) at every time step
                                 → GRU over time → graph convolution on the final state

Both models
* share an airport embedding (8-d) and the same forecasting head,
* forecast horizons h = 1, 2, 3 slots,
* output a mean delay rate AND a variance (heteroscedastic Gaussian head), so every
  forecast carries an uncertainty band (mean ± 1.96·σ).

Shapes:  x_hist [B, T_IN, N, F_HIST]   x_fut [B, H, N, F_FUT]   →   mean, logvar [B, H, N]
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """Kipf & Welling graph convolution:  H' = Â H W  (Â pre-normalised)."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, h, adj):                       # h [*, N, d]
        return torch.matmul(adj, self.lin(h))


class GATLayer(nn.Module):
    """Graph attention (Veličković et al.) — dense, masked to the route graph."""

    def __init__(self, d_in, d_out, heads=4):
        super().__init__()
        assert d_out % heads == 0
        self.h, self.dk = heads, d_out // heads
        self.lin = nn.Linear(d_in, d_out, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, self.dk) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(heads, self.dk) * 0.1)
        self.last_attention = None

    def forward(self, h, mask):                      # mask [N, N] bool
        lead = h.shape[:-2]
        N = h.shape[-2]
        z = self.lin(h).view(*lead, N, self.h, self.dk)            # [*, N, H, dk]
        s = (z * self.a_src).sum(-1)                               # [*, N, H]
        d = (z * self.a_dst).sum(-1)
        e = F.leaky_relu(d.unsqueeze(-2) + s.unsqueeze(-3), 0.2)   # [*, N_dst, N_src, H]
        e = e.masked_fill(~mask[..., None], float("-inf"))
        att = torch.softmax(e, dim=-2)
        self.last_attention = att.detach()
        out = torch.einsum("...ijh,...jhd->...ihd", att, z)
        return out.reshape(*lead, N, self.h * self.dk)


class ForecastHead(nn.Module):
    """[node state ‖ future covariates of step h] → mean & log-variance for each horizon."""

    def __init__(self, d_state, f_fut, horizons, hidden=64):
        super().__init__()
        self.horizons = horizons
        self.step_emb = nn.Embedding(horizons, 8)
        self.mlp = nn.Sequential(
            nn.Linear(d_state + f_fut + 8, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 2),
        )

    def forward(self, state, x_fut):
        # state [B, N, d]; x_fut [B, H, N, f] — channel 10 = airport's usual rate for that slot
        B, H, N, _ = x_fut.shape
        s = state.unsqueeze(1).expand(B, H, N, state.shape[-1])
        step = self.step_emb.weight[:, None, :].expand(H, N, 8).unsqueeze(0).expand(B, H, N, 8)
        out = self.mlp(torch.cat([s, x_fut, step], -1))
        logit0 = torch.logit(x_fut[..., 10].clamp(1e-3, 1 - 1e-3))
        mean = torch.sigmoid(logit0 + out[..., 0])           # residual on the airport's usual slot rate
        logvar = out[..., 1].clamp(-9.0, 0.0)
        return mean, logvar


class TemporalRNN(nn.Module):
    """Model 3 — per-airport GRU/LSTM, no information shared between airports."""

    def __init__(self, n_nodes, f_hist, f_fut, horizons, hidden=64, layers=2, cell="gru", dropout=0.1):
        super().__init__()
        self.kind = f"temporal_{cell}"
        self.node_emb = nn.Embedding(n_nodes, 8)
        self.inp = nn.Linear(f_hist + 8, hidden)
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn(hidden, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = ForecastHead(hidden, f_fut, horizons)

    def forward(self, x_hist, x_fut, adj=None, mask=None):
        B, T, N, F_ = x_hist.shape
        e = self.node_emb.weight[None, None].expand(B, T, N, 8)
        h = F.gelu(self.inp(torch.cat([x_hist, e], -1)))            # [B, T, N, H]
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        out, _ = self.rnn(h)
        state = out[:, -1].view(B, N, -1)
        return self.head(state, x_fut)


class STGNN(nn.Module):
    """Model 4 — AeroPredict 2.0 spatio-temporal graph neural network."""

    def __init__(self, n_nodes, f_hist, f_fut, horizons, hidden=64, conv="gcn", gnn_layers=2, dropout=0.1):
        super().__init__()
        self.kind = f"stgnn_{conv}"
        self.conv_type = conv
        self.node_emb = nn.Embedding(n_nodes, 8)
        self.inp = nn.Linear(f_hist + 8, hidden)
        make = (lambda: GCNLayer(hidden, hidden)) if conv == "gcn" else (lambda: GATLayer(hidden, hidden))
        self.spatial = nn.ModuleList([make() for _ in range(gnn_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(gnn_layers)])
        self.temporal = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.post = make()
        self.post_norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.head = ForecastHead(hidden, f_fut, horizons)

    def _graph(self, layer, h, adj, mask):
        return layer(h, adj) if self.conv_type == "gcn" else layer(h, mask)

    def forward(self, x_hist, x_fut, adj, mask):
        B, T, N, F_ = x_hist.shape
        e = self.node_emb.weight[None, None].expand(B, T, N, 8)
        h = F.gelu(self.inp(torch.cat([x_hist, e], -1)))            # [B, T, N, H]
        for layer, norm in zip(self.spatial, self.norms):           # spatial: every time step
            h = norm(h + self.drop(F.gelu(self._graph(layer, h, adj, mask))))
        z = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)              # temporal: per airport
        out, _ = self.temporal(z)
        state = out[:, -1].view(B, N, -1)
        state = self.post_norm(state + F.gelu(self._graph(self.post, state, adj, mask)))
        return self.head(state, x_fut)


def build_model(name, n_nodes, f_hist, f_fut, horizons, hidden=64):
    if name in ("gru", "lstm"):
        return TemporalRNN(n_nodes, f_hist, f_fut, horizons, hidden=hidden, cell=name)
    if name in ("stgnn_gcn", "stgnn_gat"):
        return STGNN(n_nodes, f_hist, f_fut, horizons, hidden=hidden, conv=name.split("_")[1])
    raise ValueError(f"unknown model {name}")


def gaussian_nll(mean, logvar, y, weight):
    """Weighted heteroscedastic Gaussian NLL (ignores cells with too few flights)."""
    nll = 0.5 * (logvar + (y - mean) ** 2 / logvar.exp() + math.log(2 * math.pi))
    return (nll * weight).sum() / weight.sum().clamp_min(1.0)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model"], cfg["n_nodes"], cfg["f_hist"], cfg["f_fut"], cfg["horizons"], cfg["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, cfg
