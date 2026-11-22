"""
AeroPredict - Shared PyTorch DNN definition & inference helpers
==============================================================
Used by both ``train_dnn.py`` (training) and ``flask_app.py`` (serving)
so that the network definition can never drift between training and inference.

The network is a multi-task MLP:
    shared trunk  ->  classification head  (P(delay > 15 min))
                  ->  regression head      (arrival delay in minutes)
"""

from __future__ import annotations

# ── locate project folders (see core/project_paths.py) ──
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "core" / "project_paths.py").exists())
_sys.path.insert(0, str(_ROOT / "core"))
import project_paths as P

import joblib
import numpy as np
import torch
import torch.nn as nn

CLASSIFICATION_THRESHOLD_MINUTES = 15.0


class DelayNet(nn.Module):
    """Shared-trunk multi-task network for delay risk + delay minutes."""

    def __init__(self, input_dim: int, hidden=(256, 128, 64), dropout=(0.20, 0.15)):
        super().__init__()
        h1, h2, h3 = hidden
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout[0]),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout[1]),
            nn.Linear(h2, h3),
            nn.ReLU(),
        )
        self.cls_head = nn.Linear(h3, 1)   # logit for delay_label
        self.reg_head = nn.Linear(h3, 1)   # standardised Arr_Delay

    def forward(self, x):
        z = self.trunk(x)
        return self.cls_head(z).squeeze(-1), self.reg_head(z).squeeze(-1)


def _emb_dim(cardinality: int) -> int:
    """Rule of thumb embedding width (min 3, max 16)."""
    return int(min(16, max(3, round(1.6 * cardinality ** 0.56))))


class DelayNetV2(nn.Module):
    """V2 multi-task network: categorical embeddings + numeric tower.

    Differences from :class:`DelayNet` (v1):

    * Categorical fields (airline, both airports, departure slot, distance
      bucket) are learned **embeddings** instead of one-hot columns, so the
      network can generalise between similar airports/airlines.
    * Numeric inputs are batch-normalised before the trunk.
    * Wider/deeper trunk for the full 6.7M-row dataset.

    Outputs are still ``(logit, standardised_delay)`` — one forward pass gives
    both the delay probability and the delay minutes.
    """

    def __init__(self, n_numeric: int, cardinalities, hidden=(512, 256, 128),
                 dropout=(0.25, 0.15, 0.10), emb_dims=None):
        super().__init__()
        cardinalities = list(cardinalities)
        if emb_dims is None:
            emb_dims = [_emb_dim(c) for c in cardinalities]
        self.cardinalities = cardinalities
        self.emb_dims = list(emb_dims)

        self.embeddings = nn.ModuleList(
            [nn.Embedding(c, d) for c, d in zip(cardinalities, self.emb_dims)]
        )
        self.num_bn = nn.BatchNorm1d(n_numeric)

        h1, h2, h3 = hidden
        in_dim = n_numeric + sum(self.emb_dims)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout[0]),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout[1]),
            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.ReLU(),
            nn.Dropout(dropout[2]),
        )
        self.cls_head = nn.Linear(h3, 1)   # delay > 15 min logit
        self.reg_head = nn.Linear(h3, 1)   # standardised Arr_Delay

    def forward(self, x_num, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        z = torch.cat([self.num_bn(x_num)] + embs, dim=1)
        z = self.trunk(z)
        return self.cls_head(z).squeeze(-1), self.reg_head(z).squeeze(-1)


def make_features(columns, numeric_values, categorical_values):
    """Build a single feature vector for the exact columns the DNN was trained on.

    Parameters
    ----------
    columns : list[str]
        Ordered feature names stored in the preprocessor.
    numeric_values : dict[str, float]
        Values for continuous features (e.g. ``{"tavg": 8.4, ...}``).
    categorical_values : dict[str, str]
        Prefix -> level, e.g. ``{"Airline": "Delta Air Lines Inc"}``.
        The matching one-hot column is ``"Airline_<level>"``.
    """
    row = np.zeros(len(columns), dtype=np.float32)
    index = {name: i for i, name in enumerate(columns)}

    for name, value in numeric_values.items():
        if name in index:
            row[index[name]] = float(value)

    for prefix, level in categorical_values.items():
        key = f"{prefix}_{level}"
        if key in index:
            row[index[key]] = 1.0

    return row


def load_dnn(model_path=P.DNN_V1_MODEL, preprocessor_path=P.DNN_V1_PREP):
    """Load the trained DNN checkpoint + preprocessing metadata (CPU only)."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = DelayNet(
        input_dim=cfg["input_dim"],
        hidden=tuple(cfg["hidden"]),
        dropout=tuple(cfg["dropout"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    prep = joblib.load(preprocessor_path)
    prep["y_mean"] = float(prep["y_mean"])
    prep["y_std"] = float(prep["y_std"])
    return model, prep


def load_dnn_v2(model_path=P.DNN_V2_MODEL, preprocessor_path=P.DNN_V2_PREP):
    """Load the V2 network (categorical embeddings) + FeatureBuilder state."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = DelayNetV2(
        n_numeric=cfg["n_numeric"],
        cardinalities=cfg["cardinalities"],
        hidden=tuple(cfg["hidden"]),
        dropout=tuple(cfg["dropout"]),
        emb_dims=cfg["emb_dims"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    prep = joblib.load(preprocessor_path)
    prep["y_mean"] = float(prep["y_mean"])
    prep["y_std"] = float(prep["y_std"])
    return model, prep


@torch.no_grad()
def predict_dnn_v2(model, prep, x_num, x_cat):
    """Return ``(delay_probability, estimated_delay_minutes)`` for V2 inputs."""
    num = torch.from_numpy(np.asarray(x_num, dtype=np.float32))
    cat = torch.from_numpy(np.asarray(x_cat, dtype=np.int64))
    if num.ndim == 1:
        num = num[None, :]
    if cat.ndim == 1:
        cat = cat[None, :]

    logits, reg = model(num, cat)
    probs = torch.sigmoid(logits).numpy()
    minutes = np.clip(reg.numpy() * prep["y_std"] + prep["y_mean"], 0.0, 300.0)
    if probs.size == 1:
        return float(probs[0]), float(minutes[0])
    return probs, minutes


@torch.no_grad()
def predict_dnn(model, prep, features):
    """Return ``(delay_probability, estimated_delay_minutes)``.

    ``features`` may be a single 1-D vector or a 2-D batch.  Scalars are
    returned for 1-D input, arrays for 2-D input.
    """
    arr = np.asarray(features, dtype=np.float32)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]

    mean = np.asarray(prep["mean"], dtype=np.float32)
    scale = np.asarray(prep["scale"], dtype=np.float32)
    arr = (arr - mean) / scale

    logits, reg_std = model(torch.from_numpy(arr))
    probs = torch.sigmoid(logits).numpy()
    minutes = reg_std.numpy() * prep["y_std"] + prep["y_mean"]
    minutes = np.clip(minutes, 0.0, 300.0)

    if single:
        return float(probs[0]), float(minutes[0])
    return probs, minutes
