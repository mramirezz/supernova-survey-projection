"""Modelos chicos (16 GB): GRU tipo RAPID y transformer encoder con tiempo continuo tipo ATAT/Astromer.

Entrada comun: x [B, L, N_FEAT], t [B, L] (dt en dias), mask [B, L] (True = token valido, padding a la derecha),
g [B, n_glob] (globales). Salida: logits [B, n_cls].

Tiempo continuo: embedding sinusoidal de dt con periodos geometricos entre 1 y 1000 dias, proyectado a d. No hay
posiciones enteras, porque la cadencia es irregular.
"""
import math
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TimeEmbedding(nn.Module):
    def __init__(self, d, n_freq=16, p_min=1.0, p_max=1000.0):
        super().__init__()
        periods = torch.logspace(math.log10(p_min), math.log10(p_max), n_freq)
        self.register_buffer("omega", 2 * math.pi / periods)
        self.proj = nn.Linear(2 * n_freq, d)

    def forward(self, t):
        a = t.unsqueeze(-1) * self.omega
        return self.proj(torch.cat([torch.sin(a), torch.cos(a)], dim=-1))


def _head(d_in, d, n_cls, dropout):
    return nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, n_cls))


class TransformerClf(nn.Module):
    """3 capas, d = 64, 4 cabezas, pre-norm. Pooling = media enmascarada. Globales concatenados antes de la cabeza."""

    def __init__(self, n_feat, n_glob, n_cls, d=64, heads=4, layers=3, ff=128, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(n_feat, d)
        self.temb = TimeEmbedding(d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = _head(d + n_glob, d, n_cls, dropout)

    def forward(self, x, t, mask, g):
        h = self.norm(self.enc(self.inp(x) + self.temb(t), src_key_padding_mask=~mask))
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(torch.cat([pooled, g], dim=-1))


class GRUClf(nn.Module):
    """GRU de 2 capas, oculto 64 (RAPID). Cada paso recibe el token, el embedding de dt y log(1 + gap al anterior).
    Se clasifica con el estado oculto final de la capa superior (secuencia empaquetada, sin leer el padding)."""

    def __init__(self, n_feat, n_glob, n_cls, hidden=64, layers=2, d_in=64, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(n_feat + 1, d_in)
        self.temb = TimeEmbedding(d_in)
        self.gru = nn.GRU(d_in, hidden, layers, batch_first=True, dropout=dropout)
        self.head = _head(hidden + n_glob, hidden, n_cls, dropout)

    def forward(self, x, t, mask, g):
        gap = torch.log1p(torch.diff(t, dim=1, prepend=t[:, :1]).clamp(min=0.0))
        h = self.inp(torch.cat([x, gap.unsqueeze(-1)], dim=-1)) + self.temb(t)
        lengths = mask.sum(1).to("cpu", torch.int64)
        _, hn = self.gru(pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False))
        return self.head(torch.cat([hn[-1], g], dim=-1))


def build_model(kind, n_feat, n_glob, n_cls):
    if kind == "gru":
        return GRUClf(n_feat, n_glob, n_cls)
    if kind == "transformer":
        return TransformerClf(n_feat, n_glob, n_cls)
    raise ValueError(f"modelo desconocido: {kind}")
