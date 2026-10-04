"""Modelos chicos (16 GB): GRU tipo RAPID y transformer encoder con tiempo continuo.

Entrada comun: x [B, L, n_feat], t [B, L] (dt en dias), mask [B, L] (True = token valido, padding a la derecha),
g [B, n_glob] (globales), b [B, L] (indice de banda, 0 = g, 1 = r). Salida: logits [B, n_cls] (con jerarquica, log P).

Tiempo (time_enc):
- "sin" (default): embedding sinusoidal de dt con periodos geometricos entre 1 y 1000 dias, proyectado a d y SUMADO
  a la proyeccion lineal del token. No hay posiciones enteras, porque la cadencia es irregular.
- "atat": TimeModulator de ATAT (Cabrera-Vives et al. 2024, 2024A&A...689A.289C, Sec. 2.2.1, ecs. 1 y 2; codigo
  oficial alercebroker/ATAT, layers/time_modulator.py, commit 0db532f). TM(x, t) = LL(x) * gamma1_b(t) + gamma2_b(t),
  con gamma una serie de Fourier de H armonicos FIJOS h / T_max y coeficientes APRENDIDOS, un juego por banda.
  Como en el codigo, h = 0..H-1 (el paper escribe h = 1..H) y los coeficientes se inicializan con randn. H = 64 y
  T_max = 1500 d son los del paper (Sec. 2.2.5). Reemplaza al embedding sinusoidal (no se suma nada mas).
  NO es una replica de ATAT, es el TimeModulator de ATAT aplicado a nuestro token (revision H3):
  (1) En ATAT LL(x) = W_TM x con x = (flujo, error), una matriz E x 2 SIN sesgo (ec. 1 y nota 9). Aca LL es un
      nn.Linear CON sesgo sobre el token completo [dt/100, m - m_ref, 10 sigma_m, es_UL, banda] (en la GRU ademas
      log(1 + gap)), asi que el tiempo entra tambien linealmente, y la banda va en el token y en el juego de
      coeficientes.
  (2) En ATAT t son los dias desde el primer punto de fotometria forzada. Aca son los dias desde la primera deteccion
      (marco observado), negativos en los UL previos (hasta -60 d).
  (3) Magnitudes relativas a la mediana y UL con bandera, no flujo de fotometria forzada.
  (4) Sin token [CLS], sin rama tabular ni QFT: el transformer promedia con mascara.

Pooling de la GRU (gru_pool):
- "last" (default): estado oculto final de la capa superior (concatena las dos direcciones si es bidireccional).
- "attn": attention pooling de ORACLE-2 (Shah et al. 2026, 2026arXiv260700228S, ec. 1; repo dev-ved30/Oracle,
  GRU_Improved._attention_pool): e_t = v^T tanh(W h_t + b), a = softmax enmascarado, c = sum a_t h_t sobre las
  salidas de la capa superior. ORACLE-2 lo usa sobre una GRU bidireccional (bidir=True).

Cabeza (jerarquica):
- False (default): una cabeza, softmax plano sobre las clases. Ruta identica a la de antes del flag.
- True (HierHead): dos cabezas sobre el mismo encoder, p(Ia) contra p(CC) y p(k | CC) sobre las demas clases (II,
  Ibc y con four_classes IIn). La clase 0 tiene que ser Ia (data.classes). La salida es log P combinada,
  log P(Ia) = log p(Ia) y log P(k) = log p(CC) + log p(k | CC). Como sum P = 1, softmax(log P) = P y evaluate, calib
  y ensemble la usan como logits sin cambios. La perdida es train.hier_loss. Es la version de red del jerarquico Ia
  contra CC de clf_villar (hier_*_Ia), con las dos cabezas entrenadas juntas.
"""
import math
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

N_BANDS = 2


class TimeEmbedding(nn.Module):
    def __init__(self, d, n_freq=16, p_min=1.0, p_max=1000.0):
        super().__init__()
        periods = torch.logspace(math.log10(p_min), math.log10(p_max), n_freq)
        self.register_buffer("omega", 2 * math.pi / periods)
        self.proj = nn.Linear(2 * n_freq, d)

    def forward(self, t):
        a = t.unsqueeze(-1) * self.omega
        return self.proj(torch.cat([torch.sin(a), torch.cos(a)], dim=-1))


class TimeModulator(nn.Module):
    """ATAT, ecs. 1 y 2: e * gamma1_b(t) + gamma2_b(t), un juego de coeficientes de Fourier por banda."""

    def __init__(self, d, n_bands=N_BANDS, harmonics=64, t_max=1500.0):
        super().__init__()
        self.alpha_sin = nn.Parameter(torch.randn(n_bands, harmonics, d))
        self.alpha_cos = nn.Parameter(torch.randn(n_bands, harmonics, d))
        self.beta_sin = nn.Parameter(torch.randn(n_bands, harmonics, d))
        self.beta_cos = nn.Parameter(torch.randn(n_bands, harmonics, d))
        self.register_buffer("h", torch.arange(harmonics, dtype=torch.float32))
        self.t_max = float(t_max)

    def forward(self, e, t, b):
        a = 2 * math.pi * t.unsqueeze(-1) * self.h / self.t_max            # [B, L, H]
        s, c = torch.sin(a), torch.cos(a)
        out = torch.zeros_like(e)
        for k in range(self.alpha_sin.shape[0]):
            g1 = s @ self.alpha_sin[k] + c @ self.alpha_cos[k]
            g2 = s @ self.beta_sin[k] + c @ self.beta_cos[k]
            out = torch.where((b == k).unsqueeze(-1), e * g1 + g2, out)
        return out


class _Embed(nn.Module):
    """Token -> R^d con el tiempo segun time_enc."""

    def __init__(self, n_in, d, time_enc="sin", tm_harmonics=64, tm_tmax=1500.0):
        super().__init__()
        self.inp = nn.Linear(n_in, d)
        self.time_enc = time_enc
        if time_enc == "sin":
            self.temb = TimeEmbedding(d)
        elif time_enc == "atat":
            self.tmod = TimeModulator(d, harmonics=tm_harmonics, t_max=tm_tmax)
        else:
            raise ValueError(f"time_enc desconocido: {time_enc}")

    def forward(self, x, t, b):
        e = self.inp(x)
        return e + self.temb(t) if self.time_enc == "sin" else self.tmod(e, t, b)


def _head(d_in, d, n_cls, dropout):
    return nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, n_cls))


def combine_log_probs(top, sub):
    """top [B, 2] logits de (Ia, CC), sub [B, K - 1] logits de k | CC -> log P [B, K] con la clase 0 = Ia."""
    lt = torch.log_softmax(top, -1)
    return torch.cat([lt[:, :1], lt[:, 1:] + torch.log_softmax(sub, -1)], -1)


class HierHead(nn.Module):
    """Ia contra CC y la subclase dentro de CC. forward da log P combinada (combine_log_probs)."""

    def __init__(self, d_in, d, n_cls, dropout):
        super().__init__()
        self.top = _head(d_in, d, 2, dropout)
        self.sub = _head(d_in, d, n_cls - 1, dropout)

    def heads(self, h):
        return self.top(h), self.sub(h)

    def forward(self, h):
        return combine_log_probs(*self.heads(h))


def make_head(d_in, d, n_cls, dropout, jerarquica=False):
    return HierHead(d_in, d, n_cls, dropout) if jerarquica else _head(d_in, d, n_cls, dropout)


class TransformerClf(nn.Module):
    """3 capas, d = 64, 4 cabezas, pre-norm. Pooling = media enmascarada. Globales concatenados antes de la cabeza."""

    def __init__(self, n_feat, n_glob, n_cls, d=64, heads=4, layers=3, ff=128, dropout=0.1, time_enc="sin",
                 tm_harmonics=64, tm_tmax=1500.0, jerarquica=False, **_):
        super().__init__()
        self.emb = _Embed(n_feat, d, time_enc, tm_harmonics, tm_tmax)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = make_head(d + n_glob, d, n_cls, dropout, jerarquica)

    def forward(self, x, t, mask, g, b=None):
        b = torch.zeros_like(t, dtype=torch.long) if b is None else b
        h = self.norm(self.enc(self.emb(x, t, b), src_key_padding_mask=~mask))
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(torch.cat([pooled, g], dim=-1))


class GRUClf(nn.Module):
    """GRU de 2 capas, oculto 64 (RAPID). Cada paso recibe el token, el tiempo y log(1 + gap al anterior). Secuencia
    empaquetada (no se lee el padding). Pooling "last" o "attn" (ORACLE-2), opcionalmente bidireccional."""

    def __init__(self, n_feat, n_glob, n_cls, hidden=64, layers=2, d_in=64, dropout=0.1, time_enc="sin",
                 tm_harmonics=64, tm_tmax=1500.0, gru_pool="last", bidir=False, jerarquica=False, **_):
        super().__init__()
        if gru_pool not in ("last", "attn"):
            raise ValueError(f"gru_pool desconocido: {gru_pool}")
        self.emb = _Embed(n_feat + 1, d_in, time_enc, tm_harmonics, tm_tmax)
        self.gru = nn.GRU(d_in, hidden, layers, batch_first=True, dropout=dropout, bidirectional=bidir)
        self.pool, self.bidir = gru_pool, bidir
        h_out = hidden * (2 if bidir else 1)
        if gru_pool == "attn":
            self.attn_W = nn.Linear(h_out, hidden)
            self.attn_v = nn.Linear(hidden, 1, bias=False)
        self.head = make_head(h_out + n_glob, hidden, n_cls, dropout, jerarquica)

    def forward(self, x, t, mask, g, b=None):
        b = torch.zeros_like(t, dtype=torch.long) if b is None else b
        gap = torch.log1p(torch.diff(t, dim=1, prepend=t[:, :1]).clamp(min=0.0))
        h = self.emb(torch.cat([x, gap.unsqueeze(-1)], dim=-1), t, b)
        lengths = mask.sum(1).to("cpu", torch.int64)
        out, hn = self.gru(pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False))
        if self.pool == "last":
            z = torch.cat([hn[-2], hn[-1]], dim=-1) if self.bidir else hn[-1]
        else:
            o, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
            e = self.attn_v(torch.tanh(self.attn_W(o))).squeeze(-1).masked_fill(~mask, float("-inf"))
            z = (torch.softmax(e, dim=1).unsqueeze(-1) * o).sum(1)
        return self.head(torch.cat([z, g], dim=-1))


def build_model(kind, n_feat, n_glob, n_cls, **opts):
    if kind == "gru":
        return GRUClf(n_feat, n_glob, n_cls, **opts)
    if kind == "transformer":
        return TransformerClf(n_feat, n_glob, n_cls, **opts)
    raise ValueError(f"modelo desconocido: {kind}")
