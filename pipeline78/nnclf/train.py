"""Entrenamiento del clasificador NN sobre las sims de la T9 final.

PESOS: w = w_z * c_clase, con c_clase tal que cada clase suma el mismo peso total en el train (balance de clases
sobre el peso volumetrico de z), normalizado a media 1. La seleccion espectroscopica S(m) todavia no existe: las
sims no estan pesadas por la probabilidad de que una SN de esa magnitud tenga espectro, y las reales de validacion
si pasaron por esa seleccion.

EARLY STOPPING: perdida ponderada sobre la validacion interna (plantillas fuera, data.split_templates). Cada curva
de validacion entra completa y en una copia raleada fija (el mismo aumento con rng seed + 1). Las reales nunca entran
al entrenamiento ni a la eleccion de la epoca.

VARIANTES DE LA LITERATURA (nn-lit-brief, todas por flag y apagadas por defecto): band_enc "lambda" (data, regla 11),
time_enc "atat" (models, TimeModulator de ATAT), gru_pool "attn" y bidir (models, ORACLE-2), trunc (data, regla 12).

JERARQUICA (--jerarquica, apagada por defecto): cabeza models.HierHead, p(Ia) contra p(CC) y p(k | CC). Perdida
hier_loss = entropia cruzada Ia contra CC (media ponderada sobre todos los objetos) + entropia cruzada de la subclase
(media ponderada SOLO sobre los CC, los Ia no entran). Los pesos son los mismos de la plana (cada clase suma lo
mismo): el nivel 1 ve Ia:CC = 1:(K - 1) y el nivel 2 clases balanceadas, asi que el prior efectivo de P combinada
sigue uniforme sobre las K clases (lo que supone calib.prior_adjustment), como en el jerarquico de clf_villar. Con
las dos medias sobre todos los objetos la suma seria la NLL plana de P. La media del segundo termino sobre los CC lo
pesa sum w / sum w_CC veces mas (K / (K - 1) con los pesos balanceados). La epoca se elige con la NLL ponderada de P
combinada (weighted_loss sobre log P), la misma cifra que en la plana. Sin el flag la ruta plana no cambia (mismos
modulos, mismo orden de construccion, misma perdida).
"""
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score
from pipeline78.nnclf import data as D
from pipeline78.nnclf.models import build_model


@dataclass
class Config:
    name: str = "nn"
    model: str = "transformer"          # gru | transformer
    use_z: bool = False
    four_classes: bool = False
    five_classes: bool = False          # Ia, II, IIb, Ibc, IIn (pisa four_classes; data.modo)
    use_magerr: bool = True
    max_epochs: int = 60
    patience: int = 8
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "auto"                # auto | cpu | mps
    threads: int = 2
    max_sims: int = 0                   # 0 = todas
    n_folds: int = 5
    fold: int = 0
    seed: int = D.SEED
    p_thin: float = 0.8
    p_ronly: float = 0.5
    max_len: int = D.MAX_LEN
    sim_run: str = str(D.SIM_RUN)
    real_dir: str = str(D.REAL_DIR)
    out_root: str = str(D.OUT_ROOT)
    band_enc: str = "onehot"            # onehot | lambda
    time_enc: str = "sin"               # sin | atat
    tm_harmonics: int = 64              # ATAT: H
    tm_tmax: float = 1500.0             # ATAT: T_max en dias
    gru_pool: str = "last"              # last | attn (ORACLE-2)
    bidir: bool = False                 # GRU bidireccional (ORACLE-2)
    trunc: str = "none"                 # none | pow2 | frac | both (ORACLE-2)
    p_trunc: float = 1.0                # prob. de truncar cada curva en cada epoca (ORACLE-2 trunca todas)
    jerarquica: bool = False            # cabezas Ia contra CC y k | CC (models.HierHead, hier_loss)

    @property
    def out(self):
        return Path(self.out_root) / self.name


def pick_device(name):
    if name == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return name


def model_opts(cfg):
    return {"time_enc": cfg.time_enc, "tm_harmonics": cfg.tm_harmonics, "tm_tmax": cfg.tm_tmax,
            "gru_pool": cfg.gru_pool, "bidir": cfg.bidir, "jerarquica": cfg.jerarquica}


def collate(items):
    """items: lista de (x, dt, g, b) de data.tokenize. Padding a la derecha, mask True = token valido."""
    B, L = len(items), max(len(it[0]) for it in items)
    x = torch.zeros(B, L, items[0][0].shape[1])
    t = torch.zeros(B, L)
    b = torch.zeros(B, L, dtype=torch.long)
    mask = torch.zeros(B, L, dtype=torch.bool)
    for j, (xi, ti, _, bi) in enumerate(items):
        n = len(xi)
        x[j, :n] = torch.from_numpy(xi)
        t[j, :n] = torch.from_numpy(ti)
        b[j, :n] = torch.from_numpy(bi)
        mask[j, :n] = True
    g = torch.from_numpy(np.stack([it[2] for it in items]))
    return x, t, mask, g, b


def encode(curves, cfg, rng=None):
    """Tokeniza. Con rng aplica antes el aumento (data.augment)."""
    out = []
    for c in curves:
        if rng is not None:
            c = D.augment(c, rng, cfg.p_thin, cfg.p_ronly, trunc=cfg.trunc, p_trunc=cfg.p_trunc)
        out.append(D.tokenize(c, cfg.max_len, cfg.use_magerr, cfg.use_z, cfg.band_enc))
    return out


def sample_weights(curves, n_cls):
    return D.balance_weights([c.y for c in curves], [c.w for c in curves], n_cls)


def logits_of(model, enc, device, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(enc), bs):
            x, t, m, g, b = (a.to(device) for a in collate(enc[i:i + bs]))
            out.append(model(x, t, m, g, b).float().cpu())
    return torch.cat(out) if out else torch.zeros(0, 0)


def weighted_loss(logits, y, w):
    ce = nn.functional.cross_entropy(logits, y, reduction="none")
    return (ce * w).sum() / w.sum()


def hier_terms(logp, y, w):
    """(Ia contra CC, k | CC) a partir de log P combinada [B, K] (clase 0 = Ia). Cada termino es una media ponderada:
    el primero sobre todos los objetos, el segundo solo sobre los CC (0 si el lote no trae CC)."""
    lcc = torch.logsumexp(logp[:, 1:], -1)                                   # log p(CC)
    cc = y > 0
    top = -torch.where(cc, lcc, logp[:, 0])
    sub = -(logp[cc].gather(1, y[cc].unsqueeze(1)).squeeze(1) - lcc[cc])    # -log p(k | CC)
    return (top * w).sum() / w.sum(), (sub * w[cc]).sum() / w[cc].sum().clamp(min=1e-12)


def hier_loss(logp, y, w):
    top, sub = hier_terms(logp, y, w)
    return top + sub


def loss_fn(cfg):
    return hier_loss if cfg.jerarquica else weighted_loss


def load_model(out_dir, device="cpu"):
    ck = torch.load(Path(out_dir) / "model.pt", map_location="cpu", weights_only=True)
    cfg = Config(**ck["config"])
    model = build_model(cfg.model, D.n_feat(cfg.band_enc), D.n_glob(cfg.use_z), len(ck["classes"]), **model_opts(cfg))
    # checkpoints de 0e50a0f: inp y temb vivian en la raiz del modelo, ahora en emb
    sd = {("emb." + k if k.split(".")[0] in ("inp", "temb") else k): v for k, v in ck["state_dict"].items()}
    model.load_state_dict(sd)
    return model.to(device).eval(), cfg, ck


def n_params(model):
    """Parametros entrenables. La cola los usa para decidir cual arquitectura es la mas simple."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _count(curves, cls):
    return {c: int(sum(k.y == i for k in curves)) for i, c in enumerate(cls)}


def train(cfg):
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = pick_device(cfg.device)
    cls = D.classes(D.modo(cfg))
    assert not cfg.jerarquica or cls[0] == "Ia", "la cabeza jerarquica supone la clase 0 = Ia"
    out = cfg.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    curves = D.load_sims(cfg.sim_run, D.modo(cfg), max_sims=cfg.max_sims or None, seed=cfg.seed)
    pairs = D.sims_table(cfg.sim_run, D.modo(cfg))[["template", "sn_type"]].itertuples(index=False)
    val_tpl = D.split_templates(pairs, cfg.n_folds, cfg.fold, cfg.seed)
    tr = [c for c in curves if c.template not in val_tpl]
    va = [c for c in curves if c.template in val_tpl]
    assert all(c.n_det() >= D.MIN_DET for c in tr)
    print(f"[nnclf] {cfg.name}: {len(tr)} train {_count(tr, cls)} | {len(va)} val interna {_count(va, cls)} | "
          f"{len(val_tpl)} plantillas fuera | device {device} | carga {time.time() - t0:.0f} s", flush=True)

    sw_tr = sample_weights(tr, len(cls))
    sw_va = sample_weights(va, len(cls))
    y_tr = torch.tensor([c.y for c in tr])
    enc_va = encode(va, cfg) + encode(va, cfg, np.random.default_rng(cfg.seed + 1))
    y_va = torch.tensor([c.y for c in va] * 2)
    w_va = torch.tensor(np.concatenate([sw_va, sw_va]), dtype=torch.float32)

    model = build_model(cfg.model, D.n_feat(cfg.band_enc), D.n_glob(cfg.use_z), len(cls), **model_opts(cfg)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = loss_fn(cfg)
    best_loss, best_ep, best_state, bad, hist = np.inf, 0, None, 0, []
    for ep in range(1, cfg.max_epochs + 1):
        te = time.time()
        model.train()
        perm = rng.permutation(len(tr))
        tot = wsum = 0.0
        for i in range(0, len(tr), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            x, t, m, g, bb = (a.to(device) for a in collate(encode([tr[j] for j in b], cfg, rng)))
            w = torch.tensor(sw_tr[b], dtype=torch.float32, device=device)
            loss = crit(model(x, t, m, g, bb), y_tr[torch.from_numpy(b)].to(device), w)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * w.sum().item()
            wsum += w.sum().item()
        row = {"epoch": ep, "train_loss": tot / max(wsum, 1e-12)}
        if len(va):
            lv = logits_of(model, enc_va, device)
            row["val_loss"] = weighted_loss(lv, y_va, w_va).item()
            row["val_bal_acc"] = balanced_accuracy_score(y_va.numpy(), lv.argmax(1).numpy(), sample_weight=w_va.numpy())
        row["s"] = time.time() - te
        hist.append(row)
        print("[nnclf] " + " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()),
              flush=True)
        score = row.get("val_loss", row["train_loss"])
        if score < best_loss - 1e-4:
            best_loss, best_ep, bad = score, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "config": asdict(cfg), "classes": cls, "best_epoch": best_ep}, out / "model.pt")
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    (out / "config.json").write_text(json.dumps({**asdict(cfg), "n_params": n_params(model),
                                                 "versions": {"torch": torch.__version__, "numpy": np.__version__}},
                                                indent=1))
    (out / "split.json").write_text(json.dumps({
        "val_templates": sorted(val_tpl), "val_keys": [c.key for c in va],
        "n_train": _count(tr, cls), "n_val": _count(va, cls), "best_epoch": best_ep, "best_val_loss": best_loss,
        "device": device, "train_s": time.time() - t0}, indent=1))
    print(f"[nnclf] mejor epoca {best_ep} (val_loss {best_loss:.4f}) -> {out}", flush=True)
    return out
