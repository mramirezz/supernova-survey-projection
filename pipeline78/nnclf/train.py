"""Entrenamiento del clasificador NN sobre las sims de la T9 final.

PESOS: w = w_z * c_clase, con c_clase tal que cada clase suma el mismo peso total en el train (balance de clases
sobre el peso volumetrico de z), normalizado a media 1. La seleccion espectroscopica S(m) todavia no existe: las
sims no estan pesadas por la probabilidad de que una SN de esa magnitud tenga espectro, y las reales de validacion
si pasaron por esa seleccion.

EARLY STOPPING: perdida ponderada sobre la validacion interna (plantillas fuera, data.split_templates). Cada curva
de validacion entra completa y en una copia raleada fija (el mismo aumento con rng seed + 1). Las reales nunca entran
al entrenamiento ni a la eleccion de la epoca.
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

    @property
    def out(self):
        return Path(self.out_root) / self.name


def pick_device(name):
    if name == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return name


def collate(items):
    """items: lista de (x, dt, g) de data.tokenize. Padding a la derecha, mask True = token valido."""
    B, L = len(items), max(len(it[0]) for it in items)
    x = torch.zeros(B, L, D.N_FEAT)
    t = torch.zeros(B, L)
    mask = torch.zeros(B, L, dtype=torch.bool)
    for j, (xi, ti, _) in enumerate(items):
        n = len(xi)
        x[j, :n] = torch.from_numpy(xi)
        t[j, :n] = torch.from_numpy(ti)
        mask[j, :n] = True
    g = torch.from_numpy(np.stack([it[2] for it in items]))
    return x, t, mask, g


def encode(curves, cfg, rng=None):
    """Tokeniza. Con rng aplica antes el aumento (data.augment)."""
    out = []
    for c in curves:
        if rng is not None:
            c = D.augment(c, rng, cfg.p_thin, cfg.p_ronly)
        out.append(D.tokenize(c, cfg.max_len, cfg.use_magerr, cfg.use_z))
    return out


def sample_weights(curves, n_cls):
    return D.balance_weights([c.y for c in curves], [c.w for c in curves], n_cls)


def logits_of(model, enc, device, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(enc), bs):
            x, t, m, g = (a.to(device) for a in collate(enc[i:i + bs]))
            out.append(model(x, t, m, g).float().cpu())
    return torch.cat(out) if out else torch.zeros(0, 0)


def weighted_loss(logits, y, w):
    ce = nn.functional.cross_entropy(logits, y, reduction="none")
    return (ce * w).sum() / w.sum()


def load_model(out_dir, device="cpu"):
    ck = torch.load(Path(out_dir) / "model.pt", map_location="cpu", weights_only=True)
    cfg = Config(**ck["config"])
    model = build_model(cfg.model, D.N_FEAT, D.n_glob(cfg.use_z), len(ck["classes"]))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), cfg, ck


def _count(curves, cls):
    return {c: int(sum(k.y == i for k in curves)) for i, c in enumerate(cls)}


def train(cfg):
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = pick_device(cfg.device)
    cls = D.classes(cfg.four_classes)
    out = cfg.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    curves = D.load_sims(cfg.sim_run, cfg.four_classes, max_sims=cfg.max_sims or None, seed=cfg.seed)
    pairs = D.sims_table(cfg.sim_run, cfg.four_classes)[["template", "sn_type"]].itertuples(index=False)
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

    model = build_model(cfg.model, D.N_FEAT, D.n_glob(cfg.use_z), len(cls)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_loss, best_ep, best_state, bad, hist = np.inf, 0, None, 0, []
    for ep in range(1, cfg.max_epochs + 1):
        te = time.time()
        model.train()
        perm = rng.permutation(len(tr))
        tot = wsum = 0.0
        for i in range(0, len(tr), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            x, t, m, g = (a.to(device) for a in collate(encode([tr[j] for j in b], cfg, rng)))
            w = torch.tensor(sw_tr[b], dtype=torch.float32, device=device)
            loss = weighted_loss(model(x, t, m, g), y_tr[torch.from_numpy(b)].to(device), w)
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
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    (out / "split.json").write_text(json.dumps({
        "val_templates": sorted(val_tpl), "val_keys": [c.key for c in va],
        "n_train": _count(tr, cls), "n_val": _count(va, cls), "best_epoch": best_ep, "best_val_loss": best_loss,
        "device": device, "train_s": time.time() - t0}, indent=1))
    print(f"[nnclf] mejor epoca {best_ep} (val_loss {best_loss:.4f}) -> {out}", flush=True)
    return out
