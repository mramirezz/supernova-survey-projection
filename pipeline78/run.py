# pipeline78/run.py
"""Runner v78. Uso:
    $PY -m pipeline78.run --run ztf_v78 --out ~/thesis_runs/ztf_v78 --fields-file ~/thesis_store/ztf_fields_1000.txt --seed 20261002 --workers 4
"""
import argparse, hashlib, json, os, subprocess, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from config import EXTINCTION_CONFIG, LUMINOSITY_CONFIG, PHILLIPS_CONFIG, SUBTYPE_FRACTIONS, LF_AFTER_HOST_DUST
from pipeline78.paths import REPO, STORE, FILTERS, DATA
from pipeline78 import bands as B, engine, sampling, project, runcfg, survey
from pipeline78.store import load_template, md5_file
from pipeline78.catalog import REF_BAND

_W = {}


def _h(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little", signed=True)


def sim_rng(seed, field, cls, k):
    return np.random.default_rng([seed, _h(field) & 0xFFFFFFFF, _h(cls) & 0xFFFFFFFF, k])


def _by_subtype(tpls, fractions, cls):
    """{subtipo: [plantillas]} solo si la clase tiene fracciones. Un subtipo con fraccion > 0 sin plantillas es error."""
    by = {}
    for t in tpls:
        by.setdefault(t.get("subtype"), []).append(t)
    for st, f in fractions.items():
        if f > 0 and not by.get(st):
            raise ValueError(f"clase {cls}: el subtipo {st!r} tiene fraccion {f} pero no hay plantillas")
    orphan = sorted(t["sn"] for st, ts in by.items() if not fractions.get(st, 0) > 0 for t in ts)
    if orphan:
        warnings.warn(f"clase {cls}: plantillas sin fraccion de subtipo, nunca se elegiran: {orphan}")
    return by


def choose_template(seed, field, cls, k, rng, tpls, fractions=None):
    """Sin fracciones: permutacion por (seed, field, cls), idem al comportamiento original (no consume rng).
    Con fracciones: sortea el subtipo con rng y elige dentro de el con permutacion por (seed, field, cls, subtipo)."""
    if not fractions:
        order = np.random.default_rng([seed, _h(field) & 0xFFFFFFFF, _h(cls) & 0xFFFFFFFF]).permutation(len(tpls))
        return tpls[order[k % len(tpls)]]
    names = sorted(st for st, f in fractions.items() if f > 0)
    p = np.array([fractions[st] for st in names], float)
    st = names[int(rng.choice(len(names), p=p / p.sum()))]
    pool = [t for t in tpls if t.get("subtype") == st]
    order = np.random.default_rng([seed, _h(field) & 0xFFFFFFFF, _h(cls) & 0xFFFFFFFF, _h(st) & 0xFFFFFFFF]).permutation(len(pool))
    return pool[order[k % len(pool)]]


def _init(cfg, seed, log_path, fields, out):
    cat = pd.read_csv(STORE / "catalog.csv")
    _W.update(cfg=cfg, seed=seed, out=Path(out), bands=B.survey_bands(cfg["survey"], cfg["bands"]), rest=B.rest_bands(),
              log=survey.load_log(log_path, fields), mw=sampling.load_mw(cfg), z=sampling.z_sampler(cfg),
              tpl={c: [load_template(p) for p in cat[cat.clase == c].sort_values("sn").store_path]
                   for c in cfg["classes"]})
    for c in cfg["classes"]:
        if SUBTYPE_FRACTIONS.get(c):
            _by_subtype(_W["tpl"][c], SUBTYPE_FRACTIONS[c], c)


def simulate(field, cls, k, epochs, mw):
    cfg, tpls = _W["cfg"], _W["tpl"][cls]
    rng = sim_rng(_W["seed"], field, cls, k)
    tpl = choose_template(_W["seed"], field, cls, k, rng, tpls, SUBTYPE_FRACTIONS.get(cls))
    z = _W["z"](rng, cls)
    if cfg["z_mode"] == "uniform_weighted":
        zmx = cfg["zmax_by_class"][cls]
        w_z = float(sampling.z_volume_weight(z, cfg["zmin"], zmx)) * (zmx - cfg["zmin"])
    else:
        w_z = 1.0
    ebv, rv = sampling.sample_ebv_host(rng, cls, tpl.get("subtype"))
    dm15 = tpl.get("dm15_B")
    M = sampling.sample_mpeak(rng, cls, dm15, tpl.get("subtype"))
    dmag = M - tpl["M_ref"]
    A_ref = 0.0
    if cls in LF_AFTER_HOST_DUST:      # LF sin corregir por host: M es el pico ya enrojecido en la banda de referencia
        A_ref = engine.host_ext_ref(tpl, ebv, rv, _W["rest"][tpl["ref_band"]])
        dmag -= A_ref
    t_rel, mags = engine.observed_lightcurves(tpl, z, ebv, rv, mw, _W["bands"], dmag)
    sim = dict(sim_id=_h(f"{field}|{cls}|{k}"), field=field, part_index=k, sn_type=cls,
               clf_class=tpl["clf_class"], template=tpl["sn"], subtype=tpl.get("subtype"), z=z, ebmv_host=ebv, rv_host=rv,
               ebmv_mw=mw, m_peak_abs=M, A_ref_host=A_ref, w_z=w_z, dm15_used=np.nan if dm15 is None else dm15, t_anchor=np.nan,
               status="no_coverage", n_rows=0, found=False, **{f"n_det_{b}": 0 for b in cfg["bands"]})
    if not mags:
        return sim, None
    t_anchor = project.anchor_time(cfg, rng, k, cfg["n_by_class"][cls], epochs)
    t_exp_rel = None     # solo Ia: sus plantillas empiezan en el primer punto, las demas ya en la explosion (tesis cap. 3)
    if cfg.get("edge_pre", "window") == "texp" and cls == "Ia":
        t_exp_rel = (tpl["t_Bmax"] - cfg["rise_Ia_days"] - tpl["t_peak"]) * (1.0 + z)
    df = project.project_one(t_rel, mags, epochs, t_anchor, rng, cfg, t_exp_rel=t_exp_rel, z=z)
    sim["t_anchor"] = t_anchor
    if df is None:
        sim["status"] = "no_epochs"
        return sim, None
    sim.update(status="ok", n_rows=len(df), found=bool(df["found"].any()),
               **{f"n_det_{b}": int(df.loc[df["filter"] == b, "detected"].sum()) for b in cfg["bands"]})
    for c in ("sim_id", "part_index", "sn_type", "template", "subtype", "z", "ebmv_host", "rv_host", "ebmv_mw",
              "m_peak_abs", "A_ref_host", "w_z", "dm15_used"):
        df[c] = sim[c]
    df["oid"] = field
    df["part_index"] = df["part_index"].astype(np.int32)
    return sim, df


def _atomic_parquet(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def run_unit(u):
    field, k0, k1 = u
    cfg, out = _W["cfg"], _W["out"]
    name = f"{field}__{k0:05d}.parquet"
    if (out / "_sims" / name).exists():
        return u, "skip"
    epochs = _W["log"].get(field)
    if not epochs:
        return u, "sin_log"
    mw = _W["mw"].get(field, cfg.get("mw_const", 0.02))
    sims, dfs = [], []
    for cls in cfg["classes"]:
        for k in range(k0, min(k1, cfg["n_by_class"][cls])):
            s, d = simulate(field, cls, k, epochs, mw)
            sims.append(s)
            if d is not None:
                dfs.append(d)
    if dfs:
        _atomic_parquet(pd.concat(dfs, ignore_index=True), out / name)
    _atomic_parquet(pd.DataFrame(sims), out / "_sims" / name)       # al final: marca la unidad como completa
    return u, f"{sum(s['status'] == 'ok' for s in sims)}/{len(sims)} ok"


def git_state():
    c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    d = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                       capture_output=True, text=True).stdout.strip()
    u = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--",
                        "pipeline78", "tests", "core", "config.py"], cwd=REPO,
                       capture_output=True, text=True).stdout.strip()
    return c, bool(d or u)


def config_hash(cfg):
    filt = {n: md5_file(FILTERS / B.SURVEY_FILES[cfg["survey"]].format(n)) for n in cfg["bands"]}
    inputs = {"log": md5_file(Path(runcfg.log_path(cfg)))}
    for f in sorted(set(cfg.get("z_files", {}).values())):
        inputs[f] = md5_file(DATA / f)
    if cfg.get("mw_mode") == "ztf_sfd":
        inputs["sfd98_cache.parquet"] = md5_file(DATA / "sfd98_cache.parquet")
    rb = B.rest_bands()           # A_ref depende de las curvas de reposo de la banda de referencia en corrida
    inputs["rest_ref_bands"] = {n: hashlib.md5(np.concatenate([rb[n].wave, rb[n].resp, [rb[n].f0]]).tobytes()).hexdigest()
                                for n in sorted({REF_BAND[c] for c in LF_AFTER_HOST_DUST})}
    blob = json.dumps(dict(cfg=cfg, catalog=md5_file(STORE / "catalog.csv"), filters=filt, inputs=inputs,
                           lf=LUMINOSITY_CONFIG, ext=EXTINCTION_CONFIG, phillips=PHILLIPS_CONFIG,
                           subtype_fractions=SUBTYPE_FRACTIONS, lf_after_host_dust=sorted(LF_AFTER_HOST_DUST)),
                      sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fields-file", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    cfg = runcfg.RUNS_CFG[a.run]
    commit, dirty = git_state()
    if dirty and not a.allow_dirty:
        sys.exit("ERROR: el repo tiene cambios sin commit. Commitea (o --allow-dirty solo para pilotos).")
    out = Path(a.out).expanduser()
    man = out / "run_manifest.json"
    h = config_hash(cfg)
    if man.exists():
        old = json.loads(man.read_text())
        if old["config_hash"] != h or old["seed"] != a.seed:
            sys.exit(f"ERROR: {out} tiene otra configuracion o semilla. No se mezcla: usa otra carpeta.")
        if old.get("git") != commit:
            sys.exit(f"ERROR: {out} se creo con otro commit ({old.get('git')} vs {commit}). No se mezcla: usa otra carpeta.")
    else:
        out.mkdir(parents=True, exist_ok=True)
        man.write_text(json.dumps(dict(run=a.run, cfg=cfg, seed=a.seed, config_hash=h, git=commit, dirty=dirty,
                                       started=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))
    fields = [l.strip() for l in open(a.fields_file) if l.strip()][: a.limit]
    us = runcfg.units(cfg, fields)
    t0 = time.time()
    sin_log = []
    with Pool(a.workers, initializer=_init, initargs=(cfg, a.seed, runcfg.log_path(cfg), fields, str(out))) as pool:
        for i, (u, st) in enumerate(pool.imap_unordered(run_unit, us), 1):
            if st == "sin_log":
                sin_log.append(u[0])
            print(f"[{i}/{len(us)}] {u[0]} {u[1]}-{u[2]} {st}  {time.time() - t0:.0f}s", flush=True)
    sin_log = sorted(set(sin_log))
    (out / "_sin_log.txt").write_text("".join(f + "\n" for f in sin_log))
    parts = [pd.read_parquet(p) for p in sorted((out / "_sims").glob("*.parquet"))]
    if not parts:
        print(f"WARNING: {len(sin_log)} campos sin log; no hay simulaciones", flush=True)
        sys.exit("ERROR: no se genero ninguna simulacion (ver _sin_log.txt).")
    sims = pd.concat(parts, ignore_index=True)
    sims.to_parquet(out / "_sims_all.parquet", index=False)
    print(sims.groupby(["sn_type", "status"]).size().to_string())
    if sin_log:
        print(f"WARNING: {len(sin_log)} campos sin log, NO simulados (ver {out / '_sin_log.txt'})", flush=True)
    return out


if __name__ == "__main__":
    main()
