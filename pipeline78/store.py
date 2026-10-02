"""Biblioteca congelada (.dat de texto, 5-270 MB) -> matrices npy locales, una vez."""
import hashlib, io, json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline78.paths import LIB, STORE


def md5_file(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def stable_md5(path, reader=md5_file):
    """Dos lecturas completas: la primera hidrata el archivo de Drive, la segunda lo confirma."""
    a, b = reader(path), reader(path)
    if a != b:
        raise IOError(f"{path}: md5 distinto entre dos lecturas (archivo de Drive a medio hidratar)")
    return a


def parse_dat(path):
    times, counts, rows, n = [], [], [], 0
    with open(path) as fh:
        for line in fh:
            if line.startswith("# time"):
                if times:
                    counts.append(n)
                times.append(float(line.split(":", 1)[1]))
                n = 0
            elif line.strip() and not line.lstrip().startswith("#"):
                rows.append(line)
                n += 1
    if not times:
        raise ValueError(f"{path}: sin bloques '# time'")
    counts.append(n)
    data = pd.read_csv(io.StringIO("".join(rows)), sep=r"\s+", header=None).to_numpy(dtype=float)
    if not np.all(np.isfinite(data[:, :2])):
        raise ValueError(f"{path}: flujo o lambda no finito")
    blocks = np.split(data[:, :2], np.cumsum(counts)[:-1])
    wave = blocks[0][:, 0]
    for t, b in zip(times, blocks):
        if b.shape[0] != wave.size or not np.allclose(b[:, 0], wave, rtol=0.0, atol=1e-4):
            raise ValueError(f"{path}: la grilla de lambda del bloque {t} difiere del primero")
    flux = np.vstack([b[:, 1] for b in blocks])
    order = np.argsort(times)
    return np.asarray(times, dtype=float)[order], wave, flux[order]


def tdir(clase, sn):
    return STORE / "templates" / clase / sn


def save_template(d, time, wave, flux, meta):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "time.npy", np.asarray(time, dtype=float))
    np.save(d / "wave.npy", np.asarray(wave, dtype=float))
    np.save(d / "flux.npy", np.asarray(flux, dtype=np.float32))
    (d / "meta.json").write_text(json.dumps(meta, indent=1))


def load_template(d):
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text())
    return dict(meta, time=np.load(d / "time.npy"), wave=np.load(d / "wave.npy"),
                flux=np.load(d / "flux.npy", mmap_mode="r"))


def build_store(force=False):
    man = pd.read_csv(LIB / "MANIFIESTO.csv")
    report = []
    for r in man.itertuples():
        src = LIB / r.clase / "mangled" / f"{r.sn}.dat"
        d = tdir(r.clase, r.sn)
        md5 = stable_md5(src)
        if not force and (d / "meta.json").exists() and json.loads((d / "meta.json").read_text()).get("md5_src") == md5:
            report.append((r.sn, r.clase, "ya estaba"))
            continue
        t, w, f = parse_dat(src)
        save_template(d, t, w, f, dict(sn=r.sn, clase=r.clase, md5_src=md5, n_epochs=int(t.size),
                                       t_first=float(t[0]), t_last=float(t[-1]), wmin=float(w[0]),
                                       wmax=float(w[-1]), n_neg=int((f < 0).sum())))
        report.append((r.sn, r.clase, f"ok {t.size} epocas"))
        print(r.clase, r.sn, report[-1][2], flush=True)
    rep = pd.DataFrame(report, columns=["sn", "clase", "estado"])
    STORE.mkdir(parents=True, exist_ok=True)
    rep.to_csv(STORE / "build_report.csv", index=False)
    return rep


if __name__ == "__main__":
    build_store()
