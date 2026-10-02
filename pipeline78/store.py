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


def parse_dat(path, return_crop=False):
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

    # Validar que cada bloque tenga step uniforme y encontrar overlap común
    steps = []
    for i, b in enumerate(blocks):
        w = b[:, 0]
        if len(w) < 2:
            raise ValueError(f"{path}: bloque {times[i]} tiene menos de 2 puntos")
        step = np.diff(w)
        if not np.allclose(step, step[0], rtol=0.0, atol=1e-4):
            raise ValueError(f"{path}: la grilla de lambda del bloque {times[i]} difiere del primero")
        steps.append(step[0])

    # Verificar que todos los steps sean iguales
    if not np.allclose(steps, steps[0], rtol=0.0, atol=1e-4):
        raise ValueError(f"{path}: la grilla de lambda del bloque difiere del primero")

    # Encontrar el overlap común
    w_mins = np.array([b[0, 0] for b in blocks])
    w_maxs = np.array([b[-1, 0] for b in blocks])
    w_min_overlap = np.max(w_mins)
    w_max_overlap = np.min(w_maxs)

    # Calcular cuánto se cropea en cada borde
    w_min_all = np.min(w_mins)
    w_max_all = np.max(w_maxs)
    crop_blue = w_min_overlap - w_min_all
    crop_red = w_max_all - w_max_overlap

    # Rechazar si el crop es > 50 Å en algún borde
    if crop_blue > 50.0 or crop_red > 50.0:
        raise ValueError(f"{path}: la grilla de lambda del bloque difiere del primero (crop {crop_blue:.0f} A azul, {crop_red:.0f} A rojo)")

    # Crop todos los bloques al overlap y validar
    cropped_blocks = []
    wave = None
    for i, b in enumerate(blocks):
        w = b[:, 0]
        # Encontrar índices dentro del overlap (con tolerancia 1e-4 consistente)
        mask = (w >= w_min_overlap - 1e-4) & (w <= w_max_overlap + 1e-4)
        cropped_b = b[mask]

        # Validar que la grilla dentro del overlap es correcta
        expected_wave = cropped_b[0, 0] + np.arange(cropped_b.shape[0]) * steps[0]
        if not np.allclose(cropped_b[:, 0], expected_wave, rtol=0.0, atol=1e-4):
            raise ValueError(f"{path}: la grilla de lambda del bloque difiere del primero")

        # Si es el primer bloque, guardar la wave de referencia
        if i == 0:
            wave = cropped_b[:, 0]
        else:
            # Validar que este bloque tenga exactamente la misma grilla que el primero
            if cropped_b.shape[0] != wave.size or not np.allclose(cropped_b[:, 0], wave, rtol=0.0, atol=1e-4):
                raise ValueError(f"{path}: la grilla de lambda del bloque {times[i]} difiere del primero")

        cropped_blocks.append(cropped_b)

    flux = np.vstack([b[:, 1] for b in cropped_blocks])
    order = np.argsort(times)

    times_out = np.asarray(times, dtype=float)[order]
    flux_out = flux[order]

    if return_crop:
        return times_out, wave, flux_out, crop_blue, crop_red
    else:
        return times_out, wave, flux_out


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

        # Verificar si ya existe y es válido
        skip = False
        if not force and (d / "meta.json").exists():
            meta = json.loads((d / "meta.json").read_text())
            if meta.get("md5_src") == md5:
                # Validar que los archivos .npy existan y tengan forma correcta
                time_path = d / "time.npy"
                wave_path = d / "wave.npy"
                flux_path = d / "flux.npy"
                if time_path.exists() and wave_path.exists() and flux_path.exists():
                    try:
                        # Cargar flux con mmap para validar shape
                        flux_loaded = np.load(flux_path, mmap_mode="r")
                        if flux_loaded.shape[0] == meta.get("n_epochs"):
                            report.append((r.sn, r.clase, "ya estaba"))
                            skip = True
                    except Exception:
                        pass  # Proceder a reconstruir si hay error al cargar

        if skip:
            continue

        t, w, f, crop_blue, crop_red = parse_dat(src, return_crop=True)
        save_template(d, t, w, f, dict(sn=r.sn, clase=r.clase, md5_src=md5, n_epochs=int(t.size),
                                       t_first=float(t[0]), t_last=float(t[-1]), wmin=float(w[0]),
                                       wmax=float(w[-1]), n_neg=int((f < 0).sum()),
                                       crop_blue_A=float(crop_blue), crop_red_A=float(crop_red)))
        report.append((r.sn, r.clase, f"ok {t.size} epocas"))
        print(r.clase, r.sn, report[-1][2], flush=True)
    rep = pd.DataFrame(report, columns=["sn", "clase", "estado"])
    STORE.mkdir(parents=True, exist_ok=True)
    rep.to_csv(STORE / "build_report.csv", index=False)
    return rep


if __name__ == "__main__":
    build_store()
