"""Una fila por template: maximo en r de reposo (ancla), M de referencia y dm15(B)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline78.paths import STORE, DATA
from pipeline78.store import load_template
from pipeline78.bands import rest_bands, synphot, COVERAGE_MIN

CLF_CLASS = {"Ia": "Ia", "II": "II", "IIb": "II", "IIn": "II", "Ibc": "Ibc"}           # D1
REF_BAND = {"Ia": "R_rest", "II": "R_rest", "IIb": "r_rest", "IIn": "r_rest", "Ibc": "r_rest"}
# Ia: Prieto+2006 calibra en R (Bessell). IIb: Taddia+2018 en r. Ibc: Taddia (2018, 2019) mide en r (antes R por Drout+2011). II: Li+2011b mide en R (Vega).

# Decision 2026-10-02: el pico de enfriamiento por shock se excluye del ancla (se usa el pico principal de Ni).
EARLY_DAYS = 5.0          # el maximo global solo cuenta como enfriamiento si cae en los primeros 5 d
LOCAL_HALF_WINDOW = 3.0   # semiventana (d) para definir un maximo local de brillo
DIP_MIN = 0.2             # caida minima (mag) entre el pico temprano y el principal


def rest_mag(tpl, band):
    F, cov = synphot(tpl["wave"], tpl["flux"], band)
    if cov <= COVERAGE_MIN:
        raise ValueError(f"{tpl['sn']}: {band.name} no queda cubierta en reposo")
    return -2.5 * np.log10(np.clip(F, 1e-300, None) / band.f0)


def main_peak_index(t, m):
    t = np.asarray(t, float); m = np.asarray(m, float)
    i0 = int(np.argmin(m))
    if t[i0] - t[0] > EARLY_DAYS:
        return i0
    best = None
    for j in np.where(t > t[i0] + 3.0)[0]:
        win = np.abs(t - t[j]) <= LOCAL_HALF_WINDOW
        if m[j] <= m[win].min() and (best is None or m[j] < m[best]):
            if m[i0:j + 1].max() - m[j] >= DIP_MIN:
                best = int(j)
    return i0 if best is None else best


def peak_and_dm15(t, m):
    i = main_peak_index(t, m)
    tp = float(t[i])
    dm15 = float(np.interp(tp + 15.0, t, m) - m[i]) if tp + 15.0 <= t[-1] else float("nan")
    return tp, float(m[i]), bool(i == 0 or i == len(m) - 1), dm15


def build_catalog(store_dir=STORE, subtypes_csv=DATA / "ibc_subtypes.csv", ii_csv=DATA / "ii_subtypes.csv"):
    rb = rest_bands()
    sub = {}
    for c in (subtypes_csv, ii_csv):
        if Path(c).exists():
            sub.update(dict(pd.read_csv(c)[["sn", "subtype"]].values))
    rows = []
    for meta_p in sorted(Path(store_dir).glob("templates/*/*/meta.json")):
        tpl = load_template(meta_p.parent)
        cls = tpl["clase"]
        if cls in ("Ibc", "II") and tpl["sn"] not in sub:
            raise ValueError(f"{tpl['sn']}: {cls} sin subtipo en {subtypes_csv if cls == 'Ibc' else ii_csv}")
        subtype = sub[tpl["sn"]] if cls in ("Ibc", "II") else cls
        t = tpl["time"]
        m_r = rest_mag(tpl, rb["r_rest"])
        t_peak, _, edge, _ = peak_and_dm15(t, m_r)
        i_arg = int(np.argmin(m_r))
        m_ref = rest_mag(tpl, rb[REF_BAND[cls]])
        M_ref = float(m_ref[main_peak_index(t, m_ref)])
        dm15 = peak_and_dm15(t, rest_mag(tpl, rb["B_rest"]))[3] if cls == "Ia" else float("nan")
        meta = json.loads(meta_p.read_text())
        meta.update(t_peak=t_peak, peak_at_edge=edge, M_ref=M_ref, ref_band=REF_BAND[cls],
                    dm15_B=None if np.isnan(dm15) else dm15, clf_class=CLF_CLASS[cls], subtype=subtype,
                    t_peak_argmin=float(t[i_arg]), early_peak=bool(t_peak != float(t[i_arg])),
                    dm_early=float(m_r[main_peak_index(t, m_r)] - m_r[i_arg]))
        meta_p.write_text(json.dumps(meta, indent=1))
        rows.append(dict(meta, store_path=str(meta_p.parent)))
    cat = pd.DataFrame(rows)
    cat.to_csv(Path(store_dir) / "catalog.csv", index=False)
    return cat


if __name__ == "__main__":
    c = build_catalog()
    print(c.groupby("clase").size().to_dict())
    print(c[["sn", "clase", "t_peak", "peak_at_edge", "M_ref", "dm15_B", "early_peak"]].to_string(index=False))
    print("early_peak True (pico de enfriamiento excluido):")
    print(c[c.early_peak][["sn", "clase", "t_peak_argmin", "t_peak", "dm_early"]].to_string(index=False))
