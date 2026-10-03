"""Compara dos features.csv de run_parquet.py sobre las mismas tareas (benchmark del MCMC)."""
import sys
import numpy as np
import pandas as pd

KEYS = ["oid", "part_index", "sn_type", "filter_band"]
PARS = ["f", "t_rise", "t_fall", "gamma"]


def compare(csv_a, csv_b):
    a, b = pd.read_csv(csv_a), pd.read_csv(csv_b)
    # Drop duplicates on merge keys to ensure clean comparison
    a = a.drop_duplicates(subset=KEYS, keep='first')
    b = b.drop_duplicates(subset=KEYS, keep='first')
    m = a.merge(b, on=KEYS, suffixes=("_a", "_b"))
    rows = []
    for p in PARS:
        sig = np.sqrt(m[f"{p}_err_a"] ** 2 + m[f"{p}_err_b"] ** 2).replace(0, np.nan)
        r = (m[f"{p}_a"] - m[f"{p}_b"]).abs() / sig
        rows.append(dict(par=p, n=len(m), mediana=r.median(), p90=r.quantile(0.9)))
    dA = (np.log10(m["A_a"]) - np.log10(m["A_b"])).abs()
    rows.append(dict(par="log10A", n=len(m), mediana=dA.median(), p90=dA.quantile(0.9)))
    rows.append(dict(par="tiempo_b/a", n=len(m), mediana=m.elapsed_s_b.median() / m.elapsed_s_a.median(), p90=np.nan))
    rows.append(dict(par="solo_en_a", n=len(a) - len(m), mediana=np.nan, p90=np.nan))
    rows.append(dict(par="solo_en_b", n=len(b) - len(m), mediana=np.nan, p90=np.nan))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(compare(sys.argv[1], sys.argv[2]).round(3).to_string(index=False))
