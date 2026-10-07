# tests/test_p78_comparacion_justa.py
"""La agrupacion de SUDARE I: P(II) + P(IIn) decide la familia, la mejor plantilla decide IIn."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from pipeline78.comparacion_justa import a_sudare


def test_a_sudare():
    p = pd.DataFrame({"y_true": ["II", "IIn", "Ibc", "Ia"],
                      "p_Ia": [0.0, 0.1, 0.1, 0.9], "p_II": [0.35, 0.3, 0.1, 0.05],
                      "p_IIn": [0.25, 0.3, 0.1, 0.05], "p_Ibc": [0.4, 0.3, 0.7, 0.0],
                      "best_template_clase": ["II", "IIn", "Ibc", "Ia"]})
    y3, p3, p4 = a_sudare(p)
    assert list(y3) == ["H", "H", "Ibc", "Ia"]
    assert list(p3) == ["H", "H", "Ibc", "Ia"]            # 0.35 + 0.25 > 0.4: la familia gana aunque Ibc sea el maximo
    assert list(p4) == ["II", "IIn", "Ibc", "Ia"]         # dentro de H, IIn solo si la mejor plantilla es IIn
    assert a_sudare(p.drop(columns="best_template_clase"))[2] is None
