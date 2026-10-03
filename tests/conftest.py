# tests/conftest.py
import os, sys, importlib
from pathlib import Path
import pytest

_ENV = ("P78_STORE", "P78_RUNS")
_ORDEN = ("store", "catalog", "sampling", "survey", "runcfg", "run")
_REPO = str(Path(__file__).resolve().parents[1])


@pytest.fixture(autouse=True)
def _repo_primero_en_path():
    """pilot_report y real_to_parquet ponen ZLF (con su propio config.py) al frente de sys.path. Los workers de
    multiprocessing (spawn) heredan ese orden: sin esto importan el config equivocado, mueren y el Pool los relanza."""
    if sys.path[0] != _REPO:
        while _REPO in sys.path:
            sys.path.remove(_REPO)
        sys.path.insert(0, _REPO)


@pytest.fixture(autouse=True)
def _aislar_p78_env():
    """Varios tests apuntan P78_STORE/P78_RUNS a un tmp y recargan los modulos. Al terminar se restauran y se recarga."""
    antes = {k: os.environ.get(k) for k in _ENV}
    yield
    if any(os.environ.get(k) != v for k, v in antes.items()):
        for k, v in antes.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        importlib.reload(sys.modules["pipeline78.paths"])
        for n in _ORDEN:
            if f"pipeline78.{n}" in sys.modules:
                importlib.reload(sys.modules[f"pipeline78.{n}"])
