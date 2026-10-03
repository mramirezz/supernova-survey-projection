# tests/conftest.py
import os, sys, importlib
import pytest

_ENV = ("P78_STORE", "P78_RUNS")
_ORDEN = ("store", "catalog", "sampling", "survey", "runcfg", "run")


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
