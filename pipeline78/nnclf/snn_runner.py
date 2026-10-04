"""Corre SuperNNova dentro de ~/venvs/snn. No importa nada de pipeline78 (el venv tiene pandas 2 y el env series
pandas 3). Lo llama pipeline78/nnclf/snn.py como script:

    <venv>/bin/python snn_runner.py make_train <split.csv> -- <args de snn make_data>
    <venv>/bin/python snn_runner.py make_test -- <args de snn make_data, con --data_testing>
    <venv>/bin/python snn_runner.py train <model_path.json> -- <args de snn train_rnn>
    <venv>/bin/python snn_runner.py predict <out.csv> <model.pt> -- <args de snn validate_rnn>

make_train: la funcion propia de SuperNNova arma SNID.pickle (supernnova/data/make_dataset.py,
build_traintestval_splits) y despues se reemplazan TODAS las columnas dataset_* por el split por plantilla de
split.csv (0 train, 1 val, -1 fuera). Asi SuperNNova entrena y elige el modelo con el mismo split que la red.
train: solo supernnova.training.train_rnn.train (sin la prediccion ni las figuras que agrega la accion de la CLI).
predict: supernnova.validation.validate_rnn.get_predictions con la normalizacion guardada del modelo
(conf.get_settings_from_dump lee data_norm.json), y escribe la media por SNID de all_class* (curva completa; con
modelos bayesianos promedia las muestras).
Los .pickle que se leen (SNID.pickle, PRED_*.pickle) los escribe SuperNNova en nuestros propios dump_dir locales en
la misma corrida: no vienen de fuera, y SuperNNova no ofrece otro formato.
"""
import json
import os
import re
import sys

import torch  # noqa: F401  ANTES que pandas/pyarrow: en el orden inverso SuperNNova muere con segfault (-11) en
#               training_utils.get_data_batch (torch.FloatTensor), reproducido 2/2 el 2026-10-04 en macOS arm64
import numpy as np
import pandas as pd


def _split_args(argv):
    i = argv.index("--")
    return argv[:i], argv[i + 1:]


def _seed(settings):
    import torch
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.set_num_threads(int(os.environ.get("SNN_THREADS", "2")))


def main():
    cmd, (own, snn_args) = sys.argv[1], _split_args(sys.argv[2:])
    from supernnova import conf
    from supernnova.data import make_dataset
    sys.argv = ["snn"] + snn_args
    if cmd in ("make_train", "make_test"):
        if cmd == "make_train":
            split = pd.read_csv(own[0], dtype={"SNID": str}).set_index("SNID")["dataset"]
            orig = make_dataset.build_traintestval_splits

            def patched(settings):
                orig(settings)
                f = f"{settings.processed_dir}/SNID.pickle"
                df = pd.read_pickle(f)
                assert set(df.SNID.astype(str)) == set(split.index), "SNID de SuperNNova distintos de split.csv"
                for col in [c for c in df.columns if c.startswith("dataset_")]:
                    df[col] = df.SNID.astype(str).map(split).astype(int).to_numpy()
                df.to_pickle(f, protocol=4)
                print(f"[snn_runner] split por plantilla: {split.value_counts().to_dict()}", flush=True)

            make_dataset.build_traintestval_splits = patched
        settings = conf.get_settings("make_data")
        _seed(settings)
        make_dataset.make_dataset(settings)
    elif cmd == "train":
        from supernnova.training import train_rnn
        settings = conf.get_settings("train_rnn")
        _seed(settings)
        make_dataset.resolve_sntypes(settings)
        train_rnn.train(settings)
        model_file = f"{settings.rnn_dir}/{settings.pytorch_model_name}.pt"
        assert os.path.exists(model_file), model_file
        with open(own[0], "w") as fh:
            json.dump({"model_file": model_file, "model_name": settings.pytorch_model_name}, fh)
    elif cmd == "predict":
        from supernnova.validation import validate_rnn
        out_csv, model_file = own
        settings = conf.get_settings("validate_rnn")
        _seed(settings)
        make_dataset.resolve_sntypes(settings)
        ms = conf.get_settings_from_dump(settings, model_file)
        if settings.num_inference_samples != ms.num_inference_samples:
            ms.num_inference_samples = settings.num_inference_samples
        files = validate_rnn.get_predictions(ms, model_file=model_file)
        df = pd.read_pickle(files[0])
        cols = sorted([c for c in df.columns if re.fullmatch(r"all_class\d+", c)], key=lambda c: int(c[9:]))
        out = df.groupby("SNID")[cols].mean().reset_index()
        out["n_samples"] = df.groupby("SNID").size().to_numpy()
        out.to_csv(out_csv, index=False)
    else:
        raise SystemExit(f"comando desconocido: {cmd}")


if __name__ == "__main__":
    main()
