"""
RUN SN LIST (MULTIBAND)
======================
Ejecuta proyecciones multi-banda para una lista de SNe/targets definida en:
`data/sn_list_to_project.csv`

La lista tiene columnas:
- sn_name: OID/target a proyectar (ZTF oid)
- filter_band: banda "preferida" (se usa como filtro acoplado en config; en multibanda se proyecta g/r/i)
- sn_type: string tipo (p.ej. "SN II", "SN Ibc", "SN Ia")
- z: redshift (puede estar vacío)

Este runner:
- Usa el z de cada fila si es válido (z > 0)
- Si z falta/invalid, puede muestrear (fallback) o saltar (opción)
- Fija el campo/oid por fila (processing.fixed_field)
- Selecciona template aleatorio del tipo correspondiente (reproducible por seed)

Notas:
- Para "brillo realista" en II/Ibc, usa el parche de normalización de luminosidad
  configurado en `config.py` (LUMINOSITY_CONFIG).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import gc
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

import config
from core.correction import sample_extinction_by_type, sample_cosmological_redshift


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, record: dict) -> None:
    """Append a JSON record to a .jsonl file (best-effort)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # no rompemos el run por fallos de IO en progreso
        return


def _write_json(path: Path, record: dict) -> None:
    """Write a JSON file atomically-ish (best-effort)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        return


@contextmanager
def _redirect_stdout_stderr(to_file):
    """Redirige stdout+stderr a un file-like (best-effort)."""
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = to_file
        sys.stderr = to_file
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err


def _progress_write(console, msg: str) -> None:
    """Actualiza una línea tipo 'barra' en consola."""
    try:
        # limpiar resto de línea (padding)
        line = msg[:240].ljust(240)
        console.write("\r" + line)
        console.flush()
    except Exception:
        return


def _parse_sn_type(value: str) -> str:
    s = str(value).strip().lower()
    # formatos típicos: "SN II", "SN Ibc", "II", "Ibc", ...
    if "ia" in s:
        return "Ia"
    if "ibc" in s or s.endswith(" ib") or s.endswith(" ic") or " ib" in s or " ic" in s:
        return "Ibc"
    if "ii" in s:
        return "II"
    raise ValueError(f"Tipo SN no reconocido: {value!r}")


def _parse_redshift(z_value) -> Optional[float]:
    if z_value is None:
        return None
    try:
        if pd.isna(z_value):
            return None
    except Exception:
        pass
    try:
        z = float(z_value)
    except Exception:
        return None
    # negativos/cero no son físicos para este uso (y vienen como basura a veces)
    if not np.isfinite(z) or z <= 0:
        return None
    return z


def _load_targets_with_coords(data_dir: Path) -> Dict[str, Tuple[float, float]]:
    """
    Carga un mapa sn_name -> (ra_deg, dec_deg) desde ztf_targets_with_coords_multicat_summary.csv.
    """
    path = data_dir / "ztf_targets_with_coords_multicat_summary.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=["sn_name", "ra_used_deg", "dec_used_deg"])
    coords = {}
    for _, r in df.iterrows():
        name = str(r["sn_name"])
        try:
            ra = float(r["ra_used_deg"])
            dec = float(r["dec_used_deg"])
        except Exception:
            continue
        if np.isfinite(ra) and np.isfinite(dec):
            coords[name] = (ra, dec)
    return coords


def _combine_projected_csvs(survey: str, batch_id: str, run_registry: list) -> Optional[pd.DataFrame]:
    """
    Combina todas las `projected.csv` guardadas en un solo DataFrame.

    Nota: Solo se combinan las proyecciones que efectivamente se guardaron
    (i.e. runs exitosos/aceptados). Las iteraciones fallidas no tienen projected.csv.
    """
    output_root = Path(config.PATHS.get("output_dir", "outputs"))
    base_dir = output_root / "multiband_runs" / f"{survey}_{batch_id}"
    if not base_dir.exists():
        print(f"[WARNING] No existe carpeta multiband para batch: {base_dir}")
        return None

    projected_files = sorted(base_dir.rglob("projected.csv"))
    if not projected_files:
        print(f"[WARNING] No se encontraron projected.csv en {base_dir}")
        return None

    # Index rápido por iteration_label (si existe en registry)
    registry_by_label = {}
    for rec in run_registry or []:
        label = rec.get("iteration_label", None)
        if label:
            registry_by_label[str(label)] = rec

    dfs = []
    for p in projected_files:
        try:
            df_p = pd.read_csv(p)
        except Exception as e:
            print(f"[WARNING] No pude leer {p}: {e}")
            continue

        # Inferir iteration_label desde el nombre del folder: <SNNAME>_iter_0001_try_02
        iteration_label = None
        folder = p.parent.name
        if "_iter_" in folder:
            iteration_label = "iter_" + folder.split("_iter_", 1)[1]

        rec = registry_by_label.get(str(iteration_label), None) if iteration_label else None

        # Metadatos mínimos
        df_p["batch_id"] = str(batch_id)
        df_p["iteration_label"] = str(iteration_label) if iteration_label else ""
        df_p["source_projected_csv"] = str(p)

        # Enriquecer con parámetros del run si están disponibles
        if rec:
            for k in [
                "iteration_id",
                "iteration_index",
                "survey",
                "field_oid",
                "sn_type",
                "sn_name",
                "template_file",
                "redshift",
                "ebmv_host",
                "ebmv_mw",
                "required_filter",
                "min_detections_required",
                "offset_search_mode",
                "force_brighten_to_min_detections",
                "max_force_brightening_mag",
                "forced_brightening_mag",
                "forced_brightening_applied",
            ]:
                if k in rec:
                    df_p[k] = rec.get(k)

        dfs.append(df_p)

    if not dfs:
        print("[WARNING] No se pudo cargar ninguna projected.csv para combinar.")
        return None

    df_all = pd.concat(dfs, ignore_index=True)

    # Guardar outputs combinados
    out_parquet = base_dir / "all_projected_combined.parquet"
    out_csv_gz = base_dir / "all_projected_combined.csv.gz"

    try:
        df_all.to_parquet(out_parquet, index=False)
        print(f"[SAVED] Combined projected parquet: {out_parquet}")
    except Exception as e:
        print(f"[WARNING] No pude guardar parquet combinado: {e}")

    try:
        df_all.to_csv(out_csv_gz, index=False)
        print(f"[SAVED] Combined projected csv.gz: {out_csv_gz}")
    except Exception as e:
        print(f"[WARNING] No pude guardar csv.gz combinado: {e}")

    return df_all


def main():
    parser = argparse.ArgumentParser(description="Proyecta una lista de SNe (multi-banda) usando sn_list_to_project.csv")
    parser.add_argument("--csv", type=str, default=str(Path("data") / "sn_list_to_project.csv"),
                        help="CSV con la lista (default: data/sn_list_to_project.csv)")
    parser.add_argument("--survey", choices=["ZTF", "SUDARE"], default="ZTF",
                        help="Survey para proyección (default: ZTF)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla base (default: 42)")
    parser.add_argument("--redshift-max", type=float, default=None,
                        help="z_max fallback si falta z (default: config.PROCESSING_CONFIG['sn_list_redshift_max'])")
    parser.add_argument("--require-min-detections", action="store_true",
                        help="Activar reintentos hasta mínimo de detecciones (override config)")
    parser.add_argument("--min-detections", type=int, default=None,
                        help="Mínimo de detecciones requerido (default: config.PROCESSING_CONFIG['sn_list_min_detections'])")
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="Máximo de intentos por fila (default: config.PROCESSING_CONFIG['sn_list_max_attempts'])")
    parser.add_argument("--start", type=int, default=0, help="Índice inicial (0-based) (default: 0)")
    parser.add_argument("--stop", type=int, default=None, help="Índice final exclusivo (default: fin)")
    parser.add_argument("--limit", type=int, default=None, help="Máximo de filas a procesar (default: None)")
    parser.add_argument("--types", nargs="+", choices=["Ia", "Ibc", "II"], default=None,
                        help="Filtrar por tipos (ej: --types II Ibc). Default: todos los del CSV")
    parser.add_argument("--verbose", action="store_true",
                        help="Mostrar TODA la salida en consola (default: solo progreso; todo lo demás va a un log)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Ruta de log para stdout/stderr (default: outputs/batch_runs/<batch_id>/run_sn_list_multiband.log)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe CSV: {csv_path}")

    # Redshift fallback desde config (si no viene por CLI)
    redshift_max = args.redshift_max
    if redshift_max is None:
        redshift_max = float(
            config.PROCESSING_CONFIG.get(
                "sn_list_redshift_max",
                config.PROCESSING_CONFIG.get("redshift_max", 0.03),
            )
        )
    redshift_min = float(config.PROCESSING_CONFIG.get("sn_list_redshift_min", 0.01))

    # Reintentos por mínimo de detecciones (defaults desde config)
    require_min_det = bool(config.PROCESSING_CONFIG.get("sn_list_require_min_detections", False))
    if args.require_min_detections:
        require_min_det = True

    min_det = args.min_detections
    if min_det is None:
        min_det = int(config.PROCESSING_CONFIG.get("sn_list_min_detections", 7))

    max_attempts = args.max_attempts
    if max_attempts is None:
        max_attempts = int(config.PROCESSING_CONFIG.get("sn_list_max_attempts", 5))

    df = pd.read_csv(csv_path)
    expected_cols = {"sn_name", "filter_band", "sn_type", "z"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV no tiene columnas requeridas {sorted(missing)}. Columnas: {list(df.columns)}")

    # Slice
    start = max(0, int(args.start))
    stop = int(args.stop) if args.stop is not None else len(df)
    df = df.iloc[start:stop].copy()
    if args.limit is not None:
        df = df.iloc[: int(args.limit)].copy()

    # Preload coords (para EBV_MW real por campo)
    data_dir = Path(__file__).parent / "data"
    coords_map = _load_targets_with_coords(data_dir)

    # Modo consola:
    # - default: progreso solamente (captura stdout/stderr a un log)
    # - --verbose: imprime todo en consola
    verbose = bool(args.verbose)
    if not verbose:
        # silenciar BatchLogger a consola (igual queda log en batch_dir/logs/)
        os.environ["BATCH_LOG_CONSOLE"] = "0"
        # silenciar prints al importar módulos auxiliares (simple_config/dust_maps)
        os.environ["SIMPLE_CONFIG_QUIET"] = "1"
        os.environ["DUST_MAPS_QUIET"] = "1"

    # Runner único para todo el listado
    from batch_runner_multiband import MultibandBatchRunner
    runner = MultibandBatchRunner()
    runner.stats.start_batch()

    # Progreso persistente: si se cuelga, revisa estos archivos para ver "dónde quedó"
    progress_jsonl = Path(runner.batch_dir) / "progress.jsonl"
    progress_status = Path(runner.batch_dir) / "progress_status.json"
    _append_jsonl(progress_jsonl, {"ts": _utc_now_iso(), "event": "batch_start", "batch_id": runner.batch_id, "survey": args.survey})
    _write_json(progress_status, {"ts": _utc_now_iso(), "event": "batch_start", "batch_id": runner.batch_id, "survey": args.survey})

    # Log “grande” (stdout+stderr) para debug: por default, TODO va aquí y en consola solo verás progreso.
    log_path = Path(args.log_file) if args.log_file else (Path(runner.batch_dir) / "run_sn_list_multiband.log")
    log_fh = None
    # En modo progreso, escribimos la barra directo a consola (CONOUT$) para que
    # podamos redirigir stdout/stderr (incluyendo output de R/ALR) al log.
    console = sys.stdout
    if not verbose:
        try:
            console = open("CONOUT$", "w", encoding="utf-8", buffering=1)
        except Exception:
            console = sys.stdout
    if not verbose:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fh = open(log_path, "a", encoding="utf-8", buffering=1)
            console.write(f"[INFO] Modo progreso. Log completo: {log_path}\n")
            console.write(f"[INFO] Batch ID: {runner.batch_id}\n")
            console.flush()
        except Exception:
            log_fh = None

    total = len(df)
    ran = 0
    skipped = 0

    t_batch_start = time.time()
    rows_done = 0

    # Redirigir TODO stdout/stderr al log (para no spamear consola).
    # La barra de progreso sigue yendo a `console` (guardado antes).
    old_out = None
    old_err = None
    old_fd1 = None
    old_fd2 = None
    if log_fh is not None and not verbose:
        # 1) Redirección a nivel Python
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = log_fh
        sys.stderr = log_fh
        # 2) Redirección a nivel de proceso (captura output que bypass-ea sys.stdout, p.ej. R/ALR)
        try:
            old_fd1 = os.dup(1)
            old_fd2 = os.dup(2)
            os.dup2(log_fh.fileno(), 1)
            os.dup2(log_fh.fileno(), 2)
        except Exception:
            old_fd1 = None
            old_fd2 = None

    # Imports con side-effects (prints) — aquí ya quedan capturados al log si verbose=False
    from simple_config import get_sn_templates, SNType
    from dust_maps import get_sfd98_extinction_real

    # Templates disponibles
    templates_dict = get_sn_templates()
    templates_by_type = {
        "Ia": templates_dict[SNType.IA],
        "Ibc": templates_dict[SNType.IBC],
        "II": templates_dict[SNType.II],
    }

    for i, row in enumerate(df.itertuples(index=False), start=0):
        oid = str(getattr(row, "sn_name"))
        sn_type_raw = getattr(row, "sn_type")
        filt = str(getattr(row, "filter_band") or "r").strip()
        z = _parse_redshift(getattr(row, "z", None))

        sn_type = _parse_sn_type(sn_type_raw)
        if args.types is not None and sn_type not in args.types:
            skipped += 1
            continue

        seed_i = int(args.seed) + int(start) + i
        np.random.seed(seed_i)

        # z por fila (o fallback)
        if z is None:
            z = float(sample_cosmological_redshift(n_samples=1, z_min=float(redshift_min), z_max=float(redshift_max))[0])

        # Templates disponibles para esta fila
        available_templates = list(templates_by_type[sn_type])

        # Extinción host (si la usas)
        ebmv_host = float(sample_extinction_by_type(sn_type=sn_type, n_samples=1, random_state=seed_i)[0])

        # Extinción MW: usar coordenadas del OID si están disponibles
        ebmv_mw = None
        if oid in coords_map:
            ra, dec = coords_map[oid]
            ebv_val, ok = get_sfd98_extinction_real(ra, dec)
            ebmv_mw = float(ebv_val)
        else:
            # fallback conservador: usar el default del config (siempre finito)
            ebmv_mw = float(config.SN_CONFIG.get("ebmv_mw", 0.05))

        # Reintentos (misma SN intrínseca y mismo OID/z; cambia el azar del offset/noise).
        # Si falla tras n_tries, opcionalmente probamos con OTRO template.
        base_iter_idx = int(start + i)
        attempts_log = []
        chosen_attempt = None  # (template_attempt, template, attempt, seed, n_det, n_obs)
        best_overall = None    # idem, pero max(n_det) como fallback final
        tried_templates = []

        # En modo requisito, NO guardamos intentos intermedios. Solo el aceptado final.
        n_tries = int(max_attempts) if require_min_det else 1

        # Config: reintentar con otro template si no cumple el mínimo
        try_new_template = bool(config.PROCESSING_CONFIG.get("sn_list_try_new_template_on_fail", True))
        max_template_attempts = int(config.PROCESSING_CONFIG.get("sn_list_max_template_attempts", 1))
        if not require_min_det:
            try_new_template = False
            max_template_attempts = 1
        max_template_attempts = max(1, min(max_template_attempts, max(1, len(available_templates))))

        # Progreso por fila (solo en consola; lo demás queda en log)
        if verbose:
            print(f"\n[PROGRESS] ({i+1}/{total}) oid={oid} sn_type={sn_type} req_filter={filt or 'r'} z={float(z):.5f}")
        else:
            elapsed = time.time() - t_batch_start
            rate = elapsed / max(1, rows_done + 1e-9)
            eta_min = (rate * (total - rows_done)) / 60.0
            _progress_write(
                console,
                f"[{i+1}/{total}] oid={oid} type={sn_type} req={filt or 'r'} z={float(z):.5f} | ok={ran} fail={runner.stats.runs_failed} | ETA~{eta_min:.1f}m"
            )
        _append_jsonl(progress_jsonl, {
            "ts": _utc_now_iso(),
            "event": "row_start",
            "row_index": int(base_iter_idx),
            "row_in_run": int(i),
            "total_rows": int(total),
            "oid": oid,
            "sn_type": sn_type,
            "required_filter": (filt if filt else "r"),
            "redshift": float(z),
            "seed_row": int(seed_i),
        })
        _write_json(progress_status, {
            "ts": _utc_now_iso(),
            "event": "row_start",
            "row_index": int(base_iter_idx),
            "row_in_run": int(i),
            "total_rows": int(total),
            "oid": oid,
            "sn_type": sn_type,
            "required_filter": (filt if filt else "r"),
            "redshift": float(z),
            "seed_row": int(seed_i),
        })

        def _pick_template_for_attempt(tpl_attempt: int) -> str:
            """Selección reproducible de template, evitando repetidos si es posible."""
            if tpl_attempt <= 1:
                rng = np.random.default_rng(int(seed_i))
                return str(rng.choice(available_templates))
            remaining = [t for t in available_templates if t not in tried_templates]
            pool = remaining if remaining else available_templates
            rng = np.random.default_rng(int(seed_i + tpl_attempt * 100_000))
            return str(rng.choice(pool))

        # 1) Ejecutar intentos "sin guardar" para explorar offsets/noise (y templates si aplica)
        for tpl_attempt in range(1, max_template_attempts + 1):
            template = _pick_template_for_attempt(tpl_attempt)
            tried_templates.append(template)

            if not verbose:
                _progress_write(console, f"[{i+1}/{total}] oid={oid} type={sn_type} tpl={tpl_attempt}/{max_template_attempts} ...")
            _append_jsonl(progress_jsonl, {
                "ts": _utc_now_iso(),
                "event": "template_start",
                "row_index": int(base_iter_idx),
                "template_attempt": int(tpl_attempt),
                "template_file": f"{sn_type}/{template}",
            })
            _write_json(progress_status, {
                "ts": _utc_now_iso(),
                "event": "template_start",
                "row_index": int(base_iter_idx),
                "template_attempt": int(tpl_attempt),
                "template_file": f"{sn_type}/{template}",
                "oid": oid,
                "sn_type": sn_type,
            })

            best_det_this_tpl = -1
            best_this_tpl = None  # (tpl_attempt, template, attempt, attempt_seed, n_det, n_obs)

            for attempt in range(1, n_tries + 1):
                attempt_seed = int(seed_i + tpl_attempt * 100_000 + attempt * 1000)
                np.random.seed(attempt_seed)

                is_last_attempt = bool(require_min_det and attempt == n_tries)

                # Heartbeat ANTES de ejecutar (si se cuelga adentro, esto te dice cuál fue el último intento iniciado)
                if not verbose:
                    _progress_write(
                        console,
                        f"[{i+1}/{total}] oid={oid} type={sn_type} tpl={tpl_attempt}/{max_template_attempts} try={attempt}/{n_tries} last={'Y' if is_last_attempt else 'N'}"
                    )
                _append_jsonl(progress_jsonl, {
                    "ts": _utc_now_iso(),
                    "event": "attempt_start",
                    "row_index": int(base_iter_idx),
                    "template_attempt": int(tpl_attempt),
                    "attempt": int(attempt),
                    "seed": int(attempt_seed),
                    "is_last_attempt": bool(is_last_attempt),
                })
                _write_json(progress_status, {
                    "ts": _utc_now_iso(),
                    "event": "attempt_start",
                    "row_index": int(base_iter_idx),
                    "template_attempt": int(tpl_attempt),
                    "attempt": int(attempt),
                    "seed": int(attempt_seed),
                    "is_last_attempt": bool(is_last_attempt),
                    "template_file": f"{sn_type}/{template}",
                    "oid": oid,
                    "sn_type": sn_type,
                })

                iteration_params_try = {
                    "iteration_id": f"row_{base_iter_idx:04d}_tpl_{tpl_attempt:02d}_try_{attempt:02d}",
                    "iteration_index": base_iter_idx,
                    "total_iterations": int(start + total),
                    "seed": attempt_seed,
                    "attempt": attempt,
                    "batch_id": runner.batch_id,
                    "survey": args.survey,
                    "filter_band": filt if filt else "r",
                    "sn_name": template.replace(".dat", ""),
                    "sn_type": sn_type,
                    "template_file": f"{sn_type}/{template}",
                    "redshift": float(z),
                    "ebmv_host": ebmv_host,
                    "ebmv_mw": ebmv_mw,
                    "extinction_total": float(ebmv_host + ebmv_mw),
                    "field_oid": oid,
                    # solo guardar si NO estamos en modo requisito
                    "save_outputs": (not require_min_det),
                    # Semilla fija para M_peak por fila (no por intento ni template)
                    "luminosity_random_seed": int(seed_i),
                    # Criterio mínimo por banda (para multiband_field_projection)
                    "required_filter": (filt if filt else "r"),
                    "min_detections_required": int(min_det),
                    # Fallback anti-azar: el ÚLTIMO intento de CADA template usa grid search + “último recurso” limitado
                    "offset_search_mode": ("grid" if is_last_attempt else "random"),
                    "force_brighten_to_min_detections": bool(is_last_attempt),
                    "max_force_brightening_mag": float(config.PROCESSING_CONFIG.get("sn_list_force_max_brightening_mag", 3.0)),
                }

                success_try, results_try = runner.execute_single_run(iteration_params_try)
                det_by_filter = results_try.get("detections_by_filter", {}) or {}
                req_f = (filt if filt else "r")
                n_det_i = int(det_by_filter.get(req_f, 0))
                n_obs_i = int(results_try.get("n_observations", results_try.get("observations", 0)) or 0)

                if not verbose:
                    _progress_write(
                        console,
                        f"[{i+1}/{total}] oid={oid} type={sn_type} tpl={tpl_attempt}/{max_template_attempts} try={attempt}/{n_tries} det={n_det_i}/{min_det} obs={n_obs_i}"
                    )
                attempts_log.append({
                    "template_attempt": tpl_attempt,
                    "template_file": f"{sn_type}/{template}",
                    "attempt": attempt,
                    "seed": attempt_seed,
                    "success": bool(success_try),
                    "n_detections": n_det_i,
                    "n_observations": n_obs_i,
                })

                _append_jsonl(progress_jsonl, {
                    "ts": _utc_now_iso(),
                    "event": "attempt_end",
                    "row_index": int(base_iter_idx),
                    "template_attempt": int(tpl_attempt),
                    "attempt": int(attempt),
                    "seed": int(attempt_seed),
                    "success": bool(success_try),
                    "n_detections_required_filter": int(n_det_i),
                    "n_observations": int(n_obs_i),
                })
                _write_json(progress_status, {
                    "ts": _utc_now_iso(),
                    "event": "attempt_end",
                    "row_index": int(base_iter_idx),
                    "template_attempt": int(tpl_attempt),
                    "attempt": int(attempt),
                    "seed": int(attempt_seed),
                    "success": bool(success_try),
                    "n_detections_required_filter": int(n_det_i),
                    "n_observations": int(n_obs_i),
                    "template_file": f"{sn_type}/{template}",
                    "oid": oid,
                    "sn_type": sn_type,
                })

                if bool(success_try) and n_det_i > best_det_this_tpl:
                    best_det_this_tpl = n_det_i
                    best_this_tpl = (tpl_attempt, template, attempt, attempt_seed, n_det_i, n_obs_i)

                # mantener mejor global como fallback si todo falla
                if best_this_tpl is not None:
                    if best_overall is None or best_this_tpl[4] > best_overall[4]:
                        best_overall = best_this_tpl

                if not require_min_det:
                    # modo normal: ya quedó guardado (save_outputs=True) en el único intento
                    chosen_attempt = (tpl_attempt, template, attempt, attempt_seed, n_det_i, n_obs_i)
                    break

                if bool(success_try) and n_det_i >= int(min_det):
                    chosen_attempt = (tpl_attempt, template, attempt, attempt_seed, n_det_i, n_obs_i)
                    break

            if chosen_attempt is not None:
                break

            # Si no cumplió con este template, probar otro (si está activado)
            if not try_new_template:
                break

        # 2) Si estamos en modo requisito y logramos cumplir, re-ejecutar SOLO ese intento con guardado ON
        final_success = False
        final_results = {}
        final_iteration_params = None

        if require_min_det and chosen_attempt is not None and chosen_attempt[4] >= int(min_det):
            tpl_attempt, template, attempt, attempt_seed, _, _ = chosen_attempt
            np.random.seed(int(attempt_seed))
            is_last_attempt = bool(require_min_det and int(attempt) == int(n_tries))

            final_iteration_params = {
                "iteration_id": f"row_{base_iter_idx:04d}_tpl_{tpl_attempt:02d}_try_{attempt:02d}",
                "iteration_index": base_iter_idx,
                "total_iterations": int(start + total),
                "seed": int(attempt_seed),
                "attempt": int(attempt),
                "batch_id": runner.batch_id,
                "survey": args.survey,
                "filter_band": filt if filt else "r",
                "sn_name": template.replace(".dat", ""),
                "sn_type": sn_type,
                "template_file": f"{sn_type}/{template}",
                "redshift": float(z),
                "ebmv_host": ebmv_host,
                "ebmv_mw": ebmv_mw,
                "extinction_total": float(ebmv_host + ebmv_mw),
                "field_oid": oid,
                "save_outputs": True,
                "luminosity_random_seed": int(seed_i),
                "required_filter": (filt if filt else "r"),
                "min_detections_required": int(min_det),
                "offset_search_mode": ("grid" if is_last_attempt else "random"),
                "force_brighten_to_min_detections": bool(is_last_attempt),
                "max_force_brightening_mag": float(config.PROCESSING_CONFIG.get("sn_list_force_max_brightening_mag", 3.0)),
            }
            final_success, final_results = runner.execute_single_run(final_iteration_params)

        # 3) Construir registro final (solo el último/aceptado si corresponde)
        if require_min_det:
            met = bool(chosen_attempt is not None and chosen_attempt[4] >= int(min_det))
            if not met:
                # No útil -> no guardamos outputs, pero dejamos trazabilidad en registry
                # Si probamos templates, reportamos el "mejor" global como diagnóstico.
                if best_overall is not None:
                    tpl_attempt_best, template_best, attempt_best, seed_best, det_best, obs_best = best_overall
                    template_file_best = f"{sn_type}/{template_best}"
                    sn_name_best = template_best.replace(".dat", "")
                else:
                    tpl_attempt_best, attempt_best, seed_best, det_best, obs_best = (None, None, None, 0, 0)
                    template_file_best = f"{sn_type}/{template}"
                    sn_name_best = template.replace(".dat", "")
                best_record = {
                    "iteration_id": f"row_{base_iter_idx:04d}",
                    "iteration_index": base_iter_idx,
                    "total_iterations": int(start + total),
                    "seed": int(seed_i),
                    "batch_id": runner.batch_id,
                    "survey": args.survey,
                    "filter_band": filt if filt else "r",
                    "sn_name": sn_name_best,
                    "sn_type": sn_type,
                    "template_file": template_file_best,
                    "redshift": float(z),
                    "ebmv_host": ebmv_host,
                    "ebmv_mw": ebmv_mw,
                    "extinction_total": float(ebmv_host + ebmv_mw),
                    "field_oid": oid,
                    "success": False,
                    "error": f"Min detections not met after {n_tries} attempts"
                             + (f" and {max_template_attempts} template attempts" if try_new_template else ""),
                    "n_detections": int(det_best if det_best is not None else 0),
                    "n_observations": int(obs_best if obs_best is not None else 0),
                    "required_filter": filt,
                    "tried_templates": tried_templates,
                    "best_template_attempt": tpl_attempt_best,
                    "best_attempt": attempt_best,
                    "best_attempt_seed": seed_best,
                }
            else:
                # Guardamos SOLO el intento final (ya re-ejecutado con save_outputs=True)
                best_record = {**final_iteration_params, **final_results, "success": bool(final_success)}
        else:
            # modo normal: el intento único ya guardó outputs y results_try es final
            best_record = {**iteration_params_try, **results_try, "success": bool(success_try)}

        # Enriquecer el registro final con info del criterio
        best_record["require_min_detections"] = bool(require_min_det)
        best_record["min_detections_required"] = int(min_det)
        best_record["max_attempts"] = int(n_tries)
        # Guardar attempts_log a disco para no inflar RAM en corridas largas
        try:
            attempts_dir = Path(runner.batch_dir) / "attempts"
            attempts_file = attempts_dir / f"row_{base_iter_idx:04d}_attempts.json"
            _write_json(attempts_file, {
                "row_index": int(base_iter_idx),
                "oid": oid,
                "sn_type": sn_type,
                "required_filter": (filt if filt else "r"),
                "attempts_log": attempts_log,
            })
            best_record["attempts_log_file"] = str(attempts_file)
            best_record["attempts_log_len"] = int(len(attempts_log))
        except Exception:
            best_record["attempts_log_file"] = ""
            best_record["attempts_log_len"] = int(len(attempts_log))
        # No guardar el detalle completo en memoria
        best_record["attempts_log"] = None
        # En modo "por banda", el criterio se aplica a la banda del CSV (filt)
        if require_min_det:
            det_by_filter_final = best_record.get("detections_by_filter", {}) or {}
            n_det_required_filter = int(det_by_filter_final.get(filt, best_record.get("n_detections", 0)) or 0)
            best_record["n_detections_required_filter"] = n_det_required_filter
            best_record["min_detections_met"] = (n_det_required_filter >= int(min_det))
        else:
            best_record["min_detections_met"] = True
        if require_min_det:
            best_record["required_filter"] = filt
            best_record["min_detections_mode"] = "per_filter"

        runner.run_registry.append(best_record)

        if best_record.get("success", False) and best_record.get("min_detections_met", True):
            runner.stats.add_successful_run(
                float(best_record.get("execution_time", 0.0)),
                int(best_record.get("n_detections", 0)),
                int(best_record.get("n_observations", 0)),
                sn_type,
            )
            ran += 1
        else:
            runner.stats.add_failed_run(best_record["iteration_id"], str(best_record.get("error", "Unknown error")))

        rows_done += 1
        elapsed = time.time() - t_batch_start
        rate = elapsed / max(1, rows_done)
        eta_min = (rate * (total - rows_done)) / 60.0
        _append_jsonl(progress_jsonl, {
            "ts": _utc_now_iso(),
            "event": "row_end",
            "row_index": int(base_iter_idx),
            "success": bool(best_record.get("success", False) and best_record.get("min_detections_met", True)),
            "rows_done": int(rows_done),
            "total_rows": int(total),
            "eta_minutes_estimate": float(eta_min),
        })
        _write_json(progress_status, {
            "ts": _utc_now_iso(),
            "event": "row_end",
            "row_index": int(base_iter_idx),
            "success": bool(best_record.get("success", False) and best_record.get("min_detections_met", True)),
            "rows_done": int(rows_done),
            "total_rows": int(total),
            "eta_minutes_estimate": float(eta_min),
        })

        # Liberación explícita por fila
        try:
            del attempts_log
            del best_record
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass

    runner.stats.end_batch()
    runner.save_batch_results(batch_config={"sn_list_csv": str(csv_path), "seed": args.seed, "redshift_max": redshift_max})
    _combine_projected_csvs(survey=args.survey, batch_id=runner.batch_id, run_registry=runner.run_registry)

    _append_jsonl(progress_jsonl, {"ts": _utc_now_iso(), "event": "batch_end", "batch_id": runner.batch_id})
    _write_json(progress_status, {"ts": _utc_now_iso(), "event": "batch_end", "batch_id": runner.batch_id})

    # Restaurar stdout/stderr si fueron redirigidos al log
    if old_out is not None:
        try:
            sys.stdout = old_out
            sys.stderr = old_err
        except Exception:
            pass
    # Restaurar fds 1/2 si fueron redirigidos (para no romper la consola en futuras corridas)
    if old_fd1 is not None:
        try:
            os.dup2(old_fd1, 1)
            os.close(old_fd1)
        except Exception:
            pass
    if old_fd2 is not None:
        try:
            os.dup2(old_fd2, 2)
            os.close(old_fd2)
        except Exception:
            pass

    if not verbose:
        # cerrar barra
        try:
            console.write("\n")
            console.flush()
        except Exception:
            pass
        console.write(f"[OK] Lista procesada. Ejecutados={ran} Saltados={skipped} Total={total} BatchID={runner.batch_id}\n")
        console.flush()
    else:
        print("\n" + "=" * 60)
        print("[OK] Lista procesada")
        print(f"   Ejecutados: {ran}")
        print(f"   Saltados: {skipped}")
        print(f"   Total considerado: {total}")
        print(f"   Batch ID: {runner.batch_id}")
        print("=" * 60)

    if log_fh is not None:
        try:
            log_fh.flush()
            log_fh.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

