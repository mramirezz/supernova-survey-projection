# Copilot Instructions — Supernova ZTF Projection Pipeline

Use these repository-specific instructions for coding tasks in this project.

## Build, test, and lint commands

### Environment setup
```bash
conda create -n projection python=3.10
conda activate projection
pip install -r requirements.txt
```

### Main run commands
```bash
# One OID (30 simulations: 3 SN types × 10 deterministic partitions)
python run_per_field.py --oid ZTF18aaqeasu --seed 42

# First N fields with minimum observations
python run_per_field.py --n-fields 100 --min-obs 50

# Full dataset
python run_per_field.py
```

### Validation / test-like checks
```bash
# Validate a whole run directory
python verify_output.py outputs/per_field/<run_dir>/

# Validate a single field output (strict checks)
python verify_output.py outputs/per_field/<run_dir>/ZTF17aaabgiw.parquet --strict
```

### Python tests (when test files exist)
```bash
# Full suite
pytest

# Single test file or single test
pytest tests/test_example.py
pytest tests/test_example.py::test_case_name
```

### Lint/build
- No dedicated lint command or build pipeline is configured in this repository.


## High-level architecture

`run_per_field.py` is the orchestrator. For each selected OID, it executes **30 simulations** (Ia/II/Ibc × 10 deterministic partition pivots), then writes one parquet per OID plus run metadata.

Per simulation flow:
1. Template selection from `data/{Ia,II,Ibc}/*.dat` (deterministic cycling).
2. Physical sampling (`core/correction.py`): cosmological redshift + host extinction.
3. Spectral correction (`correct_redeening`) and synthetic photometry (`Syntetic_photometry_v2`) for g/r/i.
4. LOESS smoothing + noise injection + luminosity normalization (`M_peak` distribution from `config.py`).
5. Multi-band projection (`core/multiband_projection.py`) over real ZTF cadence with a **shared temporal offset across filters**.
6. Detection labeling and serialization.

Important modules:
- `config.py`: central source for paths, processing params, redshift limits, luminosity/extinction settings.
- `core/multiband_projection.py`: deterministic partition mode and shared-offset projection logic.
- `verify_output.py`: output QA checks (structure, completeness, ranges, consistency, plausibility).
- `tools/precompute_sfd98.py` + `tools/dust_maps.py`: Milky Way extinction cache/live queries.


## Key conventions

- **Do not hardcode science/config parameters in scripts.** Read and update values in `config.py`.
- **`legacy/` is obsolete.** Prefer current entrypoints (`run_per_field.py`, `verify_output.py`, `tools/*`).
- **Projection offset is shared across filters** in multi-band mode; do not split offsets per band.
- **Each OID output should represent 30 simulations** (3 SN types × 10 `part_index` partitions), unless failed sims are explicitly expected.
- **Output schema matters:** downstream tooling expects columns such as  
  `oid, mjd, maglimit, filter, magnitud_modelo, magnitud_proyectada, upperlimit, detected, sn_type, template, z, ebmv_host, ebmv_mw, part_index, n_divisions, desplazamiento`.
- **Detection encoding is mixed by design:** `detected` is boolean, `upperlimit` uses `'T'/'F'`.
- **Template reads are cached:** `leer_spec()` uses `lru_cache`; avoid mutating cached DataFrames in-place.
- **Code language is bilingual (Spanish/English).** Keep naming and log style consistent with surrounding code.
- **MW extinction source order:** prefer precomputed `data/sfd98_cache.parquet`, then live query, then fallback value.
- **Notion bitácora order:** entries are newest-first (most recent at the top, oldest at the bottom). When appending a new entry, insert it above the current most-recent entry — never at the bottom.
