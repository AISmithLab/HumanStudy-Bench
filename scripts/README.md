# Scripts

CLI and table-generation tools for the benchmark.

**Main entry:** `run_baseline_pipeline.py` — single end-to-end port (simulation → evaluation → explain → summary/production). Run from repo root: `python scripts/run_baseline_pipeline.py --help`.

**Simple results (user-facing):**
- `simple_results.py` — PAS_raw, ECS, tokens, cost at benchmark, per-study, and per-finding levels. Output: `simple_summary.md`, `simple_studies.csv`, `simple_findings.csv`.

**Other scripts:**
- `generate_results_table.py` — aggregate results to `benchmark_summary.json` / `.md` (ECS_corr only).
- `migrate_to_benchmark.py` — set a run as baseline under `results/benchmark/`.
- `generate_study_backgrounds.py` — generate study background text.
- `run_studies_parallel.py` — run studies in parallel.
- `compute_random_alignment.py` — random alignment baseline.

**Advanced** (visualization, production tables):
- `advanced/generate_production_results.py` — LaTeX production tables (ECS vs cost).
- `advanced/generate_detailed_metrics_table.py` — detailed hierarchical metrics.

**Legacy PAS+ECS** (moved to `legacy/scripts/`):
- `legacy/scripts/legacy_generate_results_table_pas_ecs.py` — legacy results table (PAS+ECS).
- `legacy/scripts/legacy_generate_production_results_pas_ecs.py` — legacy production LaTeX tables (PAS+ECS).

Outputs: default `results/benchmark/`; with `--run-name X`, `results/runs/X/`.
