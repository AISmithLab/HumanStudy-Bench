# Generation pipeline

PDF → study pipeline (stages 1–6 + final).

**Stages:** (1) Replicability filter, (2) Study/data extraction, (3) Generate study files under `data/studies/{study_id}/`, (4) Generate `src/studies/{study_id}_config.py`, (5) Simulation (LLM agents), (6) Evaluation (ECS_corr only), (final) Finding explanations.

**Entry:** `run.py` — e.g. `python generation_pipeline/run.py --stage 1` or `--stage final --study-id study_001`. Usually invoked via `scripts/run_baseline_pipeline.py` (which drives 1–6 and final).

**Outputs:** Stage 1/2 JSON and MD under `generation_pipeline/outputs/`. Study files under `data/studies/{study_id}/`. Configs under `src/studies/`.
