# Source (core code)

Core library used by the pipeline and evaluation.

- **agents/** — LLM participant agents, prompt presets (`custom_methods/`), background loading.
- **core/** — `Study`, `Benchmark`, study config loading.
- **evaluation/** — PAS/ECS scoring (`metrics.py`, `stats_lib.py`), evaluator runner, finding explainer, response validation.
- **generators/** — evaluator code generation.
- **studies/** — per-study configs and evaluators (`study_XXX_config.py`, `study_XXX_evaluator.py`).
- **utils/** — I/O helpers.

Run from repo root so `src` is on `PYTHONPATH` (or use `pip install -e .` with `pyproject.toml`).
