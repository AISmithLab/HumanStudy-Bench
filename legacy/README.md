# Legacy

Optional, non-essential tools kept for reference.

- **gui.py** — Streamlit app for Generation Pipeline and Validation Pipeline (stage-by-stage control, file editing, validation runs). Run from repo root: `streamlit run legacy/gui.py`. Requires streamlit and the same env as the main pipeline (e.g. Gemini for validation). Not required for the CLI workflow.

- **validation_pipeline/** — Validates study implementations (metadata, specification, materials, evaluator) against the original paper using LLM-based checks. Entry: `python legacy/validation_pipeline/run_validation.py study_001`. Outputs under `legacy/validation_pipeline/outputs/`. See `legacy/validation_pipeline/README.md` for details.
