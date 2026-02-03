# Validation pipeline

Validates a study implementation (metadata, specification, materials, evaluator) against the original paper using LLM-based checks.

**Entry:** `run_validation.py` — e.g. `python legacy/validation_pipeline/run_validation.py study_001`. Optional after Stage 3/4 when using `scripts/run_baseline_pipeline.py --from-pdf --with-validation`.

**Outputs:** JSON and markdown summaries under `legacy/validation_pipeline/outputs/`.

**Requires:** Google Gemini SDK (`pip install google-genai`) and `GOOGLE_API_KEY` in `.env`.
