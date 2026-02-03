# Result Storage Strategy (Simplified)

## Overview

Results are stored in a simple, hierarchical structure that automatically organizes by study and configuration.

## Storage Structure

```
results/
├── benchmark/                    # Baseline reference (configurable via .env BENCHMARK_FOLDER)
│   ├── study_001/
│   │   └── {model}_{prompt}/
│   │       ├── full_benchmark.json
│   │       ├── detailed_stats.csv
│   │       └── evaluation_results.json
│   └── benchmark_metadata.json
└── runs/                         # All benchmark runs
    └── {run_name}/
        ├── study_001/
        │   └── {model}_{prompt}/
        │       ├── full_benchmark.json
        │       ├── detailed_stats.csv
        │       └── evaluation_results.json
        └── study_002/
            └── ...
```

## Key Principles

1. **Simple**: All runs in `results/runs/{run_name}/` (default: "default")
2. **Overwrite**: Same name = same run (directly replace)
3. **Auto-versioning**: Different configs automatically stored in subfolders
4. **Self-contained**: Each config folder has all its data
5. **Baseline**: Good results can be copied to `results/benchmark/` as reference

## Configuration

### Benchmark Folder Name

Set in `.env` file (optional):
```bash
BENCHMARK_FOLDER=benchmark
```

Default: `benchmark` (if not set in .env)

## Usage

### Basic Usage

```bash
# Run study_001 (uses "default" run name)
python generation_pipeline/run.py --stage 5 --study-id study_001 --real-llm
python generation_pipeline/run.py --stage 6 --study-id study_001

# Result: results/runs/default/study_001/{model}_{prompt}/
```

### Named Runs (Accumulate Multiple Studies)

```bash
# Use --run-name to accumulate all studies in one run
python generation_pipeline/run.py --stage 5 --study-id study_001 --real-llm --run-name "batch_20260106"
python generation_pipeline/run.py --stage 6 --study-id study_001 --run-name "batch_20260106"

python generation_pipeline/run.py --stage 5 --study-id study_002 --real-llm --run-name "batch_20260106"
python generation_pipeline/run.py --stage 6 --study-id study_002 --run-name "batch_20260106"
```

**Result:** All studies in `results/runs/batch_20260106/`

### Different Configurations

When you run the same study with different model or prompt, it automatically creates a new config folder:

```bash
# Run 1: mistralai + v3
python generation_pipeline/run.py --stage 5 --study-id study_001 --real-llm --run-name "batch" --model "mistralai/mistral-nemo"
# → results/runs/batch/study_001/mistralai_mistral_nemo_v3-human-plus-demo/

# Run 2: gpt-4 + v3
python generation_pipeline/run.py --stage 5 --study-id study_001 --real-llm --run-name "batch" --model "openai/gpt-4"
# → results/runs/batch/study_001/openai_gpt-4_v3-human-plus-demo/

# Run 3: mistralai + v2
python generation_pipeline/run.py --stage 5 --study-id study_001 --real-llm --run-name "batch" --system-prompt-preset "v2_human"
# → results/runs/batch/study_001/mistralai_mistral_nemo_v2-human/
```

**Key Point:** Same `run_name` + different config = different config folders (no overwrite)

### Re-running Same Config

If you re-run with the **same** run_name, model, and prompt, it will **overwrite**:

```bash
# First run
python generation_pipeline/run.py --stage 5 --study-id study_001 --real-llm --run-name "batch" --model "mistralai/mistral-nemo"
# → results/runs/batch/study_001/mistralai_mistral_nemo_v3-human-plus-demo/

# Re-run (overwrites)
python generation_pipeline/run.py --stage 5 --study-id study_001 --real-llm --run-name "batch" --model "mistralai/mistral-nemo"
# → results/runs/batch/study_001/mistralai_mistral_nemo_v3-human-plus-demo/ (overwritten)
```

## Setting Baseline

To set a good result as baseline reference:

```bash
# Migrate study_001 to benchmark folder
python scripts/migrate_to_benchmark.py --run-name batch_20260106 --study-id study_001
```

This copies the study to `results/benchmark/` (or folder specified in `.env` as `BENCHMARK_FOLDER`).

## Config Folder Naming

Format: `{model_slug}_{prompt_slug}`

- **Model slug**: `model.replace("/", "_").replace("-", "_")`
  - `mistralai/mistral-nemo` → `mistralai_mistral_nemo`
  - `openai/gpt-4` → `openai_gpt_4`
  
- **Prompt slug**: `prompt_preset.replace("_", "-")`
  - `v3_human_plus_demo` → `v3-human-plus-demo`
  - `v2_human` → `v2-human`

## File Contents

Each config folder contains:

1. **full_benchmark.json**: Complete raw data for this study+config
   - Individual participant responses
   - All runs (if repeats > 1)
   - Metadata (model, prompt, timestamp)

2. **detailed_stats.csv**: Evaluation results in CSV format
   - One row per test
   - Includes PAS, pi_human, pi_agent, etc.

3. **evaluation_results.json**: Detailed evaluation results
   - Aggregated scores
   - Per-test statistics
   - Normalized scores

## Best Practices

1. **For accumulating studies**: Use `--run-name` consistently
2. **For comparing configs**: Use same `--run-name`, different model/prompt
3. **For preserving good results**: Migrate to `results/benchmark/` using migration script
4. **For parallel runs**: Use `scripts/run_studies_parallel.py` with `--run-name`
