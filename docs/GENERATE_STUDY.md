# Generate New Study from PDF

Turn a psychology paper PDF into a benchmarkable study (Stage 1-4).

## Quick Start

```bash
# 1. Put PDF in data/studies/study_XXX/
mkdir -p data/studies/study_013
cp my_paper.pdf data/studies/study_013/

# 2. Generate study files (uses Gemini by default)
export GOOGLE_API_KEY=AIzaSy...
python scripts/run_baseline_pipeline.py --study-id study_013 --from-pdf --until 4

# 3. Review outputs
ls generation_pipeline/outputs/study_013/

# 4. Run benchmark
python scripts/run_baseline_pipeline.py --study-id study_013 --real-llm --model mistralai/mistral-nemo
```

## Stage 1-4 Pipeline

The generation pipeline extracts study information from PDF:

1. **Stage 1** - Filter for replicability (checks if study is suitable)
2. **Stage 2** - Extract study data (hypothesis, conditions, measures)
3. **Stage 3** - Generate Python config (creates `src/studies/study_XXX_config.py`)
4. **Stage 4** - Generate JSON materials (trials, prompts, ground truth)

## Use Different LLM Provider

### Option 1: Use Claude instead of Gemini

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...
python scripts/run_baseline_pipeline.py \
  --study-id study_013 \
  --from-pdf \
  --until 4 \
  --generation-provider anthropic \
  --generation-model claude-3-5-sonnet-20241022
```

### Option 2: Use GPT-4

```bash
export OPENAI_API_KEY=sk-proj-...
python scripts/run_baseline_pipeline.py \
  --study-id study_013 \
  --from-pdf \
  --until 4 \
  --generation-provider openai \
  --generation-model gpt-4o
```

### Option 3: Use Grok

```bash
export XAI_API_KEY=xai-...
python scripts/run_baseline_pipeline.py \
  --study-id study_013 \
  --from-pdf \
  --until 4 \
  --generation-provider xai \
  --generation-model grok-2
```

## Output Files

After Stage 1-4, check `generation_pipeline/outputs/study_013/`:

```
study_013/
├── stage1_filter.json          # Replicability assessment
├── stage2_extraction.json      # Extracted study data
├── stage3_config.py            # Python config (copy to src/studies/)
└── stage4_materials/
    ├── metadata.json           # Study metadata
    ├── trials.json             # Trial definitions
    ├── ground_truth.json       # Expected results
    └── participants.json       # Participant profiles
```

## Manual Review (Recommended)

Before running benchmark, review and edit generated files:

```bash
# 1. Check extraction quality
cat generation_pipeline/outputs/study_013/stage2_extraction.json

# 2. Review config
cat generation_pipeline/outputs/study_013/stage3_config.py

# 3. Check trials
cat generation_pipeline/outputs/study_013/stage4_materials/trials.json

# 4. Manually copy/edit if needed
cp generation_pipeline/outputs/study_013/stage3_config.py src/studies/study_013_config.py
cp -r generation_pipeline/outputs/study_013/stage4_materials/* data/studies/study_013/
```

## Validation Pipeline (Optional)

Run validation after Stage 3 or 4 to check consistency:

```bash
python scripts/run_baseline_pipeline.py \
  --study-id study_013 \
  --from-pdf \
  --until 4 \
  --with-validation
```

This runs `legacy/validation_pipeline/` to verify generated files.

## Troubleshooting

**Stage 1 fails (PDF not replicable):**
- Check if study has clear conditions and measurable outcomes
- Some qualitative studies may not be suitable

**Stage 2 extraction incomplete:**
- Use `--generation-model gpt-4o` for better extraction
- Manually edit `stage2_extraction.json` and re-run Stage 3-4

**Stage 3 config errors:**
- Review Python syntax in generated config
- Compare with existing studies in `src/studies/`

**Stage 4 trials missing details:**
- Manually edit `trials.json` to add missing info
- Re-run Stage 4 with regeneration instructions

## Full Pipeline Example

```bash
# Complete flow: PDF → benchmark → results
export GOOGLE_API_KEY=AIzaSy...
export OPENROUTER_API_KEY=sk-or-v1-...

# Generate study (Stage 1-4)
python scripts/run_baseline_pipeline.py --study-id study_013 --from-pdf --until 4

# Review outputs
cat generation_pipeline/outputs/study_013/stage2_extraction.json

# Run benchmark (Stage 5-6)
python scripts/run_baseline_pipeline.py \
  --study-id study_013 \
  --real-llm \
  --model mistralai/mistral-nemo \
  --presets v1_empty v2_human v3_human_plus_demo

# Check results
cat results/benchmark/study_013/mistralai_mistral-nemo_v2-human/evaluation_results.json
```

## CLI Flags

```bash
--study-id study_XXX           # Study ID
--from-pdf                     # Enable PDF generation pipeline
--until 4                      # Stop after Stage 4 (before benchmark)
--generation-provider PROVIDER # openai | anthropic | xai | openrouter | gemini
--generation-model MODEL       # Model for PDF processing
--with-validation              # Run validation after Stage 3/4
```
