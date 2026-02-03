# Multi-Provider Support Migration Guide

## Overview

HS-Bench now supports multiple LLM providers (OpenAI, Anthropic Claude, xAI Grok, OpenRouter, Google Gemini) for both benchmark testing and study generation pipelines.

## What Changed

### 1. Unified LLM Client (`src/llm/`)

New abstraction layer supporting:
- **OpenAI** (ChatCompletions API)
- **Anthropic** (Messages API with image support)
- **xAI** (OpenAI-compatible endpoint)
- **OpenRouter** (multi-model aggregator)
- **Gemini** (legacy wrapper for PDF workflows)

### 2. Benchmark Testing (Stage 5)

`src/agents/llm_participant_agent.py` now auto-detects provider from model name:
- `mistralai/mistral-nemo` → OpenRouter
- `gpt-4o` → OpenAI
- `claude-3-5-sonnet` → Anthropic
- `grok-2` → xAI

### 3. Generation Pipeline (Stage 1-4)

`generation_pipeline/` now accepts `--provider` flag:
- Default: Gemini (requires `GOOGLE_API_KEY`)
- Optional: `--generation-provider openai/anthropic/xai/openrouter`

PDFs are now extracted to text instead of uploaded (supports all providers).

### 4. Formatter Configuration

Env vars control the response formatter (used in sanity checks):
- `FORMATTER_PROVIDER` (default: `openrouter`)
- `FORMATTER_MODEL` (default: `deepseek/deepseek-v3.2`)

## Usage Examples

### Benchmark with OpenRouter (unchanged, backward compatible)

```bash
export OPENROUTER_API_KEY=sk-or-...
python scripts/run_baseline_pipeline.py --study-id study_001 --real-llm --model mistralai/mistral-nemo
```

### Benchmark with Claude

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/run_baseline_pipeline.py --study-id study_001 --real-llm --model claude-3-5-sonnet-20241022
```

### Benchmark with Grok

```bash
export XAI_API_KEY=xai-...
python scripts/run_baseline_pipeline.py --study-id study_001 --real-llm --model grok-2
```

### Generate study with Claude (instead of Gemini)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/run_baseline_pipeline.py \
  --study-id study_013 \
  --from-pdf \
  --until 4 \
  --generation-provider anthropic \
  --generation-model claude-3-5-sonnet-20241022
```

## Environment Variables

```bash
# Benchmark models (Stage 5)
OPENROUTER_API_KEY=sk-or-...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
XAI_API_KEY=xai-...

# Generation pipeline (Stage 1-4, optional)
GOOGLE_API_KEY=...  # Default provider

# Formatter (optional)
FORMATTER_PROVIDER=openrouter  # or: openai, anthropic, xai
FORMATTER_MODEL=deepseek/deepseek-v3.2
```

## Backward Compatibility

All existing scripts and workflows continue to work without changes:
- OpenRouter models (`mistralai/*`, etc.) auto-detected
- Gemini remains default for `--from-pdf`
- Existing environment variables still respected

## Dependencies

Updated `requirements.txt`:
- `anthropic>=0.3.0` - for Claude API
- `pypdf>=3.0.0` - for PDF text extraction (multi-provider)
- `PyPDF2>=3.0.0` - for legacy DocumentLoader

Install: `pip install -r requirements.txt`
