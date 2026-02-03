# Environment Variables Reference

## Required API Keys

Set at least one API key for benchmark testing (Stage 5):

```bash
# OpenRouter (multi-model aggregator)
export OPENROUTER_API_KEY=sk-or-v1-...

# OpenAI
export OPENAI_API_KEY=sk-proj-...

# Anthropic (Claude)
export ANTHROPIC_API_KEY=sk-ant-api03-...

# xAI (Grok)
export XAI_API_KEY=xai-...
```

Or add to `.env` file:

```bash
echo "OPENROUTER_API_KEY=sk-or-v1-..." >> .env
echo "ANTHROPIC_API_KEY=sk-ant-api03-..." >> .env
```

## Optional Variables

### Generation Pipeline (Stage 1-4)

For `--from-pdf` study generation:

```bash
# Default provider (optional, default: gemini)
GOOGLE_API_KEY=AIzaSy...

# Or use CLI flags: --generation-provider openai --generation-model gpt-4o
```

### Response Formatter

Controls the LLM used for formatting participant responses:

```bash
# Default: OpenRouter + DeepSeek
FORMATTER_PROVIDER=openrouter
FORMATTER_MODEL=deepseek/deepseek-v3.2

# Use Claude for formatting
FORMATTER_PROVIDER=anthropic
FORMATTER_MODEL=claude-3-haiku-20240307

# Use GPT-4o-mini for formatting
FORMATTER_PROVIDER=openai
FORMATTER_MODEL=gpt-4o-mini
```

## Model Auto-detection

You don't need to specify provider - it's auto-detected from model name:

| Model Name | Provider | API Key Required |
|------------|----------|------------------|
| `mistralai/mistral-nemo` | OpenRouter | `OPENROUTER_API_KEY` |
| `deepseek/deepseek-v3.2` | OpenRouter | `OPENROUTER_API_KEY` |
| `qwen/qwen-2.5-72b-instruct` | OpenRouter | `OPENROUTER_API_KEY` |
| `gpt-4o` | OpenAI | `OPENAI_API_KEY` |
| `gpt-4o-mini` | OpenAI | `OPENAI_API_KEY` |
| `claude-3-5-sonnet-20241022` | Anthropic | `ANTHROPIC_API_KEY` |
| `claude-3-haiku-20240307` | Anthropic | `ANTHROPIC_API_KEY` |
| `grok-2` | xAI | `XAI_API_KEY` |

## Full Example

```bash
# .env file
OPENROUTER_API_KEY=sk-or-v1-...
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-api03-...
XAI_API_KEY=xai-...

# Optional: customize formatter (default is deepseek via OpenRouter)
FORMATTER_PROVIDER=openai
FORMATTER_MODEL=gpt-4o-mini

# Optional: for PDF study generation
GOOGLE_API_KEY=AIzaSy...
```

Then run benchmarks without additional config:

```bash
# Uses OPENROUTER_API_KEY (auto-detected)
python scripts/run_baseline_pipeline.py --study-id study_001 --real-llm --model mistralai/mistral-nemo

# Uses ANTHROPIC_API_KEY (auto-detected)
python scripts/run_baseline_pipeline.py --study-id study_001 --real-llm --model claude-3-5-sonnet-20241022

# Uses OPENAI_API_KEY (auto-detected)
python scripts/run_baseline_pipeline.py --study-id study_001 --real-llm --model gpt-4o
```
