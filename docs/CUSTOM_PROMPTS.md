# Custom Prompts Guide

## Built-in Presets

Use with `--system-prompt-preset PRESET` or `--presets PRESET1 PRESET2 ...`:

- **`v1_empty`** - Minimal prompt (no demographics)
- **`v2_human`** - Basic demographics (age, gender, etc.)
- **`v3_human_plus_demo`** - Demographics + example response format
- **`v4_background`** - Full background narrative

```bash
python scripts/run_baseline_pipeline.py \
  --study-id study_001 \
  --real-llm \
  --model gpt-4o \
  --presets v1_empty v2_human v3_human_plus_demo
```

## Create Custom Prompt

### Step 1: Create Python file

Create `src/agents/custom_methods/my_method.py`:

```python
from typing import Dict, Any

def generate_prompt(profile: Dict[str, Any]) -> str:
    """
    Generate custom system prompt for participant agent.
    
    Args:
        profile: Participant profile dict
            - age: int
            - gender: str
            - education: str
            - occupation: str
            - (other demographic fields from study)
    
    Returns:
        System prompt string
    """
    age = profile.get("age", 25)
    gender = profile.get("gender", "unknown")
    
    return f"""You are a {age}-year-old {gender} participant.
Answer concisely and naturally based on your demographics."""
```

**Important:** No imports needed! The function is automatically loaded.

### Step 2: Run with your preset

```bash
python scripts/run_baseline_pipeline.py \
  --study-id study_001 \
  --real-llm \
  --model gpt-4o \
  --system-prompt-preset my_method
```

## Example: Multi-language Support

`src/agents/custom_methods/bilingual.py`:

```python
def generate_prompt(profile: Dict[str, Any]) -> str:
    age = profile.get("age", 25)
    language = profile.get("language", "English")
    
    return f"""You are a {age}-year-old participant.
Primary language: {language}
Respond naturally in {language} when appropriate."""
```

## Example: Personality Traits

`src/agents/custom_methods/personality.py`:

```python
def generate_prompt(profile: Dict[str, Any]) -> str:
    traits = profile.get("personality_traits", {})
    openness = traits.get("openness", 0.5)
    conscientiousness = traits.get("conscientiousness", 0.5)
    
    style = "creative and open-minded" if openness > 0.7 else "practical and traditional"
    detail = "detail-oriented" if conscientiousness > 0.7 else "big-picture focused"
    
    return f"""You are a participant who is {style} and {detail}.
Respond in a way that reflects your personality."""
```

## Profile Fields Available

Common fields in `profile` dict:
- `age` (int)
- `gender` (str)
- `education` (str)
- `occupation` (str)
- `nationality` (str)
- Study-specific fields defined in `data/studies/{study_id}/participants.json`

## Tips

1. **Keep it simple** - Complex prompts don't always improve results
2. **Test variations** - Run with multiple presets to compare
3. **Check profile fields** - See `data/studies/{study_id}/participants.json` for available fields
4. **Avoid over-constraining** - Let the model respond naturally
