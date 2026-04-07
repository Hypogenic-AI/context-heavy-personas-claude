# Datasets for Context Heavy Personas Research

## Overview

These datasets support research into whether certain LLM personas only emerge after conditioning on very large contexts (10K+ tokens).

## 1. Anthropic Model-Written Evals (Persona Dataset)

- **Source:** [Anthropic/model-written-evals](https://huggingface.co/datasets/Anthropic/model-written-evals)
- **Paper:** Perez et al. (2022), "Discovering Language Model Behaviors with Model-Written Evaluations"
- **Local path:** `anthropic_model_written_evals/`
- **Size:** ~21 MB
- **Contents:**
  - `persona/` - 99 persona JSONL files, each with 1000 yes/no statements
  - `advanced-ai-risk/` - Human and LM-generated evals for AI safety behaviors
  - Each statement has `question`, `answer_matching_behavior`, `answer_not_matching_behavior`
- **Relevance:** Core dataset for persona elicitation experiments. Each persona file can be used to construct in-context examples for PICLe-style persona steering.

### Download instructions

```python
from huggingface_hub import snapshot_download
snapshot_download("Anthropic/model-written-evals", repo_type="dataset", local_dir="datasets/anthropic_model_written_evals")
```

## 2. PersonaHub (Tencent AI Lab)

- **Source:** [proj-persona/PersonaHub](https://huggingface.co/datasets/proj-persona/PersonaHub)
- **Local path:** `personahub/`
- **Size:** ~12 MB (10K sample per config; full dataset is much larger)
- **Configs downloaded:**
  - `persona_sample_10k.jsonl` - 10K persona descriptions (from ~200K+ total)
  - `elite_persona_sample_10k.jsonl` - 10K elite personas with domain classifications
  - `instruction_sample_10k.jsonl` - 10K persona-conditioned instructions
  - `train_sample_10k.jsonl` - 10K math problems (persona-conditioned)
- **Available configs:** math, instruction, reasoning, knowledge, npc, tool, persona, elite_persona
- **Relevance:** Diverse persona descriptions for generating in-context conditioning text. The `persona` and `elite_persona` configs are most relevant.

### Download instructions

```python
from datasets import load_dataset

# Stream to avoid downloading full dataset (~1B+ entries in some configs)
ds = load_dataset("proj-persona/PersonaHub", "persona", streaming=True)
examples = list(itertools.islice(ds['train'], 10000))
```

## 3. LMSYS-Chat-1M

- **Source:** [lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
- **Paper:** Zheng et al. (2023), used in the Persona Vectors paper
- **Local path:** `lmsys_chat_1m/`
- **Size:** Not yet downloaded (gated dataset)
- **Status:** REQUIRES AUTHENTICATION - this is a gated dataset
- **Contents:** ~1M real conversations from Chatbot Arena with metadata (model, language, moderation scores)
- **Relevance:** Real-world chat data for analyzing naturally occurring persona patterns in user-LLM interactions.

### Download instructions

```bash
# 1. Accept terms at https://huggingface.co/datasets/lmsys/lmsys-chat-1m
# 2. Login
huggingface-cli login
```

```python
from datasets import load_dataset
import itertools

# Stream a sample (full dataset is very large)
ds = load_dataset("lmsys/lmsys-chat-1m", streaming=True)
examples = list(itertools.islice(ds['train'], 5000))
```

## Notes

- Large files are excluded from git via `.gitignore`
- To regenerate all datasets, activate the venv and run the download scripts above
- All datasets are used under their respective licenses (see HuggingFace pages for details)
