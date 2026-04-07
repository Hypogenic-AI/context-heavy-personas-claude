# Context-Heavy Personas -- Cloned Repositories

This directory contains cloned repositories relevant to research on context-heavy personas
in large language models. None of the dependencies have been installed; this document
serves as a reference index.

---

## 1. Persona Vectors

- **URL:** https://github.com/safety-research/persona_vectors
- **Location:** `code/persona_vectors/`
- **Purpose:** Monitoring and controlling character traits (personas) in language models
  via "persona vectors" -- activation-space directions extracted by contrasting positive
  and negative persona prompts. Supports inference-time steering and training-time
  preventative steering. From Anthropic's safety research group.
- **Key files/scripts:**
  - `generate_vec.py` -- Compute persona vectors from positive/negative activation diffs
  - `training.py` -- Fine-tune models (LoRA) with optional steering during training
  - `activation_steer.py` -- Inference-time activation steering logic
  - `eval/eval_persona.py` -- Evaluate personas with OpenAI-judge scoring
  - `eval/cal_projection.py` -- Calculate projections onto persona vectors
  - `data_generation/prompts.py` -- Prompts used to generate trait artifacts
  - `configs/train_instruct_7b.json`, `configs/train_instruct_7b_steer.json` -- Training configs
  - `scripts/generate_vec.sh`, `scripts/eval_steering.sh`, `scripts/eval_persona.sh`, `scripts/cal_projection.sh` -- Pipeline shell scripts
  - `dataset.zip` -- Training datasets (normal + misaligned examples per trait)
- **Dependencies:** Python 3, PyTorch 2.6, transformers 4.52, peft 0.15, trl 0.15,
  vllm 0.8.5, openai, accelerate, bitsandbytes, unsloth, datasets, pandas, pydantic
- **Relevance to context-heavy personas research:** Directly addresses how persona
  information is encoded in model activations. The extraction method (mean-diff of
  activations under contrasting system prompts) shows how rich persona context
  collapses into steering vectors. Useful for understanding whether long persona
  descriptions can be distilled into compact representations, and for controlling
  undesired persona emergence.

---

## 2. PICLe (Persona In-Context Learning)

- **URL:** https://github.com/deeplearning-wisc/picle
- **Location:** `code/picle/`
- **Purpose:** Eliciting diverse behaviors from LLMs using Persona In-Context Learning.
  PICLe selects in-context examples that maximize the likelihood of a target persona,
  enabling LLMs to adopt specific personality traits without fine-tuning. Published at
  ICML 2024.
- **Key files/scripts:**
  - `src/main.py` -- Entry point; supports modes: `persona_sft`, `picle`, baselines
  - `src/icl_strategies/` -- ICL example selection strategies (PICLe, similarity, etc.)
  - `src/models/` -- Model wrappers for Llama-2, Vicuna, GPT-J
  - `src/data/` -- Data loading and persona dataset handling
  - `src/personas.txt` -- List of persona types
  - `scripts/llama2/picle.sh`, `scripts/llama2/persona_sft.sh` -- Experiment scripts
  - `environment.yml` -- Conda environment specification
- **Dependencies:** Python 3.8, PyTorch 2.1, transformers 4.36, peft 0.4, trl 0.7,
  accelerate, bitsandbytes, datasets, scikit-learn, wandb, xformers, sentencepiece
- **Relevance to context-heavy personas research:** Core method for using in-context
  learning to steer LLM persona behavior. Demonstrates that carefully selected ICL
  examples (i.e., context) can elicit target personality traits -- directly relevant to
  understanding how context length and content affect persona adoption. The likelihood-
  based selection is a principled approach to "context-heavy" persona prompting.

---

## 3. Long-Context ICL

- **URL:** https://github.com/abertsch72/long-context-icl
- **Location:** `code/long-context-icl/`
- **Purpose:** In-depth exploration of in-context learning with long-context models (up
  to 80k tokens). Studies how ICL performance scales with the number of demonstration
  examples when models have large context windows. Built on the Parallel Context Windows
  (PCW) codebase.
- **Key files/scripts:**
  - `run_evaluation.py` -- Main evaluation script; runs ICL experiments with configurable
    number of shots (1 to 2000+)
  - `model_loaders.py` -- Load long-context models (Llama-80k, etc.)
  - `datasets_loader.py` -- Dataset loading (Banking77, etc.)
  - `pcw_wrapper.py` -- Parallel Context Windows wrapper
  - `replacement_modeling_llama.py` -- Modified Llama modeling for PCW
  - `constants.py` -- Dataset and model constants
  - `experiment_manager.py` -- Experiment orchestration
  - `dataset-splits/` -- Pre-computed test set IDs for reproducibility
  - `final-results/` -- Pre-computed outputs for all experiments in the paper
  - `env.yml` -- Conda environment (Python 3.10, PyTorch 2.0, CUDA 11.8)
- **Dependencies:** Python 3.10, PyTorch 2.0, transformers 4.28, datasets 2.9,
  accelerate, faiss-gpu, nltk, sacremoses, sentencepiece, wandb, matplotlib, pandas
- **Relevance to context-heavy personas research:** Provides empirical evidence on how
  model behavior changes as context grows from few-shot to many-shot (thousands of
  examples). Directly relevant to understanding the scaling dynamics of context-heavy
  persona prompts: how does adding more persona-defining context (examples, backstory,
  behavioral demonstrations) affect consistency and quality? The PCW technique may also
  be applicable for structuring very long persona contexts.

---

## 4. PersonaHub

- **URL:** https://github.com/tencent-ailab/persona-hub
- **Location:** `code/persona-hub/`
- **Purpose:** Scaling synthetic data creation using 1 billion diverse personas
  automatically curated from web data. Each persona acts as a "distributed carrier of
  world knowledge" to drive diverse synthetic data generation (math problems, reasoning,
  instructions, game NPCs, tools). From Tencent AI Lab.
- **Key files/scripts:**
  - `code/openai_synthesize.py` -- Synthesize data using OpenAI API (GPT-4o)
  - `code/vllm_synthesize.py` -- Synthesize data using open-source models via vLLM
  - `code/prompt_templates.py` -- Prompt templates for persona-driven synthesis
  - `demo_openai_synthesize.sh` -- Demo script (OpenAI)
  - `demo_vllm_synthesize.sh` -- Demo script (vLLM / open-source)
  - `data/` -- Sample synthetic data (math, reasoning, instructions, NPCs, tools)
  - `requirements.txt` -- Minimal dependency list
- **Dependencies:** datasets, transformers, openai, tqdm (plus vllm for local inference)
- **Relevance to context-heavy personas research:** The largest-scale persona dataset
  available (1B personas, 370M "elite" personas). Demonstrates that personas can be
  used as a scalable axis for generating diverse data. Directly relevant as a source of
  persona descriptions for context-heavy experiments: each persona is a compact text
  description that can be expanded into rich context. Also shows the paradigm of
  persona-as-prompt for controlling LLM output diversity.

---

## Cross-Cutting Themes

These four repositories collectively address the research question of how persona context
influences LLM behavior:

| Aspect | Relevant Repos |
|--------|---------------|
| Persona representation (vectors vs. text) | persona_vectors, picle |
| In-context learning for persona control | picle, long-context-icl |
| Scaling context length with persona examples | long-context-icl, persona-hub |
| Large-scale persona datasets | persona-hub |
| Activation-level persona monitoring | persona_vectors |
| Training-time vs. inference-time persona control | persona_vectors, picle |
| Synthetic data generation from personas | persona-hub |
