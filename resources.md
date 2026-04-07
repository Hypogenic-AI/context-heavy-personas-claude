# Resources Catalog

## Summary
This document catalogs all resources gathered for the "Context Heavy Personas" research project, investigating whether certain LLM personas can only be activated after conditioning on very large contexts (10K+ tokens).

---

## Papers
Total papers downloaded: 15

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Persona Vectors | Chen, Arditi et al. (Anthropic) | 2025 | papers/2507.21509_persona_vectors.pdf | Linear persona directions in activation space; monitoring & steering |
| 2 | Geometry of Persona (Soul Engine) | Wang | 2025 | papers/2512.07092_geometry_of_persona.pdf | OCEAN traits as orthogonal linear subspaces |
| 3 | PICLe | Choi, Li (ICML) | 2024 | papers/2405.02501_PICLe_persona_icl.pdf | Bayesian framework for persona ICL; 99-persona benchmark |
| 4 | Long-Context ICL | Bertsch et al. (CMU) | 2024 | papers/2405.00200_icl_long_context.pdf | ICL scales to 2000+ demos; key evidence for context-dependent capabilities |
| 5 | PersonaHub (1B Personas) | Tencent AI Lab | 2024 | papers/2406.20094_persona_hub_billion.pdf | 1B diverse personas from web data |
| 6 | Lost in the Middle | Liu et al. | 2024 | papers/2307.03172_lost_in_middle.pdf | Positional bias in long contexts |
| 7 | RoleLLM | Wang et al. | 2024 | papers/2310.00746_RoleLLM.pdf | Role-playing benchmark, 168K samples |
| 8 | In-Context Vectors | Liu et al. | 2023 | papers/2311.06668_in_context_vectors.pdf | ICL as latent space steering |
| 9 | Emergent Abilities | Wei et al. (Google) | 2022 | papers/2206.07682_emergent_abilities.pdf | Seminal paper on emergent abilities |
| 10 | Emergent Abilities Survey | 2025 survey | 2025 | papers/2503.05788_emergent_abilities_survey.pdf | Comprehensive survey |
| 11 | Impostor Among Us | 2025 | 2025 | papers/2501.04543_impostor_among_us.pdf | LLM persona complexity evaluation |
| 12 | Mixture-of-Personas | 2025 | 2025 | papers/2504.05019_mixture_of_personas.pdf | Probabilistic persona mixture models |
| 13 | Context Length Hurts | 2025 | 2025 | papers/2510.05381_context_length_hurts.pdf | Context length degradation effects |
| 14 | Consistent Personas via RL | 2025 | 2025 | papers/2511.00222_consistent_personas_rl.pdf | Persona drift mitigation via RL |
| 15 | Fundamental Limits of LLMs | 2025 | 2025 | papers/2511.12869_fundamental_limits_llm.pdf | Theoretical capacity limits |

See [papers/README.md](papers/README.md) for detailed descriptions.

---

## Datasets
Total datasets downloaded: 2 (+ 1 documented with download instructions)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Anthropic Model-Written Evals | HuggingFace: Anthropic/model-written-evals | 99 personas, 99K statements | Persona elicitation (agree/disagree) | datasets/anthropic_model_written_evals/ | Primary evaluation dataset |
| PersonaHub | HuggingFace: proj-persona/PersonaHub | 10K samples (subset) | Persona description & synthesis | datasets/personahub/ | 4 configs sampled |
| LMSYS-Chat-1M | HuggingFace: lmsys/lmsys-chat-1m | 1M conversations | Real-world chat analysis | datasets/lmsys_chat_1m/ (gated) | Requires HF auth; instructions provided |

See [datasets/README.md](datasets/README.md) for download instructions and detailed descriptions.

---

## Code Repositories
Total repositories cloned: 4

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| persona_vectors | github.com/safety-research/persona_vectors | Persona vector extraction & steering pipeline | code/persona_vectors/ | Anthropic; requires PyTorch, transformers |
| picle | github.com/deeplearning-wisc/picle | Persona In-Context Learning framework | code/picle/ | ICML 2024; Llama-2, Vicuna, GPT-J |
| long-context-icl | github.com/abertsch72/long-context-icl | Long-context ICL experiments | code/long-context-icl/ | CMU; includes pre-computed results |
| persona-hub | github.com/tencent-ailab/persona-hub | PersonaHub persona synthesis scripts | code/persona-hub/ | Tencent AI Lab; requires OpenAI API or vLLM |

See [code/README.md](code/README.md) for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
1. **Paper-finder service** (attempted first, service unavailable)
2. **Web search** across arXiv, Semantic Scholar, Papers with Code using multiple query formulations:
   - "context-dependent personas in large language models"
   - "LLM steering vectors persona representation geometry"
   - "many-shot in-context learning persona behavior"
   - "number of distinct personas LLM capacity limits"
3. **Citation following** from key papers (PICLe references, Persona Vectors references)

### Selection Criteria
- **Direct relevance**: Papers studying persona representation, elicitation, or control in LLMs
- **Methodological relevance**: Papers on long-context ICL, activation steering, emergent abilities
- **Theoretical relevance**: Papers on representational capacity limits, fundamental limits of LLMs
- **Recency**: Focus on 2023-2025 papers for state-of-the-art

### Challenges Encountered
- No existing paper directly studies the "context-heavy personas" hypothesis -- this is genuinely novel
- LMSYS-Chat-1M requires gated access (HuggingFace authentication)
- Paper-finder service was unavailable, required manual search via web

### Gaps and Workarounds
- No direct dataset for "context-heavy persona elicitation" exists; will need to construct experimental protocol using Anthropic personas + long-context demonstrations
- No code for many-shot persona elicitation exists; will need to adapt PICLe and long-context-icl code

---

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **Anthropic Model-Written Evals**: 99 personas with binary evaluation protocol, enables direct comparison with PICLe baselines. Use this to systematically test persona elicitation accuracy as a function of number of in-context examples K = {0, 1, 3, 10, 50, 100, 500}.
- **PersonaHub**: Source of diverse persona descriptions for generating long-context persona demonstrations.

### 2. Baseline Methods
- Base model (zero-shot): ~65% accuracy
- System prompt only: ~54-75% depending on method
- PICLe K=3: 88.1% (published baseline on Llama-2)
- Random ICL K=3: ~80%
- These baselines establish what's achievable with short context

### 3. Evaluation Metrics
- **Action Consistency** (from PICLe): Primary metric, enables direct comparison
- **Persona Vector Projection** (from Persona Vectors): Measures internal persona activation as continuous function of context length
- **Persona Elicitation Accuracy vs K curve**: The key experimental deliverable

### 4. Code to Adapt/Reuse
- **persona_vectors/**: Use the persona vector extraction pipeline to measure how persona projections change with context length
- **picle/**: Adapt the PICLe framework to test with varying K values well beyond K=3
- **long-context-icl/**: Adapt the many-shot ICL infrastructure for persona elicitation tasks

### 5. Experimental Protocol Sketch
1. For each of the 99 Anthropic personas:
   a. Construct demonstration sets of varying sizes K = {0, 1, 3, 10, 50, 100, 500}
   b. Measure persona elicitation accuracy at each K
   c. Identify "easy" personas (high accuracy at low K) vs "hard" personas (require high K)
2. For models with activation access (Qwen 2.5, Llama 3.1):
   a. Extract persona vectors for each persona
   b. Measure projection onto persona vector as function of K
   c. Look for personas where projection only reaches threshold at high K
3. Analysis:
   a. Do "context-heavy" personas exist (accuracy only rises with K >> 3)?
   b. How does the persona difficulty distribution relate to persona vector geometry?
   c. Does the number of "accessible" personas increase with model size or context length?
