# Research Plan: Context Heavy Personas

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are increasingly used to simulate diverse perspectives, characters, and expert personas. Understanding whether some personas require extensive context conditioning (10K+ tokens) to emerge has implications for: (1) the practical design of persona-based systems, (2) our theoretical understanding of LLM representational capacity, and (3) the fundamental question of how many distinct "modes of being" a model can express. If context-heavy personas exist, it means short prompts access only a subset of the model's full persona space.

### Gap in Existing Work
PICLe (2024) showed dramatic persona elicitation improvement with just K=3 ICL examples but never tested beyond K=3. Long-context ICL (Bertsch et al., 2024) scaled to 2000+ demos but only for classification tasks, not personas. Persona Vectors (2025) showed many-shot prompting progressively shifts persona projections but tested only 0-20 examples. **No study has systematically measured persona elicitation accuracy across a wide range of context lengths (K=0 to K=500+) to identify whether "hard" personas exist that require extensive conditioning.**

### Our Novel Contribution
We conduct the first systematic study of persona elicitation accuracy as a function of in-context example count K, spanning K=0 to K=500, across the full Anthropic 99-persona benchmark. We test whether:
1. Some personas are "context-heavy" (only achievable at high K)
2. The persona-accuracy curve shows phase transitions or continuous improvement
3. If no context-heavy personas exist, what this implies about the total persona space

### Experiment Justification
- **Experiment 1 (Persona Scaling Curves)**: Measures accuracy vs K for each persona. This is the core test of the hypothesis — do accuracy curves plateau early or keep rising?
- **Experiment 2 (Persona Difficulty Taxonomy)**: Clusters personas by their K-sensitivity profile to identify "easy" vs "hard" personas.
- **Experiment 3 (Diminishing Returns Analysis)**: Tests whether marginal accuracy gains decrease with K, and whether any personas show late-onset improvements.

## Research Question
Do certain LLM personas require conditioning on large contexts (many in-context examples) to be reliably elicited? If not, what does this imply about the total number of representable personas?

## Background and Motivation
The PICLe framework models LLMs as mixtures of latent persona distributions. ICL examples shift the model toward a target persona. The key question: does this shifting process saturate quickly (all personas reachable with few examples), or do some personas require extensive context to "find"?

## Hypothesis Decomposition
- **H1**: Persona elicitation accuracy increases monotonically with K for most personas
- **H2**: Some personas show significant accuracy gains between K=10 and K=100+ ("context-heavy" personas)
- **H3**: "Easy" personas (high accuracy at K≤3) are more aligned with the model's default behavior
- **H4**: If all personas saturate by K≤10, the effective persona space is "shallow" — all accessible personas can be specified with short prompts

## Proposed Methodology

### Approach
Use real LLM API (GPT-4.1 via OpenAI) to test persona elicitation on the Anthropic Model-Written Evals benchmark. For each persona, systematically vary the number of in-context demonstrations K and measure accuracy.

### Experimental Steps
1. **Data Preparation**: For each of the 99 personas, split 1000 examples into demo pool (800) and test set (200)
2. **Baseline (K=0)**: Query model with no persona context, measure accuracy
3. **Scaling Experiment**: For K ∈ {1, 3, 5, 10, 25, 50, 100, 200, 500}, construct prompts with K demonstrations, measure accuracy on test set
4. **Repeat**: Run 3 seeds per condition for reliability (vary demo selection)
5. **Analysis**: Fit scaling curves, identify phase transitions, cluster personas by difficulty

### Cost Estimation
- 99 personas × 10 K-values × 200 test examples × 3 seeds = 594,000 API calls
- This is too expensive. Optimization:
  - Select 20 diverse personas (spanning categories)
  - Use 50 test examples per persona
  - K ∈ {0, 1, 3, 10, 25, 50, 100, 200}
  - 2 seeds
  - Total: 20 × 8 × 50 × 2 = 16,000 calls
  - At ~200 tokens/call average: ~3.2M tokens ≈ $4-8 with GPT-4.1-mini or ~$40 with GPT-4.1
  - Use GPT-4.1-mini for main experiment, GPT-4.1 for validation subset

### Baselines
- K=0 (zero-shot, no persona conditioning)
- K=1 (single example)
- K=3 (PICLe baseline equivalent)
- Published PICLe K=3 accuracy: 88.1% (on Llama-2, not directly comparable but useful reference)

### Evaluation Metrics
- **Action Consistency**: Fraction of test examples where model response matches persona ground truth
- **Persona Difficulty Score**: K value at which accuracy first exceeds 80%
- **Marginal Gain**: Accuracy improvement from K to 2K (diminishing returns analysis)

### Statistical Analysis Plan
- Per-persona accuracy with 95% Wilson confidence intervals
- Spearman correlation between persona difficulty and K-sensitivity
- Permutation test for significance of accuracy differences between K values
- Clustering (k-means or hierarchical) of personas by accuracy profile

## Expected Outcomes
- **If H2 is supported**: Some personas show clear accuracy jumps at high K → "context-heavy personas" exist → the persona space has depth beyond what short prompts can access
- **If H2 is refuted**: All personas saturate by K≤10 → the effective persona space is "shallow" → the total number of distinct accessible personas may be smaller than expected, as all can be specified concisely

## Timeline and Milestones
1. Setup & data prep: 15 min
2. Implementation: 45 min
3. Run experiments: 60-90 min (API calls)
4. Analysis & visualization: 30 min
5. Documentation: 30 min

## Potential Challenges
- API rate limits → use exponential backoff and parallel requests
- Cost management → start with subset, scale if promising
- Prompt length limits → at K=500 with ~50 tokens/example, prompt is ~25K tokens, well within GPT-4.1 128K context
- "Lost in the middle" effect → randomize demo order

## Success Criteria
- Clear accuracy-vs-K curves for 20+ personas
- Statistical test of whether any personas show significant late-onset improvement (K>10)
- Taxonomy of personas by context-sensitivity
- Clear answer to: do context-heavy personas exist?
