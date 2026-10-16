# 8. Evaluation criteria and experimental protocol

An optimum exists only relative to an objective and a set of constraints. The supplied notes emphasize this point: choosing a generator or spectrum depends on the task, dependency lengths, context window, stability, and training.

## Optimization objective and constraints

For a candidate positional mechanism $\eta$, imagine an evaluation distribution $\mathcal D$:

$$
\eta^*=\arg\min_\eta
\mathbb E_{(x,y)\sim\mathcal D}
[-\log p_{\theta_\eta}(y\mid x)],
$$

subject to memory, latency, training-budget, and reliability constraints. Here $\theta_\eta$ is the model trained with that mechanism under the chosen procedure. This is not optimization over phase geometry alone.

Changing the task mixture, training length, optimizer, model size, or adaptation budget can change the answer. Two candidates may be incomparable: one is more accurate locally, another retrieves distant facts with less memory. Such tradeoffs form a Pareto frontier rather than a universal ordering.

## Evaluation criteria and measurement procedures

| Criterion | Evaluation purpose | Measurement |
|---|---|---|
| Local resolution | Syntax and local bindings need nearby distinctions | Fixed-lag copying; short-context loss |
| Long-range use | Definitions and evidence may be far away | Lookup/retrieval by length and target location |
| Shift consistency | An arbitrary origin should not alter the positional factor | Fixed-content common-shift test |
| Numerical stability | Position should not cause unintended norm explosion | Norm, precision, and finite-gradient checks |
| Learned generalization | New offsets may be out of distribution | Explicit train/test length separation |
| Generation correctness | A good training implementation can still have broken caching | Full-prefix versus cached decoding |
| Cost | Useful accuracy has a runtime and memory budget | Parameters, latency, memory on fixed hardware |

Some rows are algebraic correctness properties; others require training and evaluation. Passing the first group does not prove the second.

## Controlled decoder architecture and positional variants

The experiment uses six variants of one two-layer pre-norm causal decoder:

| Method | Injection point | Positional parameters |
|---|---|---|
| none | Causal mask only | None |
| learned | Add a position table to token embeddings | 256 × 64 learned entries |
| sinusoidal | Add fixed sine/cosine features to token embeddings | Base 10,000 |
| rope | Rotate each head's Q and K | Base 10,000; head dimension 16 |
| single_rope | Rotate Q/K with every pair at frequency 1 | One shared rate |
| alibi | Add head-dependent linear penalties to logits | Four fixed slopes |

Common configuration: vocabulary 84, model dimension 64, four heads, two layers, feed-forward width 256, no dropout. Only the learned table adds parameters. The model uses ordinary LayerNorm and GELU, untied input/output weights, and full-vocabulary output logits.

Matched seeds **11, 22, 33** initialize every common parameter identically across methods. A test verifies this. Online training examples and their order also match for each seed. AdamW uses learning rate 0.001, weight decay 0.01, batch size 48, gradient clipping at 1, and **1200 updates** per model. Tasks alternate, giving 600 batches per task. Each model therefore sees 57,600 supervised answers, half for each task.

There is no per-method hyperparameter search or best-checkpoint selection. Evaluation uses the final checkpoint. Consequently, these are comparisons under one finite-budget training procedure, not equal-effort estimates of each method's best attainable performance. In particular, unit-amplitude additive sinusoids interact differently with the 0.02-scale embedding initialization; their weak lookup or extrapolation scores can reflect optimization and scaling as well as position representation.

## Task 1: fixed-lag token copying

Input is a beginning token, uniformly sampled values from 16 possible values, and a final LAG4 query. The answer is the value at **query position minus four**:

~~~text
[BOS, ..., vA, vB, vC, vD, LAG4] → vA
              ↑ four positions before the query
~~~

Training input lengths are sampled from 16, 24, and 32. Evaluation uses 32, 64, and 128. Repeated values are allowed, so content identity alone cannot identify a particular occurrence.

The answer is withheld from the model input. Only its next-token log probability is optimized. The model does not receive a label embedded in a future token.

## Task 2: associative key–value lookup

Input contains unique keys paired with independently sampled values, then a query key:

~~~text
[BOS, kA, vA, kB, vB, ..., ASK, kB] → vB
~~~

Keys come from 64 symbols; values come from 16 symbols. Training input lengths 15, 23, and 31 contain 6, 10, and 14 pairs. Evaluation lengths 31, 63, and 127 contain 14, 30, and 62 pairs. All key identities can appear during training; longer tests add pairs rather than unseen symbol types. The queried pair is sampled uniformly, including distant and recent pairs.

This task needs association, not just locating the repeated query key: the correct value sits immediately after that key's earlier occurrence.

## Held-out answer metrics and seed variability

Each task/length uses **768 held-out examples**. Every method and training seed sees the same held-out stream, from generators separate from training. This enables paired comparisons but means the three seeds do not represent three independent test sets.

The book reports:

- Exact answer accuracy, across the full 84-token vocabulary.
- Mean negative log probability of the correct answer, in nats.
- Conditional answer perplexity, $\exp(\text{mean answer NLL})$.

These are **answer-only conditional** metrics. They are not ordinary natural-text perplexity over every token in a corpus. A predictor uniform over the 16 valid answer values has expected accuracy $1/16=6.25\%$ and perplexity 16. A task oracle has accuracy 100% and perplexity 1. A bag-of-values heuristic can exceed uniform guessing, so crossing 6.25% does not itself demonstrate successful key binding.

Tables show mean ± sample standard deviation across three training seeds. This is variability, not a confidence interval or a claim of statistical significance. The same three trained RoPE checkpoints also receive a separate frozen inference intervention, dividing all positions by four.

## Experimental limitations and confounding factors

The lookup task may remain poorly learned at this budget, so a weak lookup ranking cannot establish an optimal long-range architecture. The causal mask supplies implicit order to the none baseline. The learned table preallocates 256 rows; rows beyond training positions receive no answer-driven gradient, although weight decay still acts on them. Longer tests of that method therefore involve untrained position rows rather than a table-index error.

No method-specific tuning, pretrained LLM evaluation, natural-text test, or long-context adaptation is performed. The numerical plots and the trained models answer different questions. The appropriate conclusion is about this protocol and its observed failures.

[Section 9: Empirical results and limitations →](09-results.md)
