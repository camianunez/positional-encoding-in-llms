# Positional Encoding in LLMs

<p class="book-subtitle">Rotary positional embeddings, relative attention, and context-length generalization</p>

<div class="book-meta">
<span>Author: <a href="https://github.com/camianunez">camianunez</a></span>
<span>Technical monograph</span>
<span>Mathematical analysis and reproducible experiments</span>
</div>

## Abstract

Positional encoding determines how a Transformer represents token order and displacement. This monograph develops Rotary Position Embedding (RoPE) from orthogonal transformations, derives its relative-position identity, and examines the effects of the frequency spectrum on positional similarity. It distinguishes phase cancellation from vector-norm attenuation and from the probabilities obtained after attention normalization.

The analysis connects these properties to autoregressive language models, including multi-head attention, key–value caching, numerical precision, and context extension. A controlled experiment compares six positional methods in 18 trained causal decoders. The results establish task- and length-dependent differences under a fixed training budget; they do not establish a universally optimal encoding.

## Scope and assumptions

The mathematical development assumes familiarity with linear algebra, scaled dot-product attention, and next-token prediction. Positions are deterministic token indices. Unless stated otherwise, queries and keys use the same rotary spectrum, the rotary dimension is even, and displacement is defined as **key position minus query position**.

The central identity is

$$
(R_s q_s)^\top(R_t k_t)=q_s^\top R_{t-s}k_t.
$$

It concerns the positional transformation for fixed content vectors. Model behavior also depends on learned representations, training data, masking, and optimization.

## Links to sections

### Part I. Mathematical foundations

1. **[Position information and attention invariance](01-order.md).** Permutation equivariance, causal masking, and the role of explicit position information.
2. **[Rotary transformations and relative position](02-rotation.md).** Two-dimensional rotations, norm preservation, and the relative-position identity.
3. **[Frequency spectra and positional representations](03-frequencies.md).** Geometric frequency schedules, complex representations, and distances between phase codes.
4. **[Phase cancellation and distance-dependent similarity](04-decay.md).** Positional kernels, Abel summation, and counterexamples to monotonic score decay.
5. **[Matrix generators and norm stability](05-generators.md).** Continuous translation groups, skew-symmetric generators, and the distinction between rotation and attenuation.

### Part II. Language-model integration

6. **[RoPE in autoregressive Transformer layers](06-llms.md).** Query/key projections, multi-head attention, KV-cache consistency, and computational cost.
7. **[Context-length generalization and position scaling](07-long-context.md).** Distribution shift, recurrence, base selection, and position interpolation.

### Part III. Evaluation and implementation

8. **[Evaluation criteria and experimental protocol](08-evaluation.md).** Task definitions, controlled model variants, evaluation metrics, and uncertainty.
9. **[Empirical results and limitations](09-results.md).** Copying and associative-lookup measurements at training and extrapolated lengths.
10. **[Reference implementation and numerical verification](10-implementation.md).** Coordinate pairing, precision, gradients, and cached-decoding checks.
11. **[Technical exercises and terminology](11-exercises.md).** Derivations, experimental extensions, and notation.

## Numerical analysis and experimental evidence

The [interactive phase analysis](explorer.html) evaluates position-dependent rotations and the normalized similarity kernel for a configurable spectrum. It computes deterministic geometric quantities rather than learned attention probabilities.

The [experimental results](09-results.md) report exact answer accuracy and conditional answer perplexity at three context lengths. Tables and figures are generated from the recorded per-seed measurements. Figure captions specify the domain, normalization, and statistical summaries.

## Source document and reproducibility

The primary mathematical source is the supplied eight-page **Rotary Positional Encoding: A Mathematical Summary**. The [source map](source-notes.md) identifies the corresponding sections and the additional assumptions made explicit here. The [bibliography](references.md) provides primary research references.

All derivations, training code, figure-generation scripts, and recorded measurements are available in [camianunez/positional-encoding-in-llms](https://github.com/camianunez/positional-encoding-in-llms). The [reproducibility instructions](reproduce.md) describe validation, figure generation, and training.
