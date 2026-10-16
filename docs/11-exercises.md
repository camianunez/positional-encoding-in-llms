# 11. Technical exercises and terminology

## Derivations and experimental extensions

**1. Bilinear score expansion.** Expand $(a,b)R(\Delta\theta)(c,d)^\top$. Which coefficient multiplies sine? What changes if you define displacement as query minus key?

*Check:* the sine coefficient for this book's convention is $bc-ad$; changing displacement sign reverses it.

**2. Construction of a distant score maximum.** Pick a nonzero vector $q$ and target displacement 5000. Construct $k$ so the relatively rotated dot product is maximal there.

*Check:* $k=R_{-5000}q$; the score at the target is $\|q\|^2$. This refutes a universal monotonic score-decay claim.

**3. Orthogonality and transpose/inverse equivalence.** A family obeys $A(s+t)=A(s)A(t)$. Can you immediately replace $A(s)^\top$ by $A(-s)$?

*Check:* no. You need orthogonality, or a different dual query/key construction. A shear is a counterexample.

**4. Continuous periodicity and integer token indices.** Set the phase-analysis interface to a two-dimensional head and frequency 1. Compare offsets 6 and 7. Why is the continuous period $2\pi$ not an exact integer-token period?

*Check:* token offsets are integers; $2\pi$ is not. Near-alignment and exact recurrence are different claims.

**5. Dependence of method rankings on evaluation length.** Using metrics.csv, rank methods by 32-token copying, then 128-token copying. Why does a changed ranking not contradict the rotation theorem?

*Check:* the theorem constrains a positional factor. It does not determine training dynamics or generalization.

**6. Retrieval evaluation with shortcut controls.** Propose an experiment that separates correct key–value binding from guessing frequent values. Specify a control, target-location analysis, and additional training budget.

*One approach:* use balanced unique values where feasible, compare to a bag-of-values baseline, stratify by target distance, vary distractor count independently of distance, and test unseen examples. Declare the protocol before inspecting results.

**7. Cached-decoding position errors.** In a private copy of the model, reset cached decoding positions to zero. Which tests fail and why?

*Check:* full-prefix/cached equivalence should fail because the query and newly cached key use incorrect relative phases. Do not modify checked-in benchmark sources while expecting their stored hash to match.

**8. Controlled comparison of rotary bases.** Compare bases 100, 10,000, and 1,000,000. What must remain fixed?

*Check:* architecture, data, seeds, optimizer, evaluation streams, and budget. Use a separate output directory and record the changed configuration. A phase-kernel sweep alone is not a trained-model comparison.

## Terminology and notation

| Term | Meaning here |
|---|---|
| Attention head | A learned query/key/value subspace and its attention computation |
| Logit | An unnormalized score before softmax |
| Rotary pair | Two coordinates rotated together at one frequency |
| Frequency | Angular change in radians per token |
| Base | Parameter controlling the geometric frequency spread |
| Relative displacement | Key position minus query position, $\Delta=t-s$ |
| Lag | Nonnegative causal distance, $r=s-t$ |
| Orthogonal transform | A linear map preserving Euclidean dot products and norms |
| Generator | Matrix $X$ in a continuous family $e^{tX}$ |
| Phase kernel | $C(\Delta)$, the mean cosine alignment across rotary pairs |
| Phase recurrence | Two offsets returning close to the same joint orientations |
| Extrapolation | Evaluating beyond the training position/length range |
| Position interpolation | Dividing positions by a fixed scale to alter phase geometry |
| KV cache | Stored earlier keys and values used during generation |
| Answer perplexity | Exponential of average NLL on held-out answer tokens only |
| Pareto frontier | Options where improving one objective sacrifices another |

[Reproducibility and publication →](reproduce.md)
