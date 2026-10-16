# 10. Reference implementation and numerical verification

A compact implementation is easier to audit against the derivation than a dense matrix construction.

## Adjacent-pair rotation implementation

The executable reference lives in [rope_lab/positional.py](https://github.com/camianunez/positional-encoding-in-llms/blob/main/rope_lab/positional.py). Its central operation is:

~~~python
# x: [..., sequence, head_dim]; positions: [sequence]
theta = base ** (-torch.arange(0, head_dim, 2) / head_dim)
angle = positions[:, None] * theta[None, :] / scale
cos, sin = angle.cos(), angle.sin()
a, b = x[..., 0::2], x[..., 1::2]
rotated = torch.stack(
    (a * cos - b * sin, a * sin + b * cos), dim=-1
).flatten(-2)
~~~

In practice, construct the frequency tensor on the same device as the input, validate an even rotary dimension, compute phases in at least float32, and cast back to the desired representation dtype. The repository implementation handles these details and preserves float64 for the high-precision tests.

For shape $[B,H,L,d_h]$, sine/cosine shape $[L,d_h/2]$ broadcasts across batch and heads. A per-example position tensor would require corresponding broadcasting changes; this reference intentionally accepts one shared one-dimensional position vector.

## Phase-computation precision and index resolution

Low-precision position construction can lose distinctions before sine and cosine are computed. At sufficiently large indices, adjacent integers map to the same floating-point value. Bfloat16 is particularly coarse, while even float32 eventually cannot distinguish all integers beyond $2^{24}$.

Construct integer position indices first, then convert to an appropriate phase-computation dtype. Casting an already-rounded bfloat16 index to float32 cannot restore the lost index. Computing phases in float32 mitigates common errors, but does not prove infinite-context numerical accuracy.

Trigonometrical argument reduction and finite precision also introduce small discrepancies in composition identities at large angles. Tests need tolerances appropriate to the dtype and position range.

## Verified mathematical and decoder invariants

The suite verifies:

- Norm preservation, composition, and inverse rotations.
- The relative-position identity and common-shift invariance.
- An explicit dense-matrix reference, catching sign and pairing errors.
- A distant-alignment counterexample to universal decay.
- The origin dependence caused by symmetric exponential damping.
- Position interpolation as a fixed frequency change.
- Rotation gradients through a finite-difference check.
- Full-prefix/cached equivalence for every positional method and several chunk sizes.
- Causal isolation: changing future tokens cannot change earlier logits.
- Target alignment in both generated tasks.
- Identical initialization of common parameters across methods.
- Expected masking and ALiBi penalty direction.
- Input validation.

These tests constrain the mechanisms actually used in training; they do not assert that one method must win a benchmark. In particular, the cache check exercises the decoder's end-to-end position, mask, and cache handling.

## Coordinate-basis transformations for checkpoint compatibility

Suppose implementation A uses adjacent pairs, and B pairs the first half of the coordinates with the second half. To transfer a checkpoint, find the coordinate permutation $P$ relating their rotary bases. Correspondingly permute the projected query/key coordinates and frequency assignment so that

$$
R^{(B)}_s=P R^{(A)}_s P^\top.
$$

Then the same underlying interaction can be represented. Substituting B's rotation function while leaving A's projection weights untouched generally changes the model.

## Scope and limitations of implementation checks

Norm preservation is a desirable stability property, but says nothing about whether a head learned to copy the correct token. Common-shift invariance is conditional on fixed content, not arbitrary changed prompts. Cache equivalence proves agreement with full-prefix computation within the tested configuration; it does not validate a different dynamic scaling algorithm.

The distinction lets the [measured failures](09-results.md) coexist coherently with passing correctness checks.

[Section 11: Technical exercises and terminology →](11-exercises.md)
