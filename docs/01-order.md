# 1. Position information and attention invariance

## Token order in language modeling

An autoregressive language model estimates the conditional distribution of a token given its preceding sequence. This distribution depends on token order: a permutation can change syntactic roles, variable scope, or the interpretation of a negation while preserving the token inventory.

A positional mechanism supplies coordinates or displacement features that can distinguish otherwise identical content representations at different sequence locations. Explicit position information is therefore part of the model's inductive bias.

## Query–key compatibility without position information

For query $q_s$ at position $s$ and key $k_t$ at position $t$, a head computes

$$
\ell_{st}=\frac{q_s^\top k_t}{\sqrt{d_h}},
$$

where $d_h$ is the head dimension. The logit is normalized over allowed keys by softmax, and the resulting weights combine the corresponding values.

This expression has no explicit dependence on $s$ or $t$. For a fixed query, identical key vectors produce identical logits regardless of their assigned positions. Position can enter through modified representations, query/key transforms, relative score terms, or attention biases.

The [original Transformer](https://arxiv.org/abs/1706.03762) adds sinusoidal position features to token representations. RoPE instead transforms projected queries and keys.

## Permutation equivariance of unmasked self-attention

Without position information, write unmasked self-attention as

$$
A(X)=\operatorname{softmax}_{\rm row}
\left(\frac{XW_Q(XW_K)^\top}{\sqrt{d_h}}\right)XW_V.
$$

For a permutation matrix $P$,

$$
A(PX)=P A(X).
$$

To obtain this identity, the permutation reorders both axes of the score matrix; row-wise softmax and value aggregation respect the same reordering. The operation is **permutation equivariant** and has no independent representation of sequence order. The test suite verifies this identity numerically.

## Order information supplied by causal masking

A causal mask permits position $s$ to attend only to positions $t\le s$. Arbitrary token permutations do not preserve that accessibility relation. Prefix boundaries and layer-wise composition can therefore provide implicit order information even without an explicit positional embedding.

Consequently, a causal decoder without RoPE is not necessarily unable to solve every order-sensitive task. The experimental **none** baseline retains the causal mask and measures performance under this constraint.

## Position information in the evaluated tasks

The experimental **lag-four copying** task requires the value exactly four positions before a final query. Repeated values prevent content identity from uniquely identifying the required occurrence.

The **associative lookup** task requires binding a key to its adjacent value and retrieving that value from an earlier list. It combines local association with retrieval over a variable distance.

These tasks impose different requirements on positional representations. The experiments evaluate those requirements under a common architecture and training budget; explicit position information alone does not determine the learned solution.

[Section 2: Rotary transformations and relative position →](02-rotation.md)
