# 6. RoPE in autoregressive Transformer layers

## Position of RoPE within the attention computation

An autoregressive decoder reads a prefix $x_0,\ldots,x_s$ and predicts $x_{s+1}$. A simplified block computes:

~~~text
token IDs → embeddings → normalization
                        ↓
                 learned Q, K, V projections
                        ↓
                reshape into attention heads
                        ↓
                 apply RoPE to Q and K
                        ↓
               scaled dot products + causal mask
                        ↓
                     softmax
                        ↓
                weighted sum of V → output projection
                        ↓
              residual connection → feed-forward block
~~~

There are many implementation variants, but the location of the standard rotary operation is the important part. The original [LLaMA architecture](https://arxiv.org/html/2302.13971v1#S2.SS2) replaces absolute positional embeddings with RoPE at each layer. Our lab uses a much smaller decoder with LayerNorm and GELU so that its positional mechanism is easy to inspect.

The queries, keys, and values are contextual hidden-state projections. They are not simply fixed dictionary embeddings of words. Later layers can combine content, order, and earlier attention results.

For each head,

$$
Q'_s=R_sW_Qh_s,\quad K'_t=R_tW_Kh_t,\quad V_t=W_Vh_t.
$$

The causal head output is

$$
o_s=\sum_{t\le s}
\operatorname{softmax}_{t}\left(
\frac{(Q'_s)^\top K'_t}{\sqrt{d_h}}\right)V_t.
$$

The language-model output layer eventually produces logits over the vocabulary. Training minimizes the negative log probability of the correct next token. Position changes which earlier information can influence those logits usefully.

## Token displacement and learned dependencies

Language-model dependencies include local syntactic relations, long-range agreement, variable binding, and retrieval of prior definitions. Their relevant displacements span different scales, so a fixed recency preference does not determine the appropriate attention target.

RoPE gives heads relative phase features. Training must teach the network to use those features with content. The transform does not hard-code grammatical rules or a retrieval algorithm.

The same word can be one or several tokens. RoPE measures token offsets, not semantic distance, word count, elapsed time, or line number. Tokenization changes the relationship between these quantities.

## Multi-head attention and shared key projections

Different heads have different learned projections and can interpret the same frequency schedule differently. If only part of a head is rotated, the untouched dimensions carry a position-independent dot-product contribution. The rotated dimension must be even and must match the checkpoint's convention.

Grouped-query attention can share keys and values among several query heads. Corresponding query/key coordinates still need compatible rotary frequencies and pairing. Sharing keys does not remove the need for consistent positions.

## Autoregressive generation and KV-cache consistency

During generation, an LLM avoids recomputing every previous key and value. It stores a cache for each layer. Standard RoPE integrates naturally:

1. Compute the new token's position.
2. Rotate its newly projected query and key.
3. Append the already-rotated key and the value to the cache.
4. Compare the new query against cached keys.
5. Preserve position indices for later steps.

If a prefix has length $L$, the next token's position is $L$, even though the current input tensor may have sequence length one. Starting every generation step at position zero changes relative phases.

Cached keys must not be rotated again. If a key at position 20 already contains $R_{20}$, applying $R_{20}$ a second time gives $R_{40}$. That is a different position.

The reference model supports chunks as well as one-token steps. The tests compare full-prefix computation with several cached chunks for every positional method, so the check covers masks, offsets, and cache composition together.

## Padding, document packing, and sliding windows

Left padding is not automatically real context. Assign positions and masks consistently so a padded batch represents the intended unpadded sequence. For independently packed documents, resetting positions and preventing cross-document attention are separate decisions: a reset alone does not isolate documents.

A sliding window drops old keys but does not justify silently resetting only the new query's position. Use the original positions, or consistently rebase both sides with the same fixed rotary schedule. With fixed content and a fixed schedule, a common rebase preserves their positional dot product.

If a dynamic scaling rule changes the frequency spectrum midway through generation, old cached keys and new queries can belong to different rotary coordinate systems. Recompute or consistently transform the cache according to the actual scheme.

## Computational and memory complexity

Applying RoPE costs $O(Ld)$ per layer's token representations. Dense attention still computes interactions with $O(L^2d_h)$ arithmetic during full-sequence processing, summed across heads. Fused implementations can avoid materializing the whole score matrix without removing all that arithmetic.

A generation cache grows roughly as $O(LH_{\rm KV}d_h)$ per layer. Extending the context still has substantial time and memory costs. RoPE supplies a position mechanism; it does not make those costs vanish.

[Section 7: Context-length generalization and position scaling →](07-long-context.md)
