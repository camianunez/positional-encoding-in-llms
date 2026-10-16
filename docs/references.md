# Research and software references

The mathematical starting point is the [supplied eight-page note](source-notes.md). These primary sources give the research and software context. All figures and experimental results in this repository were newly generated.

1. **Vaswani et al. (2017), Attention Is All You Need.** [Paper](https://arxiv.org/abs/1706.03762). Attention and additive sinusoidal position features.

2. **Su et al. (2021; revised 2023), RoFormer: Enhanced Transformer with Rotary Position Embedding.** [Paper](https://arxiv.org/abs/2104.09864) · [HTML](https://arxiv.org/html/2104.09864v5) · [Authors' implementation](https://github.com/ZhuiyiTechnology/roformer). The original rotary method and finite phase-sum analysis. This book does not reproduce its pretraining or downstream benchmarks.

3. **Press, Smith, and Lewis (2022), Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation.** [Paper](https://arxiv.org/abs/2108.12409). ALiBi's additive, head-dependent relative-distance penalties.

4. **Touvron et al. (2023), LLaMA: Open and Efficient Foundation Language Models.** [Paper](https://arxiv.org/abs/2302.13971) · [Architecture](https://arxiv.org/html/2302.13971v1#S2.SS2). A concrete LLM using RoPE. The lab is not a reproduction of LLaMA.

5. **Chen et al. (2023), Extending Context Window of Large Language Models via Positional Interpolation.** [Paper](https://arxiv.org/abs/2306.15595). Position rescaling with pretrained-model adaptation. The frozen lab intervention omits adaptation.

6. **Peng et al. (2023), YaRN: Efficient Context Window Extension of Large Language Models.** [Paper](https://arxiv.org/abs/2309.00071). An extension procedure beyond uniform interpolation. This repository does not benchmark YaRN.

7. **Puranik (2026), Using group theory to explore the space of positional encodings for attention.** [Author's article](https://blog.janestreet.com/using-group-theory-to-explore-positional-encodings-attention/). The group-theoretic source cited in the supplied notes.

## Software references

- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), used by the tiny decoder.
- [Material for MkDocs math support](https://squidfunk.github.io/mkdocs-material/reference/math/), used by the HTML book.
- [GitHub Pages publishing sources](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site), used by the publication workflow.

Research claims are linked near their context. Custom identities and counterexamples are derived in the chapters. Diagnostics and benchmark results come from the lab's executable code.
