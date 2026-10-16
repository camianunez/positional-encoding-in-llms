# Positional Encoding in LLMs

A technical monograph on rotary positional embeddings, relative attention, and context-length generalization.

**[Read the book](https://camianunez.github.io/positional-encoding-in-llms/)** · [Source chapters](docs/index.md) · [Experimental results](docs/09-results.md) · [Reproducibility](docs/reproduce.md)

The monograph derives RoPE from orthogonal transformations and matrix generators, examines phase cancellation and frequency selection, and connects these properties to autoregressive Transformer layers and KV caching. Its mathematical starting point is the supplied **Rotary Positional Encoding: A Mathematical Summary**.

The empirical study compares six positional methods across 18 small causal-decoder training runs. It reports accuracy and conditional answer perplexity at training and extrapolated sequence lengths. The conclusions are specific to the tasks and training budget.

## Contents

1. [Position information and attention invariance](docs/01-order.md)
2. [Rotary transformations and relative position](docs/02-rotation.md)
3. [Frequency spectra and positional representations](docs/03-frequencies.md)
4. [Phase cancellation and distance-dependent similarity](docs/04-decay.md)
5. [Matrix generators and norm stability](docs/05-generators.md)
6. [RoPE in autoregressive Transformer layers](docs/06-llms.md)
7. [Context-length generalization and position scaling](docs/07-long-context.md)
8. [Evaluation criteria and experimental protocol](docs/08-evaluation.md)
9. [Empirical results and limitations](docs/09-results.md)
10. [Reference implementation and numerical verification](docs/10-implementation.md)
11. [Technical exercises and terminology](docs/11-exercises.md)

The HTML edition uses a permanent dark scheme. The ten mathematical and empirical figures share one plotting specification and are exported as SVG and high-resolution PNG. The [interactive phase analysis](docs/explorer.html) is self-contained.

## Installation and validation

Python 3.11–3.13:

~~~sh
python -m venv .venv
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/report.py --check
python scripts/verify_book.py
python -m mkdocs build --strict
~~~

Activate the environment before installing dependencies. Full training and figure-regeneration commands are in [the reproduction guide](docs/reproduce.md).

## Repository and attribution

Published by [camianunez](https://github.com/camianunez) at [camianunez/positional-encoding-in-llms](https://github.com/camianunez/positional-encoding-in-llms).

New code is licensed under [MIT](LICENSE-CODE); new text and figures are licensed under [CC BY 4.0](LICENSE-TEXT). The supplied PDF retains its own rights and is documented in the [source map](docs/source-notes.md).
