# Reproducibility and publication

## Source code, figures, and recorded measurements

Source chapters, PNG and SVG figures, per-seed experimental JSON, aggregated JSON, CSV metrics, and mathematical diagnostics are checked in. The supplied PDF is preserved at the root. Read the chapters directly on GitHub or build the searchable HTML edition.

The recorded training run uses CPU PyTorch. Environment files record Python, Torch, platform, source hash, figure-library versions, seeds, and protocol.

## Environment installation

Use Python 3.11–3.13 and a virtual environment. On Windows PowerShell:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
~~~

On macOS/Linux, activate with <code>source .venv/bin/activate</code>. The recorded commands use CPU execution. CUDA execution has not been evaluated.

The recorded package versions are in environment.json and figure-environment.json under experiments. Requirements constrain tooling; they are not a bit-for-bit cross-platform environment lock.

## Verification of recorded results

~~~sh
python -m pytest -q
python scripts/report.py --check
python scripts/verify_book.py
python -m mkdocs build --strict
~~~

Tests check mathematical and decoder behavior. The report recomputes metrics/tables from all 18 raw runs and verifies the training source hash. The book checker checks source identity, local links, navigation, and graph assets. Strict site building checks Markdown/HTML configuration.

## Mathematical figure generation

~~~sh
python scripts/diagnostics.py
~~~

This computes seven mathematical diagnostics and exports each figure as SVG and high-resolution PNG. The domains and normalization are explicit; the random-content calculation uses a fixed NumPy seed. No model training is required.

The figure-generation procedure uses a shared specification in scripts/figure_style.py. It fixes the dark background, font sizes, axis treatment, semantic colors, and export settings. The HTML edition embeds SVG so labels and curves retain their resolution.

Dense kernel plots use consecutive 128-offset blocks, displaying the exact block mean and minimum/maximum rather than overlapping thousands of traces. The result figures separate task and context length into panels and show seed means with standard-deviation bars. Training trajectories show mean loss with the observed seed range. Each caption states the corresponding statistic.

## Decoder training and result aggregation

~~~sh
python scripts/train.py
python scripts/report.py
~~~

Defaults are six methods, three seeds, 1200 updates per model, and 768 held-out examples per task/length/seed. Existing per-run JSON files are skipped after source and protocol checks.

For a fresh rerun, use a separate directory:

~~~sh
python scripts/train.py --out experiments/rerun
~~~

The report intentionally targets experiments/results. Compare a rerun's individual JSON files or explicitly replace the recorded dataset after review and regenerate the report. Do not mix protocols.

For a quick pipeline smoke run:

~~~sh
python scripts/train.py --out experiments/smoke --methods rope --seeds 11 --steps 20 --eval-batches 1
~~~

This cannot substantiate the book's full results. Training saves local checkpoints under experiments/checkpoints; they are excluded from source control because they can be regenerated and raw evaluation records are small enough to inspect.

For a changed base, architecture, task, or length, edit the source, use a new output directory, and treat it as a new protocol. The stored hash prevents modified code from being silently presented as the origin of old measurements.

## Static-book build and local serving

~~~sh
python -m mkdocs serve
python -m mkdocs build --strict
~~~

Serve prints a local address; build writes site/. The [interactive explorer](explorer.html) is a self-contained HTML file that also opens after download. The HTML book uses a pinned MathJax CDN script for equations; GitHub renders the chapters' dollar-delimited source math itself.

## Browser rendering and interaction verification

With Chrome/Chromium installed, or Playwright Chromium available:

~~~sh
python -m pip install -r requirements-browser.txt
python scripts/browser_check.py
~~~

If no system browser is available, first run <code>python -m playwright install chromium</code>.
The check starts a local HTTP server, verifies rendered equations and loaded images on every page,
exercises the explorer, and saves screenshots and a JSON report under artifacts/browser.

## GitHub repository and Pages deployment

The project is associated with [camianunez](https://github.com/camianunez); its source home is [camianunez/positional-encoding-in-llms](https://github.com/camianunez/positional-encoding-in-llms).

The validation workflow runs tests, report checks, book verification, and strict site building, then uploads a downloadable HTML artifact. A separate Pages workflow deploys on the default branch or manual invocation.

For a fork, select **Settings → Pages → Build and deployment → Source → GitHub Actions**, and update site_url and repo_url in mkdocs.yml. See [GitHub's instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

## Figure provenance and rendering specification

The graph manifest identifies each figure's computational source, shared rendering module, dimensions, domains, and data. The figure environment records hashes of the diagnostic, report, and shared style scripts. The rendering specification is recorded in experiments/diagnostics/figure-style.json.

Figures are generated from the recorded training measurements and mathematical diagnostics. Training sources, per-seed metrics, and the supplied PDF are included. Captions distinguish mathematical constructions, exact block ranges, and variability across trained-model seeds.
