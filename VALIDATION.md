# Validation

Validated locally on 2026-10-16 before publication.

## Content and presentation

- The title is **Positional Encoding in LLMs** throughout the reading interface and project introduction.
- All 15 book pages use a permanent dark scheme, independent of system light/dark preferences.
- No palette selector or theme-switch control is rendered.
- Chapter and subsection headings identify their mathematical, implementation, or experimental subject explicitly.
- The introduction and exposition use definitions, assumptions, and results in place of extended analogies.
- Section navigation and search work in the book header.
- The book and numerical appendix pass mobile overflow and interaction checks.

## Figures and mathematical rendering

- Ten figures share a single rendering specification and muted dark palette.
- Primary figure assets are native vector SVG, including outlined glyphs for consistent rendering across platforms; high-resolution PNG companions are retained.
- Dense phase statistics show declared block means and exact within-block ranges.
- Experimental panels show seed means and sample standard deviations; training curves show seed means and observed seed ranges.
- All figure sources and statistics are identified in the graph manifest.
- A Chromium browser rendered **182 mathematical expressions** across the 15 pages without MathJax errors.
- All embedded figures loaded; no JavaScript exceptions were recorded.
- The numerical appendix passes common-shift, interpolation, zero-offset, keyboard, and mobile-layout checks.

## Experimental and source integrity

- **27 correctness tests passed.**
- All **18 trained-decoder records**, **126 held-out metric records**, and **42 aggregate records** were checked.
- All recorded training-source hashes and the supplied PDF fingerprint were verified.
- Mathematical diagnostics declare their domains, normalization, and random seed.
- MkDocs strict mode and local-link verification passed for source and built HTML.

Reproduction commands and interpretation limits are in [the book](docs/reproduce.md). Browser verification requires requirements-browser.txt and Chrome/Chromium or Playwright Chromium.

The synthetic experiments support conclusions within their stated protocol; they do not establish a universally optimal positional encoding or real-LLM performance.
