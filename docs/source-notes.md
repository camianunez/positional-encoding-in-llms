# Source document and mathematical assumptions

The workspace originally contained **RoPE_Positional_Encoding_Academic_Notes.pdf**. Its visible title is **Rotary Positional Encoding: A Mathematical Summary**, with subtitle **Relative Position, Matrix Exponentials, Generators, and Long-Context Behavior**. It has eight pages. The PDF does not identify an author in its title block or metadata.

[Read the supplied PDF](https://github.com/camianunez/positional-encoding-in-llms/blob/main/RoPE_Positional_Encoding_Academic_Notes.pdf) · [File provenance](https://github.com/camianunez/positional-encoding-in-llms/blob/main/experiments/paper-manifest.json)

This is the source requested by the user, distinct from Su and colleagues' original RoFormer research paper.

## Correspondence between the source document and chapters

| Supplied note sections | Subject | Book development |
|---|---|---|
| 1 | Position-free attention | [Order and the causal-mask caveat](01-order.md) |
| 2–3 | Groups, generators, and differential equations | [Assumptions and matrix exponentials](05-generators.md) |
| 4–5 | Relative rotation and vector geometry | [Derivation with fixed displacement sign](02-rotation.md) |
| 6 | Multiple frequencies | [Frequency spectra and phase representations](03-frequencies.md) |
| 7 | Positional similarity kernel | [Cancellation and counterexamples](04-decay.md) |
| 8 | Long context and aliasing | [Integer positions, recurrence, and scaling](07-long-context.md) |
| 9–11 | Generator classes | [Stability and distinction from ALiBi](05-generators.md) |
| 10, 12–13 | Task-dependent choice and key equations | [Evaluation](08-evaluation.md) and [implementation](10-implementation.md) |

## Additional assumptions and mathematical qualifications

- The group formulation assumes invertibility and $A(0)=I$. Composition alone admits degenerate families.
- A continuous real generator is an assumption beyond a sequence defined only at integer positions.
- Translation composition alone does not imply $A(s)^\top A(t)=A(t-s)$; orthogonality supplies the transpose/inverse relation.
- Purely imaginary eigenvalues do not ensure Euclidean norm preservation in a nonorthogonal basis.
- The kernel is not an attention probability, and oscillations do not imply monotonic decay of every learned logit.
- A shear's polynomial dependence connects mathematically to linear functions, but a naive shear of content is not ALiBi's additive bias.

The experiments, original plots, decoder implementation, cache checks, and frozen interpolation study are additions to the notes.

The notes cite [Alok Puranik's Jane Street article](https://blog.janestreet.com/using-group-theory-to-explore-positional-encodings-attention/), which develops a group-theoretic view of linear positional transforms. This book uses the supplied notes as its immediate mathematical source.

## Document provenance and attribution

The manifest records the original PDF's SHA-256, page count, and extracted title. The source file is preserved unchanged. New text and figures are separately licensed; the project does not assign a new license to the PDF or referenced papers.

[Primary references →](references.md)
