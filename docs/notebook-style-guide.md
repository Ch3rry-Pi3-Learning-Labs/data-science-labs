# CH3RRY PI3 Learning Labs Notebook Style Guide

## Purpose

Learning Labs should make technically serious material feel approachable without weakening its accuracy. A reader should understand not only what code to run, but why each calculation exists, what its output means and how it contributes to the complete method.

The initial standard was distilled from three internal modular teaching notebooks covering ordinary Expectation–Maximisation (EM), variational EM and Gibbs sampling. Those notebooks remain private reference material and are not part of this repository.

## Standard learning sequence

For every substantial algorithmic stage:

1. **Motivate it.** Explain why the next quantity or operation is required.
2. **Define it.** Introduce the notation and mathematical relationship.
3. **Read it.** Translate the equation into direct, plain English.
4. **Implement it.** Use one focused function or code block.
5. **Demonstrate it.** Run a small example and expose a compact output.
6. **Interpret it.** Explain what the output shows and what it does not show.
7. **Connect it.** Carry the result into the next stage of the algorithm.

Only after the individual stages are understood should the notebook assemble the complete algorithm.

## Recommended notebook anatomy

1. Title, short description and intended audience
2. Learning objectives
3. Linked table of contents
4. Imports and reproducibility
5. Problem and intuition
6. Notation and mathematical foundations
7. Modular algorithm development
8. One hand-worked or small numerical example
9. Complete implementation
10. Visualisation and diagnostics
11. Sanity checks and common failure modes
12. Summary and suggested next steps

Sections may change to fit the subject, but a notebook should not omit the learning objectives, contents, definitions, interpretation, checks or summary.

## Navigation and headings

- Use numbered level-two sections: `1.`, `2.`, `3.`.
- Use numbered level-three subsections when a section contains a sequence: `5.1`, `5.2`.
- Give every table-of-contents destination an explicit HTML anchor.
- Use descriptive headings that state the action or concept.
- Use CH3RRY PI3 blue `#4DAAFC` for displayed notebook headings.
- Avoid decorative colour in ordinary body text.

Example:

```html
<a id="learning-rate"></a>
## <font color="#4DAAFC"><strong>4. Choose the learning rate</strong></font>
```

## Explanatory rhythm

Prefer short transitions such as:

- “We need this quantity because…”
- “Equation (3.1) can be read as…”
- “The next cell implements exactly this update.”
- “The output confirms…”
- “This result becomes the input to…”

Do not assume that a formula, plot or array is self-explanatory. Explicitly show the reader how to read it.

## Tables

Use tables when they make a mapping easier to understand, particularly for:

- notation and symbol definitions;
- array shapes;
- function inputs and outputs;
- comparisons between methods;
- hyperparameters and their effects; and
- diagnostic interpretations.

Avoid large tables that merely repeat prose.

## Code and output sequence

- Separate reusable definitions from calls that demonstrate them.
- Prefer one meaningful calculation per function.
- Display compact representative output, not an uncontrolled dump.
- Introduce array shapes before relying on them.
- State explicitly when Python uses zero-based indexing but the mathematics uses one-based indexing.
- Keep reusable functions independent of demonstration constants wherever practical.
- The complete runner should call the same focused functions developed earlier.

## Visuals

Every plot should have a purpose stated before it and an interpretation afterwards. Include:

- a clear title;
- labelled axes;
- a legend when multiple quantities appear;
- units where relevant;
- accessible colour contrast; and
- a short explanation of the pattern the reader should inspect.

## Reproducibility

- Declare the random seed near the start.
- Keep data generation separate from model fitting.
- State package assumptions.
- Run the published notebook from a clean kernel from beginning to end.
- Do not publish hidden dependencies on variables from previous sessions.

## Quality boundary

A Learning Lab is ready only when it is mathematically correct, independently executable, visually legible and understandable to its intended reader. Technical sophistication is not a substitute for teaching clarity.

