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

## The no-hidden-steps rule

The reader should never have to reverse-engineer where a displayed value, formula or code operation came from. Before substituting numerical values:

1. identify and link the equation being used;
2. restate its generic form when the worked calculation is not immediately adjacent;
3. list the values being substituted;
4. show the substitution itself;
5. show the important arithmetic steps; and
6. interpret the result and explain what uses it next.

For example, do not jump directly from a stated objective to `J(0) = 9`. Show
the generic objective, substitute the parameter value, calculate the result and
then explain what that loss represents.

Derivations should follow the same principle. Do not write “differentiating gives” when the derivative is a learning objective or is needed to understand the implementation. Show the relevant rule, its application to the current expression, the simplified result and its connection to the code. A derivation may be compressed only when it has been declared prerequisite knowledge and is not central to the lesson.

## Recommended notebook anatomy

1. Title, short description and intended audience
2. Visual learning journey
3. Learning objectives
4. Linked table of contents
5. Imports and reproducibility
6. Problem and intuition
7. Notation and mathematical foundations
8. Modular algorithm development
9. One hand-worked or small numerical example
10. Complete implementation
11. Visualisation and diagnostics
12. Sanity checks and common failure modes
13. Summary and suggested next steps

Sections may change to fit the subject, but a notebook should not omit the learning objectives, contents, definitions, interpretation, checks or summary.

The opening description should also provide a short roadmap: what the notebook will build, why the stages occur in that order and what the final result will allow the reader to do.

Follow the description with a compact **Learning journey**. Show the major stages as a left-to-right flow and link each stage to the section where it is developed. Accompany the flow with a small table explaining what the reader will do at each stage. The journey should reflect the notebook's actual structure; it is a navigation and expectation-setting device, not decoration.

Every principal section should begin with two or three sentences explaining:

- why the section is needed at this point;
- what it will introduce or produce; and
- how that output connects to the wider learning journey.

## Navigation and headings

- Use numbered level-two sections: `1.`, `2.`, `3.`.
- Use numbered level-three subsections when a section contains a sequence: `5.1`, `5.2`.
- Give every table-of-contents destination an explicit HTML anchor.
- Use descriptive headings that state the action or concept.
- Use CH3RRY PI3 blue `#4DAAFC` for displayed notebook headings.
- Avoid decorative colour in ordinary body text.
- Give each important numbered equation an explicit HTML anchor.
- Link references such as `[Equation (3.1)](#equation-3-1)` back to that anchor.
- Keep the visible equation number and anchor name stable when revising prose.

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

Use a blockquote for a genuinely important intuition, warning, boundary or
takeaway. It should help the reader navigate the argument rather than merely
decorate ordinary prose.

## Tables

Use tables when they make a mapping easier to understand, particularly for:

- notation and symbol definitions;
- array shapes;
- function inputs and outputs;
- comparisons between methods;
- hyperparameters and their effects; and
- diagnostic interpretations.

Avoid large tables that merely repeat prose.

Use a readable table text size, but let the renderer determine the table's natural width. Do not stretch a compact table merely to fill the notebook column.

## Displayed mathematics

- Keep inline mathematics at the surrounding body-text size.
- Render standalone display equations slightly larger so that symbols, superscripts and subscripts remain legible in GitHub's notebook view.
- Introduce a generic rule before substituting a lesson-specific expression when that rule is important to the reader's intuition.
- Define every symbol in the generic form, then explicitly map the concrete terms onto it before simplifying.

## Code and output sequence

- Separate reusable definitions from calls that demonstrate them.
- Prefer one meaningful calculation per function.
- Display compact representative output, not an uncontrolled dump.
- Introduce array shapes before relying on them.
- State explicitly when Python uses zero-based indexing but the mathematics uses one-based indexing.
- Keep reusable functions independent of demonstration constants wherever practical.
- The complete runner should call the same focused functions developed earlier.

When code uses vectors or matrices, show both representations before relying on
them:

- the generic mathematical object and its dimensions;
- a small concrete numerical example;
- the corresponding NumPy shape and orientation; and
- the relationship between mathematical and Python indexing where relevant.

For example, define a parameter vector symbolically before showing a concrete
two-parameter vector and then its NumPy representation.

## Closing summary

The final summary should do more than announce completion. It should:

- revisit the original learning journey;
- collect and link the principal equations;
- explain how those equations combine into the final algorithm;
- distinguish what has been demonstrated from what remains unproven; and
- identify a small number of purposeful next steps.

## Visuals

Every plot should have a purpose stated before it and an interpretation afterwards. Include:

- a clear title;
- labelled axes;
- a legend when multiple quantities appear;
- units where relevant;
- accessible colour contrast; and
- a short explanation of the pattern the reader should inspect.

Use consistent native figure sizes rather than forcing every plot to the maximum notebook width. In GitHub's current 894-pixel repository iframe, Markdown occupies 724 pixels beginning 17 pixels to the right of rich image output. Published plots are therefore normalised to a 724-pixel visible width and receive a measured 17-pixel transparent left offset. They display at native size, aligning the visible canvas with the surrounding text without right padding or responsive shrinking. Prefer a restrained, light horizontal grid when guides aid interpretation; avoid a dense full grid by default.

## Reproducibility

- Declare the random seed near the start.
- Keep data generation separate from model fitting.
- State package assumptions.
- Run the published notebook from a clean kernel from beginning to end.
- Do not publish hidden dependencies on variables from previous sessions.

## Quality boundary

A Learning Lab is ready only when it is mathematically correct, independently executable, visually legible and understandable to its intended reader. Technical sophistication is not a substitute for teaching clarity.
