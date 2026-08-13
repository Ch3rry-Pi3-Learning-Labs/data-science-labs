# Mathematical Notation Guide

## Core rule

Every symbol must be defined in the surrounding text or a notation table before the reader is expected to interpret it.

## Equation numbering

Number important displayed equations by section:

```latex
J(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell_i(\theta). \tag{2.1}
```

Use `(2.1)`, `(2.2)` and so forth inside Section 2. Numbers are local to each notebook and should not inherit numbering from another document.

Refer to numbered equations explicitly:

> Equation (2.1) averages the loss over all observations.

Give important equations stable anchors and use internal links when referring
back to them:

```html
<a id="equation-2-1"></a>
```

```markdown
[Equation (2.1)](#equation-2-1)
```

Intermediate algebra does not need a number unless it is referenced later.

## Definitions and dimensions

Define both meaning and shape where relevant:

> Let (X\in\mathbb{R}^{N\times d}) be the feature matrix, where (N) is the number of observations and (d) is the number of features.

Use a notation table when more than four or five symbols are introduced together.

For every new vector or matrix, show:

1. the generic symbolic form and dimensions;
2. the meaning of its rows, columns or elements;
3. a small concrete example; and
4. its code representation and shape.

For example:

$$
\boldsymbol{\theta}
=\begin{bmatrix}\theta_1\\\theta_2\end{bmatrix}
\in\mathbb{R}^{2},
\qquad
\boldsymbol{\theta}^{(0)}
=\begin{bmatrix}0\\0\end{bmatrix}.
$$

The corresponding NumPy array is `np.array([0.0, 0.0])` with shape `(2,)`.

## Plain-English readings

After an important equation, provide a direct reading:

> Equation (4.1) says: move the current parameter in the direction opposite to the gradient, scaled by the learning rate.

The reading should explain the mathematical operation, not merely repeat the notation aloud.

## Conventions

- Scalars: italic lower-case, such as (x), (y), (alpha).
- Vectors: bold lower-case where useful, such as (mathbf{x}), (oldsymbol{\theta}).
- Matrices: upper-case, such as (X), (W).
- Sets: calligraphic capitals where useful, such as (mathcal{D}).
- Estimated quantities: hats, such as (hat{y}).
- Iterations: superscripts in parentheses, such as (	heta^{(t)}).

State any deliberate deviation from these conventions.

## Worked examples

A hand-worked example should:

1. state the starting values;
2. substitute them into the equation;
3. show the principal arithmetic steps;
4. give the result; and
5. interpret the result in the context of the algorithm.

When a worked example refers to an earlier relationship, identify and link that
equation before calculating. Unless the equation is directly above the example,
restate its generic form before substituting numerical values.

## Derivations

If a derivative or algebraic transformation is part of the lesson, show it in a
traceable sequence:

1. write the starting equation;
2. identify the rule being applied, such as the chain rule;
3. apply the rule without skipping the inner derivative;
4. simplify one meaningful stage at a time; and
5. connect the final expression to the implementation.

Avoid unexplained phrases such as “it follows that” or “differentiating gives”
when the omitted work is precisely what a beginner needs to learn.
