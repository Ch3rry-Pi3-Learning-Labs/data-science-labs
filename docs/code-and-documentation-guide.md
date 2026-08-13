# Code and Documentation Guide

## Principles

Learning Labs code is educational code: readable enough to teach from, robust enough to trust and sufficiently general to reuse in related examples.

## Functions

- Give each function one clear responsibility.
- Use descriptive `snake_case` names.
- Add type hints to public teaching functions.
- Prefer explicit intermediate variables over compressed expressions.
- Validate important public inputs at their boundary.
- Keep demonstration data outside reusable functions.

## NumPy-style docstrings

Use NumPy-style docstrings for reusable teaching functions:

```python
def mean_squared_error(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate the mean squared prediction error.

    Parameters
    ----------
    observed
        Observed target values with shape ``(N,)``.
    predicted
        Predicted target values with shape ``(N,)``.

    Returns
    -------
    float
        Mean of the squared observation-level errors.

    Raises
    ------
    ValueError
        If the two arrays do not have the same non-empty shape.
    """
```

Include `Notes`, `Examples` or `References` only when they add genuine value.

## Comments

Comments should explain mathematical correspondence, shapes, safeguards or non-obvious decisions. Avoid comments that simply translate syntax into English.

Useful:

```python
# Average over observations so the gradient scale does not grow with N.
gradient = errors.mean()
```

Unhelpful:

```python
# Calculate the mean.
gradient = errors.mean()
```

## Numerical and statistical safeguards

Where relevant, explain and implement:

- finite-value checks;
- shape validation;
- division or logarithm safeguards;
- reproducible random generators;
- convergence limits;
- train/test separation; and
- the distinction between demonstration evidence and general performance.

## Complete runners

The final algorithm runner should compose the same focused functions already developed in the notebook. Avoid replacing the teaching implementation with an unrelated library call at the end.

