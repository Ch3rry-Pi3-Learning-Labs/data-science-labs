# CH3RRY PI3 Learning Lab Planning Brief

## Working title

Feature Scaling: Why Units Change Gradient Descent

## Intended audience

Beginning data scientists and data analysts who can read basic Python and NumPy code and have met mean squared error and gradient descent. The preceding gradient-descent Learning Lab is helpful but not compulsory because the required regression formulas are recapped here.

## Learning need

Feature scaling is often taught as a preprocessing command rather than as a mathematical change of coordinates. This notebook shows why differently sized units create uneven gradient magnitudes, how standardisation changes the optimisation route without changing the underlying observations, and how to avoid train/test leakage.

## Learning outcomes

By the end, the reader should be able to:

1. explain why feature units affect gradient-descent updates;
2. calculate and implement training-set standardisation;
3. compare optimisation on raw and standardised features;
4. convert fitted coefficients back into the original units; and
5. recognise leakage, constant-feature and interpretation risks.

## Prerequisites

- basic Python and NumPy arrays;
- means, squared differences and square roots;
- the idea of a linear prediction; and
- a first encounter with gradients and learning rates.

## Scope

### Included

- population-standard-deviation scaling fitted on training data;
- a hand-worked two-feature example;
- multi-feature linear regression implemented with batch gradient descent;
- raw-versus-scaled optimisation comparisons;
- coefficient conversion back to original units; and
- validation and leakage safeguards.

### Deliberately excluded

- robust scaling, min-max scaling and whitening;
- categorical-feature encoding;
- adaptive optimisers;
- production preprocessing pipelines; and
- claims that scaling improves every model family.

## Proposed teaching sequence

1. Frame the units problem with two housing features.
2. Derive standardisation and explain its geometry.
3. Work through a small numerical example.
4. Implement a reusable training-only standardiser.
5. Rebuild multi-feature gradient descent from focused functions.
6. Compare raw and scaled optimisation behaviour.
7. restore the fitted coefficients to original units.
8. Validate predictions, discuss leakage and state limitations.

## Mathematics and notation

Principal relationships are the linear model, mean squared error, its parameter gradients, the feature mean and population standard deviation, the standardisation transform and the inverse coefficient transform. Every scalar, vector and matrix is defined with its shape before use.

## Data and examples

A fixed-seed synthetic housing dataset uses floor area in square metres and age in years to predict price in thousands of pounds. Synthetic data keeps the lesson reproducible and isolates the optimisation issue; it is not evidence about an actual housing market.

## Planned outputs and visuals

- a table comparing feature units and ranges;
- raw and standardised feature-space plots;
- a hand-worked scaling table;
- a log-scale comparison of loss histories;
- a condition-number comparison; and
- an observed-versus-predicted test-set plot.

Each visual receives a prose interpretation and uses labels or marker shapes as well as colour.

## Risks and limitations

The example is synthetic, the learning rates are deliberately chosen to reveal scale sensitivity, and standardisation does not guarantee good predictions or fix poor data. Scaling statistics must be fitted on training data only. Near-constant features require an explicit safeguard.

## Completion criteria

- [ ] Runs from a clean kernel
- [ ] Meets the declared learning outcomes
- [ ] Passes technical and mathematical review
- [ ] Passes educational and accessibility review
- [ ] Includes explicit limitations and next steps
