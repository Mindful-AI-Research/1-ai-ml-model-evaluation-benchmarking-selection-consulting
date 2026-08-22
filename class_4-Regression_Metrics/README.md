# Regression Metrics — Class 04

> Consultoria Especializada em Ciência de Dados 1 · 25 August 2026

## Table of Contents

- [Overview](#overview)
- [Learning Objectives](#learning-objectives)
- [Key Concepts](#key-concepts)
- [Theory and Mathematics](#theory-and-mathematics)
- [Worked Example](#worked-example)
- [Metric Selection](#metric-selection)
- [Residual Diagnostics](#residual-diagnostics)
- [Limitations](#limitations)
- [Repository Scope](#repository-scope)
- [References](#references)

## Overview

This repository documents Class 04, **Regression Metrics**, from the supplied lecture material. The class moves from classification, where predictions are commonly treated as correct or incorrect, to regression, where the central question is how far a numeric prediction is from the observed value. The lecture uses delivery time as its running example, while noting that the same ideas apply to price, demand, and temperature.

No source code, notebook, dataset, executable experiment, or external URL was provided in the supplied class materials.

## Learning Objectives

By the end of the class, a learner should be able to:

- Define regression error as the observed value minus the predicted value.
- Interpret the sign of an error as underestimation or overestimation.
- Explain and compare MAE and RMSE.
- Interpret R² relative to a baseline that always predicts the target mean.
- Explain the MAPE percentage interpretation and its small-denominator problem.
- Inspect residual plots for random dispersion, increasing variance, systematic curves, and bias.
- Select metrics according to the practical cost of prediction errors.

## Key Concepts

### Regression output

A regression model returns a continuous number, such as an estimated delivery time of 30 minutes. Exact predictions are uncommon; evaluation therefore measures distance from the real value rather than classification accuracy.

### Error and residual

The lecture defines the error as:

\[e_i = y_i - \hat{y}_i\]

A positive error means the real value exceeded the prediction: the model underestimated. A negative error means the real value was below the prediction: the model overestimated.

### Baseline

For R², the baseline is the simple strategy of predicting the same value—the target mean—for every observation.

## Theory and Mathematics

### Mean Absolute Error (MAE)

\[\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|\]

MAE is expressed in the target unit and provides a direct interpretation such as “the model is typically off by 12 minutes.” It does not disproportionately amplify a single large error.

### Root Mean Squared Error (RMSE)

\[\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}\]

Squaring makes large errors weigh more heavily; the square root returns the result to the target unit. For the same data, RMSE is at least MAE, and a large gap between them indicates unusually large errors.

### Coefficient of determination (R²)

\[R^2 = 1 - \frac{\sum_i(y_i-\hat{y}_i)^2}{\sum_i(y_i-\bar{y})^2}\]

Interpretation in the lecture:

- `1.0`: perfect predictions.
- `0`: equivalent to always predicting the mean.
- Negative: worse than the mean-prediction baseline.

R² is unitless and must be reported with an error metric in the target unit.

### Mean Absolute Percentage Error (MAPE)

\[\mathrm{MAPE} = \frac{1}{n}\sum_{i=1}^{n}\left|\frac{y_i-\hat{y}_i}{y_i}\right|\times100\%\]

MAPE communicates average relative error as a percentage. However, because it divides by the real value, it can become extremely large when real values are close to zero.

## Worked Example

The supplied slides provide five delivery-time observations:

| Order | Predicted | Real | Error |
|---|---:|---:|---:|
| #1 | 30 min | 35 min | +5 min |
| #2 | 25 min | 22 min | −3 min |
| #3 | 40 min | 38 min | −2 min |
| #4 | 20 min | 26 min | +6 min |
| #5 | 30 min | 74 min | +44 min (motorcycle broke) |

The lecture reports:

- MAE = `(5 + 3 + 2 + 6 + 44) / 5` = **12 minutes**.
- RMSE ≈ **20 minutes**, strongly affected by the 44-minute error.
- Mean real delivery time = **39 minutes**.
- Sum of squared model errors = **2,010**.
- Sum of squared deviations from the mean = **1,700**.
- R² = `1 − 2010 / 1700` = **−0.18**, meaning the model is worse than always predicting 39 minutes for these observations.

These are demonstrated values from the lecture example, not results from a supplied implementation or dataset.

## Metric Selection

| Practical question | Metric indicated | Why |
|---|---|---|
| What is the typical error in minutes? | MAE | Direct and easy to explain. |
| Should very large delays be penalized strongly? | RMSE | Squaring magnifies large errors. |
| Is the model better than predicting the mean? | R² | Compares model error with a mean baseline. |
| What is the average relative error? | MAPE | Reports percentage error, provided real values are not near zero. |

The lecture recommends reporting a pair in most cases: MAE or RMSE plus R².

## Residual Diagnostics

The residual plot is a diagnostic, not merely a visualization:

- A pattern-free cloud around zero is a good sign.
- A funnel indicates that error variance increases with the predicted value.
- A curve indicates systematic non-linearity or an overly simple model.
- Residuals concentrated on one side indicate bias, such as consistent underestimation or overestimation.

## Limitations

The class explicitly warns against:

- Reporting only R² without an error in meaningful units.
- Using MAPE when target values are near zero.
- Ignoring residual plots.
- Comparing MAE or RMSE across targets with different scales.
- Treating R² as directly comparable across problems with different variance.

The supplied materials do not provide a dataset, code, notebook, train/validation split, fitted model, scikit-learn implementation, or independently generated test results.

## Repository Scope

### Implemented

Not provided in the supplied class materials.

### Demonstrated

- Hand-calculated errors for five delivery orders.
- Hand-calculated MAE, RMSE interpretation, and R².
- Conceptual interpretation of MAPE.
- Residual-plot patterns.

### Conceptual

- Regression metric definitions and formulas.
- Mean-prediction baseline.
- Metric selection by error cost.

### Recommended

For a future implementation, evaluate on validation data, reserve the test set for final assessment, report a target-unit metric with R², and inspect residuals. These are engineering recommendations derived from the lecture’s good-practice guidance, not existing repository features.

### Future Extension

A future exercise could implement the formulas in Python and reproduce the five-order example. No such implementation was supplied.

## References

- Supplied source material: `CD1_Aula04_slides_vfinal.pdf`, Class 04, “Métricas de regressão,” dated 25/08/2026.

No external references or URLs were included in the supplied materials.

## Conclusion

A regression evaluation should answer both “how far off is the model?” and “is it better than a simple baseline?” MAE or RMSE supplies the error in meaningful units, R² supplies baseline-relative context, MAPE supplies a percentage only when its denominator is safe, and residual inspection reveals failure patterns that a single score can hide.
