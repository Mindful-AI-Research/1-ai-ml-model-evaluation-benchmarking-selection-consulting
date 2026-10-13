# Answer Key --- CD1 \| Lesson 10: Validation Curves

**Lesson 10 Quiz --- correct answers with explanations**

  ------------------------------------------------------------------------------------
                 Question        Correct answer         Explanation
  ----------------------- ----------------------------- ------------------------------
                        1             **B**             The x-axis shows the values of
                                                        a single hyperparameter, while
                                                        keeping the other model
                                                        settings constant.

                        2             **B**             A validation curve isolates
                                                        the effect of one
                                                        hyperparameter. In a grid
                                                        search with multiple
                                                        dimensions, the effects can
                                                        become mixed together.

                        3             **C**             The training score generally
                                                        increases with model
                                                        complexity. Choosing based
                                                        only on training performance
                                                        would favor increasingly
                                                        complex models, even when they
                                                        overfit.

                        4             **A**             The depth-1 tree has a
                                                        validation accuracy of 0.839,
                                                        below the baseline of 0.849,
                                                        indicating underfitting.

                        5             **B**             The training score rises from
                                                        0.943 to 1.000 while
                                                        validation accuracy falls from
                                                        0.888 to 0.875. The gap grows
                                                        from 0.055 to 0.125,
                                                        indicating overfitting.

                        6             **C**             The gap is the difference
                                                        between the training score and
                                                        validation score for the same
                                                        hyperparameter value.

                        7             **A**             Choose the value that
                                                        maximizes validation
                                                        performance. When results are
                                                        similar within the observed
                                                        variation, prefer the simpler
                                                        model.

                        8             **B**             The difference between the
                                                        means is small (0.002), within
                                                        the reported standard
                                                        deviation. Depth 2 is simpler
                                                        and has a smaller gap (0.018
                                                        versus 0.055).

                        9             **C**             In a `Pipeline`, use
                                                        `step_name__parameter_name`;
                                                        in this case,
                                                        `clf__max_depth`.

                       10             **B**             `validation_curve` returns two
                                                        matrices, with one row per
                                                        tested value and one column
                                                        per cross-validation fold.
                                                        `mean(axis=1)` calculates the
                                                        mean across folds for each
                                                        value.

                       11             **D**             The `Pipeline` fits
                                                        preprocessing within each
                                                        training fold, preventing
                                                        validation-fold data from
                                                        influencing preprocessing and
                                                        making the evaluation overly
                                                        optimistic.

                       12             **A**             Since `C` spans several orders
                                                        of magnitude, a logarithmic
                                                        scale makes both small and
                                                        large values easier to
                                                        visualize.

                       13             **B**             With a very small `C`,
                                                        regularization is extremely
                                                        strong and the coefficients
                                                        approach zero. The model
                                                        identifies virtually no
                                                        positive cases and matches the
                                                        baseline.

                       14             **C**             Logistic regression is a
                                                        linear model and, in this
                                                        curve, shows no signs of
                                                        overfitting across the
                                                        observed range. Not every
                                                        curve displays all three
                                                        classic regions.

                       15             **D**             In a dataset with 15.1%
                                                        positive cases, always
                                                        predicting the negative class
                                                        yields 0.849 accuracy but an
                                                        F1 score of zero. Accuracy can
                                                        hide underfitting.

                       16             **A**             First, use validation curves
                                                        to identify promising ranges;
                                                        then use a search to select
                                                        the hyperparameter
                                                        combination. Final evaluation
                                                        should use nested
                                                        cross-validation or a held-out
                                                        test set used only once.
  ------------------------------------------------------------------------------------

## Answer summary

**1-B · 2-B · 3-C · 4-A · 5-B · 6-C · 7-A · 8-B · 9-C · 10-B · 11-D ·
12-A · 13-B · 14-C · 15-D · 16-A**


