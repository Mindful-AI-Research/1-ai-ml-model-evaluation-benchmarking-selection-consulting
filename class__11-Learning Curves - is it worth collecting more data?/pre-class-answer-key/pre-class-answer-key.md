# CD1 — Lesson 11: Learning Curves

## Answer Key

This document records the answer key previously provided in the conversation. Answer choices follow the order stated earlier and have not been rechecked against the form.

| Question | Answer choice | Answer |
| --- | --- | --- |
| 1 | 2nd | A validation curve varies a hyperparameter; a learning curve varies the training set size. |
| 2 | 3rd | The training score decreases slightly, the validation score increases, and the gap between them narrows. |
| 3 | 1st | The validation curve, because it estimates performance on unseen data. |
| 4 | 4th | High bias (underfitting): the model lacks sufficient capacity. |
| 5 | 1st | High variance (overfitting): more data or regularization should help. |
| 6 | 2nd | The gap between training and validation narrows while validation performance continues to improve: collecting more data is worthwhile. |
| 7 | 3rd | Do not prioritize more data: the curves are close together and have plateaued at a low performance level, indicating high bias. |
| 8 | 4th | Saturation: increasing the amount of data produces progressively smaller gains. |
| 9 | 1st | There are 40 training runs: 8 training sizes × 5 folds, compared with 5 runs in cross_val_score. |
| 10 | 2nd | Prevent data leakage: fit preprocessing only on the training portion of each fold and training size. |
| 11 | 3rd | The first point uses very few examples, making the result less reliable. |
| 12 | 4th | The means alone do not establish an improvement, given the substantial overlap between the bands. |
| 13 | 1st | More folds leave a larger fraction of the data available for each training run. |
| 14 | 2nd | Use a more expressive model, add features, or reduce regularization. |
| 15 | 3rd | Examine the final slope of the validation curve: if it is still rising, more data may help; if it has plateaued, the expected gain is small. |
| 16 | 4th | Request more data when the curves are close together at a low performance level, indicating high bias. |

## Note on Question 16

The entry above reproduces the previous answer key. Without the full question, it is not possible to determine whether it asks for a correct action or the identification of a mistake.

## Suggested Question for the Feedback Field

> How can we determine whether an improvement in validation performance as we add more data is real or merely variation across cross-validation folds, especially when the standard-deviation bands overlap?

## Source

Translated from the previous response in this conversation concerning the form “CD1 — Aula 11: Curvas de aprendizado” (CD1 — Lesson 11: Learning Curves). This is not an official answer key.
