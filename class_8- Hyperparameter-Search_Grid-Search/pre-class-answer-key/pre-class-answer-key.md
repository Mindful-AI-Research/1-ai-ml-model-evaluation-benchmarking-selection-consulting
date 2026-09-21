
# Answer Key — Hyperparameter Tuning and GridSearchCV

<br><br>

## 1) Which of these is a HYPERPARAMETER (not a parameter)?

A) The weights learned by logistic regression

B) **The maximum depth of a decision tree ✅**

C) The split points that the tree chooses at each node

D) The estimated slope coefficient in linear regression

**Answer:** B

<br><br>

## 2) The golden rule of hyperparameter tuning is:

A) Training learns, testing chooses, validation confirms

B) Choose using the test set and confirm with validation, since it is larger

C) Use the same dataset for choosing and measuring, as long as CV has many folds

D) **Training learns, VALIDATION chooses, TEST confirms — once only ✅**

**Answer:** D

<br><br>

## 3) GridSearchCV is called an EXHAUSTIVE search because:

A) **It tests the entire grid without leaving any combination out ✅**

B) It runs until performance stops improving

C) It exhausts the machine's memory on large grids

D) It randomly samples combinations until it covers most of the search space

**Answer:** A

<br><br>

## 4) In the cake grid (3 times × 3 temperatures), how many combinations are tested and which one won?

A) 6 combinations; 30 min × 180 °C won

B) 3 combinations; 50 min × 200 °C won

C) **9 combinations; 40 min × 180 °C won, with a score of 9.0 ✅**

D) 9 combinations; 40 min × 200 °C won, with a score of 7.0

**Answer:** C

<br><br>

## 5) With the grid `C ∈ {0.1, 1, 10}` × `class_weight ∈ {None, balanced}` and 5-fold CV, how many models are TRAINED?

A) 6, one for each combination

B) 5, one for each fold

C) 11, adding the options and the folds

D) **30 — 3 × 2 = 6 combinations, each evaluated across 5 folds ✅**

**Answer:** D

<br><br>

## 6) Adding a third hyperparameter with 4 options to the same grid, the number of training runs increases from 30 to:

A) 34 — the new axis is added

B) **120 — each new axis MULTIPLIES the total ✅**

C) 38 — the 4 options are added to the 5 folds

D) 30 — the number of folds does not change

**Answer:** B

<br><br>

## 7) The class table gave these CV mean AP scores: C=0.1/None 0.61 · C=0.1/balanced 0.68 · C=1/None 0.71 · C=1/balanced 0.79 · C=10/None 0.70 · C=10/balanced 0.74. What does GridSearchCV store in `best_params_` and `best_score_`?

A) **`best_params_ = {C: 1, class_weight: balanced}` and `best_score_ = 0.79` ✅**

B) `best_params_ = {C: 10, class_weight: balanced}` and `best_score_ = 0.74`

C) `best_params_ = {C: 1, class_weight: balanced}` and `best_score_ = 0.71`

D) `best_params_ = the average of the six combinations, 0.705`

**Answer:** A

<br><br>

## 8) When searching over a Pipeline, why is the grid written as `{"clf__C": [0.1, 1, 10]}` instead of `{"C": [0.1, 1, 10]}`?

A) Because the double underscore indicates that the value is a list

B) Because `"clf"` is the name of the scoring metric

C) **Because the `step_name__parameter` prefix tells us WHICH Pipeline step the hyperparameter belongs to ✅**

D) Because sklearn requires prefixes in every search, with or without a Pipeline

**Answer:** C

<br><br>

## 9) In a problem with a rare class, setting `scoring="accuracy"` during the search is a problem because:

A) Accuracy is not accepted by GridSearchCV for binary problems

B) **Always predicting the majority class can already produce high accuracy — the search could choose a model that ignores the rare class ✅**

C) Accuracy can only be used with stratified cross-validation

D) Accuracy is always lower than F1, so it underestimates the model

**Answer:** B

<br><br>

## 10) In classification, the recommended CV strategy for the search is StratifiedKFold because it:

A) Is faster than regular KFold

B) Shuffles the data, unlike KFold

C) Uses all folds for training and none for validation

D) **Preserves the class proportions in each fold — otherwise, the rare class could disappear from one of the folds ✅**

**Answer:** D

<br><br>

## 11) Why should GridSearchCV receive the entire PIPELINE instead of only the classifier with already preprocessed data?

A) Because the Pipeline makes the search faster

B) Because only a Pipeline accepts the `scoring` parameter

C) **Because preprocessing is refitted inside each fold using only that fold's training data — preprocessing beforehand leaks information from the validation set ✅**

D) Because sklearn cannot calculate `best_score_` outside a Pipeline

**Answer:** C

<br><br>

## 12) The search ended with `best_score_ = 0.79`, and the test set, evaluated once with `best_estimator_`, gave 0.76. Which number should you report?

A) **0.76 — the test set is the final evaluation; `best_score_` was used for selection, not final reporting ✅**

B) 0.79 — it is the best estimate obtained through cross-validation across several folds

C) The average of the two, 0.775

D) 0.79, mentioning that the test was only a verification

**Answer:** A

<br><br>

## 13) Why does `best_score_` tend to be slightly OPTIMISTIC?

A) Because cross-validation always overestimates performance

B) Because it is calculated on the training set

C) Because sklearn rounds the value upward

D) **Because it is the MAXIMUM of many noisy measurements — the combination that ended up on top may have benefited partly from luck in the folds ✅**

**Answer:** D

<br><br>

## 14) Looking at `cv_results_`, three combinations are tied within the standard deviation. What should you do?

A) Increase the number of folds until one stands out

B) **Recognize the plateau and choose the SIMPLER combination among the tied ones ✅**

C) Choose the one with the highest mean, even if the difference is within the standard deviation

D) Discard all three and expand the grid around them

**Answer:** B

<br><br>

## 15) Regarding `search.best_estimator_`, which statement is CORRECT?

A) **It is the Pipeline with the best combination, ALREADY TRAINED — this is the model you take to the test set ✅**

B) It is only a dictionary containing the best values found

C) It is the model with the correct hyperparameters, but it has not been trained yet: you need to call `fit` before using it

D) It is the complete table with the mean and standard deviation for each combination

**Answer:** A

<br><br>

## 16) The test score was MUCH lower than the `best_score_` from validation. The most likely interpretation is:

A) The test set is too small and should be expanded with validation data

B) The model is underfitting and the grid needs more extreme values

C) **The search overfit — the grid was too large for the amount of data, and the top result was partly due to luck in the folds ✅**

D) There was a coding error: the two numbers must be identical

**Answer:** C

<br><br>

## 17) Was anything unclear ?

**One question I still have is: in practice, how can we tell when a hyperparameter search is starting to “overfit” the validation data? And when several combinations are very close, how do we decide whether it makes sense to choose the simpler one?**
