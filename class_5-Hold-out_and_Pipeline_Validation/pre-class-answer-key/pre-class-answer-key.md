
1. Why should you NOT evaluate a model using the same data it was trained on?

* A) because training takes longer
* B) because it memorizes the data, and the score does not measure generalization (it becomes too high) ✅
* C) because the test set needs to be larger than the training set
* D) because otherwise decision trees cannot be used

Answer: B) because it memorizes the data, and the score does not measure generalization (it becomes too high).

#

2. In a three-set hold-out, what is the role of each set?

* A) training adjusts the model, validation is untouched, and testing selects the model
* B) all three are used for training
* C) training fits the model, validation helps make decisions (model/threshold), and testing provides the final untouched score ✅
* D) validation trains the model and testing adjusts it

Answer: C) training fits the model, validation helps make decisions (model/threshold), and testing provides the final untouched score.

#

3. How do you obtain training / validation / test sets in practice?

* A) two sequential splits: separate the test set first; then split the remainder into training and validation ✅
* B) one split into three equal and always randomly selected parts
* C) it is not possible to have three sets
* D) randomly sampling the rows at each epoch

Answer: A) two sequential splits: separate the test set first; then split the remainder into training and validation.

#

4. What is `stratify=y` used for in `train_test_split`?

* A) to shuffle the data better
* B) to standardize the variables
* C) to speed up training
* D) to maintain the same class proportions in the training and test sets ✅

Answer: D) to maintain the same class proportions in the training and test sets.

#

5. What is `random_state` (the seed) used for?

* A) to improve model accuracy
* B) to make the random split reproducible — same seed, same split ✅
* C) to increase the test set size
* D) to prevent missing values

Answer: B) to make the random split reproducible — same seed, same split.

#

6. "Evaluating with a single train/test split is like flipping a coin." What does this mean?

* A) that the model randomly chooses the answer
* B) that accuracy is always 50%
* C) that the score varies depending on the luck of the split; the solution is to average multiple splits (cross-validation) ✅
* D) that a test set should not be used

Answer: C) that the score varies depending on the luck of the split; the solution is to average multiple splits (cross-validation).

#

7. Standardizing (`StandardScaler`) using the ENTIRE dataset before splitting into training/test sets is:

* A) data leakage — the test set "peeks" at the training data because its mean and standard deviation are included ✅
* B) the recommended approach
* C) irrelevant to the result
* D) necessary for `OneHotEncoder`

Answer: A) data leakage — the test set "peeks" at the training data because its mean and standard deviation are included.

#

8. What is the correct way to apply `StandardScaler`?

* A) `fit_transform` on the training set and `fit_transform` on the test set
* B) `fit` on the test set and `transform` on the training set
* C) `fit` on the training set; `transform` (without fit) on the test set ✅
* D) `fit_transform` on the entire dataset

Answer: C) `fit` on the training set; `transform` (without fit) on the test set.

#

9. What is `ColumnTransformer` used for?

* A) training multiple models at the same time
* B) applying a different transformation to each type of column (numeric × categorical) at once ✅
* C) splitting the training and test sets
* D) calculating accuracy

Answer: B) applying a different transformation to each type of column (numeric × categorical) at once.

#

10. Why does a `Pipeline` (preprocessing + model) prevent leakage during cross-validation?

* A) because it uses less memory
* B) because it removes missing values
* C) because it automatically chooses `k`
* D) because it refits the preprocessing WITHIN each fold, using only that fold's training data ✅

Answer: D) because it refits the preprocessing WITHIN each fold, using only that fold's training data.

#

11. When the data contains dates, what is the correct splitting strategy?

* A) split by time: train on the past and test on the future (do not randomly shuffle) ✅
* B) randomly shuffle the rows as usual
* C) test on the past and train on the future
* D) use only the oldest month

Answer: A) split by time: train on the past and test on the future (do not randomly shuffle).

#

12. You have sales data from 2000 to 2025 and want to forecast December 2026 (you already have January–November 2026). Regarding training:

* A) remove ALL Decembers from 2000–2025 from the training set
* B) train only with 2026 data
* C) keep previous Decembers in the training set (they teach seasonality); never use data after the forecasting point ✅
* D) use only December data for training

Answer: C) keep previous Decembers in the training set (they teach seasonality); never use data after the forecasting point.

#

13. To estimate whether the time-series model works (without having the actual 2026 value yet), what is the best practice?

* A) never test; trust the training results
* B) hold out the most recent known years as a test set (e.g., train on 2000–2023, forecast 2024–2025) and, if approved, retrain using all the data ✅
* C) test on randomly selected years
* D) use the same year for training and testing

Answer: B) hold out the most recent known years as a test set (e.g., train on 2000–2023, forecast 2024–2025) and, if approved, retrain using all the data.

#

14. What is a "global holdout" (product holdout) used for?

* A) tuning the model's parameters
* B) standardizing the variables
* C) replacing the model's test set
* D) keeping a group of users outside ALL changes, measuring the real cumulative impact on the business ✅

Answer: D) keeping a group of users outside ALL changes, measuring the real cumulative impact on the business.

#

15. What is the difference between an A/B test and a global holdout?

* A) an A/B test measures ONE change (short term); a global holdout measures the effect of EVERYTHING (long term) ✅
* B) they are exactly the same thing
* C) an A/B test does not use a control group
* D) a global holdout is used to train the model

Answer: A) an A/B test measures ONE change (short term); a global holdout measures the effect of EVERYTHING (long term).

#

16. In model production, what are skew and drift?

* A) the same problem with different names
* B) typos in the code
* C) skew = training ≠ production NOW (defense: use the same Pipeline); drift = data changes OVER TIME (defense: monitor and retrain) ✅
* D) skew happens over time and drift happens during deployment

Answer: C) skew = training ≠ production NOW (defense: use the same Pipeline); drift = data changes OVER TIME (defense: monitor and retrain).

#

17. Did you have any questions about today's class?

A question I still have is:

In practice, how do we know if the model is actually generalizing well and not just memorizing the training data? And when the data changes over time, how can we tell when it is time to reevaluate or retrain the model?

This is something I would like to understand better in practice, especially how to identify when a model is no longer generalizing well and when retraining becomes necessary.

