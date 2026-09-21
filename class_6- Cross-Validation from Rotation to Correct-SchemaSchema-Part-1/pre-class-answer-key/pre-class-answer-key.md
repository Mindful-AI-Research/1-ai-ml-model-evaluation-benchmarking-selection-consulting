# Answer Key --- Cross-Validation

## 1) "Evaluating with a single train/test split is like flipping a coin." Why?

A)  Because the model answers randomly\
B)  Because accuracy is always 50%\
C)  Because the result can vary depending on the data split; therefore,
    we use multiple splits and calculate an average (cross-validation)
    ✅\
D)  Because we do not need a test set

**Answer:** C

`<br>`{=html}`<br>`{=html}

## 2) In `cross_val_score(modelo, X, y, cv=5)`, what does `cv=5` define?

A)  Five different models\
B)  The division of the data into 5 parts, automatically alternating the
    training and validation sets ✅\
C)  Five training epochs\
D)  Five different metrics

**Answer:** B

`<br>`{=html}`<br>`{=html}

## 3) Which data should be passed to `cross_val_score`?

A)  Only the training/validation block, keeping the final test set
    separate to be used once at the end ✅\
B)  The entire dataset, including the final test set\
C)  Only the test set\
D)  Only a sample of the data

**Answer:** A

`<br>`{=html}`<br>`{=html}

## 4) In a 5-fold cross-validation, the accuracies were `[0.80, 0.90, 0.70, 0.85, 0.75]`. What is the mean?

A)  0.75\
B)  0.80 ✅\
C)  0.85\
D)  0.90

**Answer:** B

`<br>`{=html}`<br>`{=html}

## 5) What is a more honest way to report the cross-validation result?

A)  Only the highest accuracy\
B)  Only the lowest accuracy\
C)  The mean metric together with the standard deviation (mean ±
    standard deviation) ✅\
D)  Only the first fold

**Answer:** C

`<br>`{=html}`<br>`{=html}

## 6) What does `StratifiedKFold` try to maintain in each fold?

A)  The same total number of examples\
B)  The same order of the data\
C)  The same number of features\
D)  Approximately the same proportion of each class ✅

**Answer:** D

`<br>`{=html}`<br>`{=html}

## 7) We have 100 examples, only 3 of them positive, and want to use 5 stratified folds. What happens?

A)  Every fold will have exactly the same number of positive examples\
B)  At least 2 folds will have no positive examples; the number of folds
    should not be greater than the number of examples in the minority
    class ✅\
C)  The model will automatically become balanced\
D)  Accuracy will necessarily be 100%

**Answer:** B

`<br>`{=html}`<br>`{=html}

## 8) What is Leave-One-Out Cross-Validation (LOO)?

A)  A case of k-fold where `k = n`; each example is used alone as the
    test set once ✅\
B)  Validation without a test set\
C)  Validation with only one fold\
D)  A technique that eliminates training

**Answer:** A

`<br>`{=html}`<br>`{=html}

## 9) In LOO with `n=10`, how many training runs are performed?

A)  1\
B)  5\
C)  10; in each round there is only one example in the test set, so the
    average result is more informative ✅\
D)  100

**Answer:** C

`<br>`{=html}`<br>`{=html}

## 10) What does `RepeatedKFold` do?

A)  It always uses exactly the same split\
B)  It repeats k-fold several times using different random splits and
    considers all the results ✅\
C)  It removes the folds with the worst results\
D)  It trains only once

**Answer:** B

`<br>`{=html}`<br>`{=html}

## 11) Why is it a problem to have exams from the same patient in both training and test sets?

A)  Because it increases the number of features\
B)  Because it reduces the amount of data\
C)  Because it prevents the model from learning\
D)  Because the model may memorize characteristics of that patient and
    produce an artificially high score, while failing on new patients ✅

**Answer:** D

`<br>`{=html}`<br>`{=html}

## 12) What does `GroupKFold` guarantee?

A)  All rows belonging to the same group stay on the same side of the
    split, either training or test ✅\
B)  Each group appears in every fold\
C)  Groups are randomly selected within each fold\
D)  The number of features is the same in all groups

**Answer:** A

`<br>`{=html}`<br>`{=html}

## 13) For data that depends on dates, what is the most appropriate strategy?

A)  Train on past data and test on future data, avoiding random
    shuffling ✅\
B)  Shuffle all the data before splitting\
C)  Use only the most recent data for training\
D)  Mix past and future data in every fold

**Answer:** A

`<br>`{=html}`<br>`{=html}

## 14) If we train with data from 2000--2015 and use 2016--2021 as a temporal holdout, what does this represent?

A)  A validation error\
B)  An example of LOO\
C)  A single temporal split; `TimeSeriesSplit` allows us to repeat
    chronological splits to obtain a more stable evaluation ✅\
D)  An example of `GroupKFold`

**Answer:** C

`<br>`{=html}`<br>`{=html}

## 15) Why should preprocessing be placed inside a `Pipeline`?

A)  To make the code longer\
B)  To avoid training the model\
C)  To use more features\
D)  So that preprocessing is fitted again in each fold using only that
    fold's training data, preventing leakage ✅

**Answer:** D

`<br>`{=html}`<br>`{=html}

## 16) What happens when we use `StandardScaler` on the entire dataset before cross-validation?

A)  Data leakage may occur because the mean and standard deviation are
    calculated using information from portions that should act as the
    test set in each fold, potentially inflating the score ✅\
B)  The model automatically becomes more robust\
C)  Cross-validation stops existing\
D)  The test set disappears

**Answer:** A

`<br>`{=html}`<br>`{=html}

## 17) Open-ended question

**One question I still have is: in practice, how do we know if the model
is actually generalizing well and not just memorizing the training data?
And when the data changes over time, how can we tell when it is time to
reevaluate or retrain the model?**
