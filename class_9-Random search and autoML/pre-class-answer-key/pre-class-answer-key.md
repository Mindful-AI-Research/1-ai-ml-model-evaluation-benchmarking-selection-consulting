# Answer Key — Random Search, Optimization & AutoML

## 1. Why does the computational cost of the grid motivate random search?

**Correct answer:**  
> **because it would require 6,400 model fits: each new hyperparameter MULTIPLIES the cost**

**Explanation:**  
There are 1,280 combinations × 5 cross-validation folds = **6,400 model fits**. Each additional hyperparameter can multiply the number of combinations.

<br><br>

## 2. With the SAME budget of 16 measurements, why does random search usually outperform a grid?

**Correct answer:**  
> **because random points cover larger areas of the search space than aligned grid points**

**Explanation:**  
Random search tends to explore the hyperparameter space more efficiently, especially when only a few hyperparameters have a strong impact on performance.

<br><br>

## 3. What did repeating the search with 100 different seeds show?

**Correct answer:**  
> **all 100 were above the grid (mean 0.640), and seed 0 was actually below the average**

**Explanation:**  
This shows that the improvement from random search was not simply the result of one particularly favorable random seed.

<br><br>

## 4. According to $1-(1-p)^n$, what is the probability of hitting the good region?

**Correct answer:**  
> **about 93% — and the number of hyperparameters does not enter the calculation**

**Calculation:**

$$
1-(1-p)^n
$$

With $p=0.15$ and $n=16$:

$$
1-(1-0.15)^{16}
=
1-0.85^{16}
\approx 0.926
$$

Therefore, the probability is approximately **92.6%**.

<br><br>

## 5. What does `uniform(0.1, 0.8)` mean?

**Correct answer:**  
> **0.1 and 0.9 — the second argument is the WIDTH, added to the starting point**

**Explanation:**  
In `scipy.stats.uniform(loc, scale)`, the first argument is the starting point (`loc`) and the second is the width (`scale`).

Therefore:

$$
0.1+0.8=0.9
$$

The distribution samples values in the interval **[0.1, 0.9)**.

<br><br>

## 6. Why use `loguniform` for the logistic regression `C` parameter?

**Correct answer:**  
> **because with uniform, about 90% of the samples would fall between 10 and 100, whereas loguniform gives 25% to each order of magnitude**

**Explanation:**  
For parameters spanning several orders of magnitude, such as $C$ from 0.01 to 100, a logarithmic distribution provides a more balanced exploration.

<br><br>

## 7. What does it suggest when the best `ccp_alpha` is exactly at the LOWER boundary of the defined range?

**Correct answer:**  
> **the peak may be OUTSIDE the range: it is worth expanding the range and running the search again**

**Explanation:**  
When the best value appears at the boundary of the search distribution, it may indicate that the chosen range does not contain the true optimum.

<br><br>

## 8. How many model fits does `RandomizedSearchCV` perform with `n_iter=40` and `cv=5`?

**Correct answer:**  
> **200 — n_iter × folds, regardless of how many hyperparameters exist**

**Calculation:**

$$
40\times5=200
$$

That means **200 model fits**, excluding the final `refit`.

<br><br>

## 9. How should the `cv_results_` be interpreted?

**Correct answer:**  
> **it is a plateau: the difference is within the noise, so it is worth preferring the simpler or more stable configuration**

**Explanation:**  
A difference of only 0.014 between candidates, when fold-to-fold variability is approximately ±0.11, does not provide strong evidence that the first candidate is genuinely better.

<br><br>

## 10. What is the "coarse-to-fine" strategy?

**Correct answer:**  
> **start with a broad random search using log-scaled distributions, then use a small grid around the best points — stopping when the numbers become a plateau**

**Explanation:**  
First, explore the search space broadly to locate promising regions. Then, concentrate the search around those regions. Stop refining when additional tuning no longer produces meaningful improvements.

<br><br>

## 11. Which number should NOT be reported as model performance?

**Correct answer:**  
> **the best_score_, because it is the MAXIMUM of 40 measurements made on the same validation process that selected the model**

**Explanation:**  
`best_score_` is used to select the best candidate during optimization. Because it is the maximum observed across multiple trials, it can be **optimistic** as an estimate of generalization performance.

Nested cross-validation provides a more appropriate evaluation of the entire selection process, while a **held-out test set** can provide an independent final evaluation when it has genuinely remained outside the selection process.

<br><br>

## 12. What does Bayesian optimization do that grid and random search do not?

**Correct answer:**  
> **it uses the measurements already made to fit a surrogate model and decide where to measure next**

**Explanation:**  
Instead of simply sampling new points independently, Bayesian optimization uses previous results to guide the selection of future evaluations.

<br><br>

## 13. Why is a list of search spaces in the mini-AutoML a conditional search space?

**Correct answer:**  
> **because the space is conditional: `C` only exists if the search selects logistic regression, while `ccp_alpha` only exists if it selects the tree-based model**

**Explanation:**  
Each model has its own relevant hyperparameters. Therefore, which hyperparameters are applicable depends on which model family is selected.

<br><br>

## 14. What do the mini-AutoML results teach us?

**Correct answer:**  
> **choosing the MODEL mattered much more (0.31 difference) than fine-tuning the winner (0.005 between the best score and the median)**

**Explanation:**

Logistic regression:

$$
0.721
$$

Decision tree:

$$
0.413
$$

Difference:

$$
0.721-0.413=0.308
$$

Meanwhile, the difference between the best logistic regression result and its median is:

$$
0.721-0.716=0.005
$$

Therefore, **choosing the model family had a much larger impact than fine-tuning the hyperparameters within the winning model**.

<br><br>

## 15. What happens when seven more models are added while keeping `n_iter=30`?

**Correct answer:**  
> **each model receives roughly 3 searches: too few to adequately cover its search space, and the $1-(1-p)^n$ calculation no longer works in its favor**

**Explanation:**  
With more models competing for the same budget of 30 random searches, each model receives relatively few opportunities for exploration.

This reduces the ability to properly explore the hyperparameter space of each model family.

<br><br>

## 16. What does the increase in AP from 0.721 to 0.936 after including popularity tell us?

**Correct answer:**  
> **the AutoML system would happily optimize a feature that leaks the target: deciding what belongs in the model is still your responsibility**

**Explanation:**  
AutoML optimizes the metric it is given. It does not automatically know that a particular feature represents **target leakage**.

Therefore, feature selection and deciding which information is valid for the model remain the researcher's responsibility.

<br><br>

# Open Question

## Did anything from today's class remain unclear?

**Answer:**  
> **No major questions remained. I understood that random search explores relevant hyperparameters more effectively under a limited budget, that `loguniform` is useful for ranges spanning several orders of magnitude, and that `best_score_` can be optimistic, making evaluation with nested CV or a held-out test set important.**
