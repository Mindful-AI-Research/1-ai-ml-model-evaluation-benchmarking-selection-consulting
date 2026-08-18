

# ⚠️ Beware of AUC on Imbalanced Datasets

### ROC-AUC, Precision-Recall, AP, Log-Loss, and Confusion Matrix

When evaluating a classification model, it is important not to rely on a single metric. In problems where the **positive class is rare**, such as spam detection, one metric may give a more favorable impression than the model's actual performance.

Imagine a dataset with **1,000 emails**:

- **980 legitimate emails**
- **20 spam emails**

In this scenario:

$$
Prevalence = \frac{20}{1000} = 2\%
$$

This distribution is known as **class imbalance**.

<br><br>

## 1. Why Can Accuracy Be Misleading?

Imagine a model that simply classifies **all 1,000 emails as "not spam."**

We would have:

| | Predicted: Spam | Predicted: Not Spam |
|---|---:|---:|
| **Actual: Spam** | TP = 0 | FN = 20 |
| **Actual: Not Spam** | FP = 0 | TN = 980 |

Accuracy would be:

$$
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
$$

$$
Accuracy = \frac{0+980}{1000}=98\%
$$

**98% accuracy looks excellent.**

But the model did not identify **a single spam email**.

Its recall for the spam class would be:

$$
Recall = \frac{TP}{TP+FN}
$$

$$
Recall = \frac{0}{0+20}=0\%
$$

In other words:

> **98% accuracy, but 0% recall for spam.**

This example shows why, with imbalanced datasets, accuracy should not be analyzed in isolation.

<br><br>

# 2. What Does ROC-AUC Actually Measure?

The **ROC (Receiver Operating Characteristic)** curve shows how the model behaves as we vary the decision threshold.

It relates:

$$
TPR = \frac{TP}{TP+FN}
$$

to:

$$
FPR = \frac{FP}{FP+TN}
$$

The **ROC-AUC** is the area under this curve.

One important interpretation of AUC is its ability to evaluate the model's **discrimination or ranking capability**.

Imagine that the model assigns a probability of being spam to each email:

| Email | Actual Class | Spam Probability |
|---|---|---:|
| A | Spam | **0.99** |
| B | Spam | **0.95** |
| C | Spam | **0.90** |
| D | Spam | **0.85** |
| E | Not Spam | **0.30** |
| F | Not Spam | **0.20** |

The model is producing a good **ranking**:

$$
Spam > Not\ Spam
$$

The examples that are actually spam receive higher scores.

This ability to separate and rank the classes is one of the things captured by ROC-AUC.

<br><br>

# 3. But Why Be Careful with ROC-AUC?

Now imagine that there are **very few spam emails**.

For example:

- 990 legitimate emails
- 10 spam emails

The model can still achieve a **high ROC-AUC** because it may rank the spam emails above a large proportion of legitimate emails.

For example:

$$
ROC\text{-}AUC \approx 0.90
$$

This indicates **good discrimination**.

However, it does not necessarily answer the most important practical question:

> **When I actually block emails as spam, how many of the blocked emails will really be spam?**

That is a question related to **Precision**.

<br><br>

# 4. Precision-Recall: Focusing on Spam

The **Precision-Recall (PR)** curve can be especially useful when the positive class is rare.

It relates:

### Precision

Among the emails classified as spam, how many are actually spam?

$$
Precision = \frac{TP}{TP+FP}
$$

### Recall

Among all the spam emails that actually exist, how many did the model find?

$$
Recall = \frac{TP}{TP+FN}
$$

Therefore:

> **Precision → quality of the alerts**

> **Recall → coverage of the spam emails**

The PR curve is constructed by **varying the threshold**.

<br><br>

# 5. What Happens When We Change the Threshold?

Imagine that the model assigns probabilities to the emails.

If we choose:

$$
Threshold = 0.50
$$

any email with a spam probability ≥ 0.50 will be classified as spam.

But we can change this value.

### Lower Threshold

$$
Threshold = 0.30
$$

More emails will be classified as spam.

This can increase:

$$
Recall
$$

because we can detect more spam.

However, it can also increase:

$$
FP
$$

and consequently reduce:

$$
Precision
$$

### Higher Threshold

$$
Threshold = 0.80
$$

The model becomes more selective.

Precision may increase, but some spam emails may be missed, increasing:

$$
FN
$$

and reducing Recall.

<br><br>

# 6. Practical Example

Imagine again:

**1,000 emails**

- 990 not spam
- 10 spam

Now suppose that, at a particular threshold, the model produces:

- **TP = 9**
- **FN = 1**
- **FP = 90**
- **TN = 900**

Recall is:

$$
Recall = \frac{9}{9+1}=90\%
$$

This looks excellent.

But Precision is:

$$
Precision = \frac{9}{9+90}
$$

$$
Precision \approx 9.1\%
$$

In other words:

> The model found **90% of the spam**, but only approximately **9% of the emails classified as spam were actually spam**.

This is extremely important in a real-world system.

Imagine that those 90 false positives are legitimate and important emails.

<br><br>

# 7. AP — Average Precision

The PR curve contains multiple points because we are varying the threshold.

**AP (Average Precision)** summarizes the behavior of the Precision-Recall curve in a single value.

One way to express it is:

$$
AP = \sum_n (R_n-R_{n-1})P_n
$$

where:

- $P_n$ = Precision at point $n$
- $R_n$ = Recall at point $n$

A higher AP generally indicates better performance in the relationship between **Precision and Recall**.

<br><br>

# 8. The Precision-Recall Baseline

The Precision-Recall baseline corresponds approximately to the **prevalence of the positive class**:

$$
Baseline = \frac{P}{P+N}
$$

If we have:

- 200 positive examples
- 800 negative examples

then:

$$
Baseline = \frac{200}{1000}=0.20
$$

Therefore:

$$
Baseline = 20\%
$$

In the spam example:

- 10 spam emails
- 990 legitimate emails

we have:

$$
Baseline = \frac{10}{1000}=1\%
$$

Therefore, the spam class represents only **1% of the dataset**.

This helps put the performance of the PR curve into context.

<br><br>

# 9. Confusion Matrix

The confusion matrix shows what happened **after choosing a specific threshold**.

| | **Predicted: Spam (1)** | **Predicted: Not Spam (0)** |
|---|---:|---:|
| **Actual: Spam (1)** | **TP** | **FN** |
| **Actual: Not Spam (0)** | **FP** | **TN** |

### TP — True Positive

It was spam and the model predicted spam.

### FN — False Negative

It was spam, but the model predicted not spam.

### FP — False Positive

It was a legitimate email, but the model predicted spam.

### TN — True Negative

It was a legitimate email and the model predicted not spam.

<br><br>

# 10. Diagonal = Correct Predictions

In the confusion matrix:

$$
TP + TN
$$

represent the **correct predictions**.

While:

$$
FP + FN
$$

represent the **errors**.

In the spam case:

- **FP:** a legitimate email is blocked.
- **FN:** a spam email reaches the inbox.

Depending on the application, one type of error may be more costly than the other.

<br><br>

# 11. Log-Loss — When Confident Errors Cost More

While ROC-AUC and PR mainly evaluate **ranking and classification performance**, **Log-Loss** evaluates the quality of the **predicted probabilities**.

The central idea is:

> **It is not enough to be correct. The model should also be appropriately confident in its predictions.**

Log-Loss strongly penalizes the model when it **makes a highly confident mistake**.

For a single observation:

$$
LogLoss = -\left[y\log(p)+(1-y)\log(1-p)\right]
$$

where:

- $y$ = true class, with $1$ for spam and $0$ for not spam
- $p$ = probability assigned by the model to the spam class

### If the email was spam

In this case:

$$
y=1
$$

The formula becomes:

$$
LogLoss=-\log(p)
$$

### Example 1 — Confident Correct Prediction

The email was spam and the model assigned:

$$
p=0.90
$$

Then:

$$
LogLoss=-\log(0.90)\approx0.11
$$

**Small penalty → good result.**

### Example 2 — Confident Error

The email was spam, but the model assigned:

$$
p=0.10
$$

Then:

$$
LogLoss=-\log(0.10)\approx2.30
$$

**Much larger penalty → poor result.**

The difference is important:

| Situation | $p$ for Spam | Log-Loss |
|---|---:|---:|
| Spam, very confident prediction | 0.99 | ≈ 0.01 |
| Spam, correct prediction | 0.90 | ≈ 0.11 |
| Spam, uncertain prediction | 0.50 | ≈ 0.69 |
| Spam, incorrect prediction | 0.10 | ≈ 2.30 |
| Spam, very confident incorrect prediction | 0.01 | ≈ 4.61 |

The lower the Log-Loss:

$$
\boxed{Lower\ Log\text{-}Loss = Better}
$$

### 📌 What Does Log-Loss Measure?

We can think of it this way:

> **ROC-AUC:** Did the model rank the cases well?

> **PR/AP:** Can the model find the rare positive class while maintaining a good Precision-Recall trade-off?

> **Log-Loss:** Are the probabilities produced by the model reliable, or is it making overly confident mistakes?

<br><br>

# 12. Why Is Log-Loss Important?

Imagine two models:

### Model A

Says:

> “This email has a **51% chance of being spam**.”

And it was spam.

### Model B

Says:

> “This email has a **99% chance of being spam**.”

And it was also spam.

Both models predicted the correct class, but Model B was much more confident.

Now imagine the opposite:

### Model C

Says:

> “This email has a **1% chance of being spam**.”

But it was spam.

The error is much more severe because the model was **extremely confident in the wrong prediction**.

This is exactly the type of behavior that Log-Loss strongly penalizes.

<br><br>

# 13. Log-Loss Across Multiple Observations

For a dataset of $N$ observations, the mean Log-Loss is:

$$
LogLoss =
-\frac{1}{N}
\sum_{i=1}^{N}
\left[
y_i\log(p_i)
+
(1-y_i)\log(1-p_i)
\right]
$$

The lower the value:

$$
\boxed{Lower = Better}
$$

A model that provides well-calibrated probabilities tends to achieve a lower Log-Loss.

<br><br>

# 14. A Simple Comparison of the Metrics

| Metric | Main Question |
|---|---|
| **Accuracy** | How many cases were classified correctly? |
| **ROC-AUC** | Can the model separate and rank the classes well? |
| **Precision** | Of the cases classified as spam, how many are actually spam? |
| **Recall** | Of all actual spam emails, how many were detected? |
| **PR-AUC / AP** | How does the model perform across Precision and Recall for the positive class? |
| **Log-Loss** | Are the predicted probabilities reliable, or does the model make highly confident errors? |
| **Confusion Matrix** | How many TP, FP, FN, and TN occurred at the chosen threshold? |

---

# 15. The Central Idea

We are not evaluating exactly the same thing with all these metrics.

**ROC-AUC:**

$$
Discrimination\ /\ Ranking\ of\ Scores
$$

**PR:**

$$
Precision \times Recall
$$

**AP:**

$$
Summary\ of\ the\ Precision\text{-}Recall\ Curve
$$

**Log-Loss:**

$$
Quality\ of\ Probabilities\ and\ Penalty\ for\ Confident\ Errors
$$

**Confusion Matrix:**

$$
TP,\ FP,\ FN,\ TN\ at\ the\ Chosen\ Threshold
$$

Therefore, a **high ROC-AUC, such as 0.90, does not automatically mean that the model is excellent for a spam-filtering application**.

It may be ranking the probabilities well while still producing low Precision when we actually need to classify emails.

Similarly, a model may correctly classify the final class while assigning poor probabilities. In that case, **Log-Loss** can reveal problems that Accuracy or ROC-AUC do not show.

> **ROC-AUC helps answer whether the model can separate and rank the classes. PR/AP helps determine whether that ability is useful for identifying a rare positive class. Log-Loss evaluates the quality of the predicted probabilities and strongly penalizes confident errors. The confusion matrix shows what happens after choosing a specific threshold.**

<br><br>

# 🎯 16. What Is Your Objective?

The choice of metric should start with the **question we want to answer**, not with whichever metric appears most familiar or produces the highest score.

| **What do you want to evaluate?** | **Recommended Metric** |
|---|---|
| **Rank / compare models without a fixed threshold** | **ROC-AUC** |
| **Find rare positives that are costly to miss** | **PR-AUC / AP + Recall** |
| **Ensure that predicted positives are actually positive** | **Precision** |
| **Find most of the positive cases** | **Recall** |
| **Balance Precision and Recall** | **F1-score** |
| **Obtain reliable, well-calibrated probabilities** | **Log-Loss** |
| **Understand exactly which predictions were correct or incorrect at a specific threshold** | **Confusion Matrix** |

### 🧭 How Should You Think About Metric Selection?

#### 1. I want to compare the models' discrimination ability

> **“Which model can rank positive cases above negative cases more effectively?”**

Use:

$$
ROC\text{-}AUC
$$

It is especially useful when a fixed threshold has not yet been defined.

<br><br>

#### 2. I want to find a rare positive class

> **“I need to find the positives, but I do not want to generate a huge number of false positives.”**

Use:

$$
PR\text{-}AUC\ /\ AP + Recall
$$

This is particularly important in problems such as:

- Spam
- Fraud
- Anomaly detection
- Rare failures
- Rare diseases
- High-risk events

When the positive class is rare, the Precision-Recall curve provides a clearer view of the trade-off between:

$$
Precision \leftrightarrow Recall
$$

<br><br>

#### 3. I want reliable probabilities

> **“When the model says 90%, does that probability actually represent approximately a 90% chance?”**

In this case, we need to evaluate the **quality of the probabilities**, not just whether the final class was correct.

An important metric is:

$$
LogLoss
$$

Log-Loss strongly penalizes **confident errors**.

For example:

$$
-\log(0.90)\approx0.11
$$

is a small penalty when the prediction is correct and confident.

Whereas:

$$
-\log(0.10)\approx2.30
$$

is a much larger penalty when the model assigns a very low probability to the class that actually occurred.

<br><br>

## 🧠 The Metric Depends on the Question

There is no universally best metric.

The choice depends on what we want to optimize:

$$
Objective \rightarrow Metric \rightarrow Threshold \rightarrow Decision
$$

Therefore:

> **Always start with the question. The metric is a decision, not a default.**

### 📌 Example: Spam Filter

If the question is:

> **“Can the model rank emails according to their probability of being spam?”**

→ **ROC-AUC**

If the question is:

> **“Can the model find the few spam emails without blocking too many legitimate emails?”**

→ **PR-AUC / AP + Recall + Precision**

If the question is:

> **“Are the spam probabilities provided by the model reliable?”**

→ **Log-Loss + Calibration**

If the question is:

> **“What actually happened after choosing a threshold of 0.50?”**

→ **Confusion Matrix + Precision + Recall + F1-score**

<br><br>

## 🔑 Final Rule

$$
Do\ not\ choose\ the\ metric\ first.
$$

$$
First\ define\ what\ “good”\ means\ for\ the\ problem.
$$

<br>

### <p align="center">The metric should answer the business or application question — not the other way around.</p>

