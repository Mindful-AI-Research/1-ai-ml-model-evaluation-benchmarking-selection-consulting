# [CD1 — Lesson 12: Decision thresholds and calibration]()

## [Answer key]()

> Expected answers to the form, based on the supplied text. Options are numbered from top to bottom (1–4). This document mirrors the lesson’s expected answer key, rather than providing a technical review of its claims.

## [Quick reference]()

| Question | Option |
| :--- | :---: |
| 1 | 3 |
| 2 | 2 |
| 3 | 4 |
| 4 | 1 |
| 5 | 2 |
| 6 | 3 |
| 7 | 1 |
| 8 | 4 |
| 9 | 2 |
| 10 | 3 |
| 11 | 1 |
| 12 | 4 |
| 13 | 2 |
| 14 | 1 |
| 15 | 3 |
| 16 | 2 |
| 17 | 4 |

<br><br>

## [1. The lesson used ROC-AUC rather than accuracy to compare the four candidates. Why?]()

### [Correct answer — option 3]()

> because accuracy is accuracy AT THE 0.50 THRESHOLD — comparing models by accuracy would mean selecting a model with a threshold already built in, precisely the threshold the lesson will show is wrong

<br><br>

## [2. Logistic regression scored 0.960 ± 0.014 and the random forest 0.945 ± 0.025 on the same 10 folds. The practical tie range is ≥ 0.946. What should you do?]()

### [Correct answer — option 2]()

> the two are practically equivalent (the forest misses the range by one thousandth, with overlapping error bars), so choose logistic regression because it is simpler and more explainable

<br><br>

## [3. The depth-4 tree scored 0.803 ± 0.135. What is this huge standard deviation warning you about?]()

### [Correct answer — option 4]()

> the model changes its mind from fold to fold: with 90 positives in 595 rows, changing the sample changes the tree — 0.803 averages results that disagree with one another

<br><br>

## [4. The model returns 0.80 for a patient. What exactly does that number claim?]()

### [Correct answer — option 1]()

> among all patients assigned approximately 0.80 by the model, around 80% should be high-risk — a claim that can be checked against what actually happened

<br><br>

## [5. What does `modelo.predict(X)` do in a two-class problem?]()

### [Correct answer — option 2]()

> it compares the probability with 0.50: `pred = (proba >= 0.50).astype(int)` returns exactly the same vector

<br><br>

## [6. When is 0.50 actually the right threshold?]()

### [Correct answer — option 3]()

> when both types of error have the same cost AND the classes are balanced — neither condition holds for our patients

<br><br>

## [7. In the threshold table, from 0.15 to 0.70, precision rises from 0.567 to 0.909 and recall falls from 0.889 to 0.556. How should this be interpreted?]()

### [Correct answer — option 1]()

> raising the threshold trades one type of error for another: no threshold improves precision and recall simultaneously, so the choice requires an external criterion

<br><br>

## [8. With one FN costing as much as 5 FP, the 0.50 threshold costs 184 and the curve minimum, at 0.14, costs 103. Check the calculation for 0.50 and state the gain.]()

### [Correct answer — option 4]()

> 184 = 19 + 33·5; the 0.14 threshold saves 44%, without any retraining

<br><br>

## [9. The business team says: "I need to catch at least 90% of cases." How does this become a threshold?]()

### [Correct answer — option 2]()

> scan the curve and choose the HIGHEST threshold that still meets the target — this maximizes precision within the constraint; in our dataset, 0.148, with recall 0.900 and precision 0.570

<br><br>

## [10. Which question must be answered BEFORE choosing any threshold?]()

### [Correct answer — option 3]()

> how much each type of error costs the client

<br><br>

## [11. In the 0.50–1.00 range, the forest predicted 0.677, but the observed fraction was 0.806. What does this mean — and why?]()

### [Correct answer — option 1]()

> it underestimates precisely the highest-risk cases; 0.677 is the highest number it can produce, because it averages 200 trees and approximately 190 would need to agree to reach 0.95

<br><br>

## [12. Why does logistic regression come calibrated "out of the box" in this problem?]()

### [Correct answer — option 4]()

> because it is trained by minimizing log-loss, the very metric that penalizes dishonest probabilities — it was optimized for that

<br><br>

## [13. Regarding the reliability diagram (predicted on the x-axis, observed on the y-axis), which statement is CORRECT?]()

### [Correct answer — option 2]()

> no threshold enters the calculation: predictions are grouped into probability bins and the fraction of positives in each bin is counted; the diagonal represents perfect honesty

<br><br>

## [14. Your model’s probability scale is distorted. What should you do first?]()

### [Correct answer — option 1]()

> check whether another candidate ranks cases equally well and is already honest — if so, use it: no additional stage and no reserved data

<br><br>

## [15. The minimum-cost threshold is 0.23 for the forest and 0.14 for logistic regression; after correcting the forest’s scale, its 0.23 becomes 0.11. What is the practical consequence?]()

### [Correct answer — option 3]()

> the threshold lives on the model’s scale: switching models or correcting the scale requires RESCANNING the threshold — choose the model, check its probabilities, and only then set the threshold

<br><br>

## [16. In which situation is probability honesty NOT required?]()

### [Correct answer — option 2]()

> when you only need to rank the queue — selecting the 100 highest-risk patients to call requires the right order, not honest probability values, and ROC-AUC has already addressed that

<br><br>

## [17. On the test set, opened once: at threshold 0.50, validation predicted 0.309 per patient and the test delivered 0.310; at the chosen threshold, 0.173 was predicted versus 0.137 delivered. How should this be interpreted?]()

### [Correct answer — option 4]()

> the 0.50 line is the control and shows that validation estimates well (0.309 versus 0.310); 0.173 → 0.137 came out better than promised, and that is sampling luck, not merit

<br><br>

## [Open-ended field — Any questions about today’s lesson?]()

Describe what was unclear (the practical tie range, what a probability claims, the hidden 0.50 threshold, the cost curve, the recall target, predicted versus observed, why the forest stalls at 0.677, switching models instead of calibrating, the order of the three decisions, the audit…). We will review it next lesson.

### [Suggested relevant question]()

> How can we check whether the minimum-cost threshold selected during validation is stable rather than just a consequence of the sample, without using the test set to tune it?

This field is open-ended and has no single correct answer.
