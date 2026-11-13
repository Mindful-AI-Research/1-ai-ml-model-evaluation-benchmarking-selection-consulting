# Class 14 — Target Leakage and Reproducibility

Institution: Pontifical Catholic University of São Paulo (PUC-SP)  
School: FACEI — Computer Science Department  
Program: BSc in Human-Centered AI & Data Science · 6th Semester · 2026  
Subject: Data Science & Machine Learning — Model Evaluation, Benchmarking & Selection  
Professor: Giovani Giulio Tristão Thibes Vieira  
Documentation: Fabiana Campanari

[Mindful AI Research](https://github.com/Mindful-AI-Research)

> Numbering updated at the author’s request. The original handout identifies this content as Class 13. Academic metadata supplied by the author.

## 1. Overview

Target leakage, temporal leakage, and reproducibility: evaluate whether a strong metric represents a valid prediction task.

## 2. The decisive question

Would I have this column, in this form, at the moment of prediction? If not, exclude it. If uncertain, investigate its provenance. Availability matters even when a feature barely changes accuracy.

## 3. Predicting failure in week 6

Target: reprovou. Available: curso, semestre_ingresso, nota_P1 (week 5), faltas_ate_semana_6. Investigate frequentou_monitoria: did attendance occur before or after week 6?

## 4. Unavailable inputs

Exclude faltas_no_semestre, nota_P2, fez_substitutiva, media_final, and sala_prova_final. They contain later information or closely reflect the outcome. This is failure prediction, not a demonstrated dropout task.

## 5. Reported experiment

Simulated dataset: 1,000 students; 10-fold cross-validation. Accuracy with end-of-semester features: 0.953; week-6 features only: 0.744; always predicting “did not fail”: 0.628. These are handout results, not independently reproduced experiments.

## 6. One feature at a time

Added to week-6 inputs: nota_P2 → 0.861; faltas_no_semestre → 0.746; fez_substitutiva → 0.806; media_final → 0.955; sala_prova_final → 0.745. The exam room remains invalid despite negligible impact.

## 7. Temporal validation

For future-prediction tasks, random date mixing can put future observations in training. The handout proposes forward validation with TimeSeriesSplit: train on earlier observations and evaluate later observations. The timeline is conceptual.

## 8. Aggregates can leak

When predicting on day 3, the whole-month mean includes future days. A mean restricted to days 1 and 2 respects the illustrated cutoff. A valid split does not automatically make every feature valid.

## 9. Reproducibility

Keep the random split fixed for repeated comparisons. The handout shows train_test_split(X, y, test_size=0.2, random_state=42). This excerpt assumes imports and data already exist. The seed is not a parameter to search for a better score.

## 10. Audit → Remove → Document

Define the target and prediction moment; audit feature availability; remove unavailable inputs; check temporal aggregates; choose appropriate validation; fix the seed; save the complete Pipeline and prepare a model card. These are recommendations, not verified saved artifacts.

## 11. Evidence and limitations

Primary source: 2-Handout.pdf, dated November 10, 2026, labeled Class 13 in the original. This edition is labeled Class 14 at the author’s request. The original has 21 PDF pages and slide numbering jumps from 19 to 23. No missing content is reconstructed. Code, environment versions, holdout results, and deployment are not established by the PDF alone.

## References and scope

Primary source: `2-Handout.pdf`. References mentioned by the handout: Kapoor and Narayanan, Patterns, 2023; H2O.ai (antibiotic); IBM (chargeback); Larsen and Becker, 2019 (credit). Full bibliographic details were not independently verified. Companion notebooks are outside this document’s implementation evidence scope.
