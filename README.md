<!--START  🇬🇧English LANGUAGE BUTTON  -->
##### \[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇬🇧 English](README.md)**\]   

<br><br>
<!--END 🇬🇧English LANGUAGE BUTTON  --  -




<!-- ======================================= Start Title ======================================= -->
# <p align="center">[Data Science & AI/ML]() 🎯 [Model Evaluation, Benchmarking & Selection]() ➝ [Research & Consulting Hub]()</p>

#### <p align="center">This repository presents a structured journey through **machine learning model evaluation, benchmarking, comparison, and selection**, combining theoretical foundations, hands-on experimentation, and practical decision-making across diverse datasets and real-world scenarios.</p>

<br><br>

<!-- ========= START TEASER ========= -->
<p align="center"><em>Training models is easy ... Choosing the winner is where the fun begins.</em></p>

### <p align="center">⚡️</p>
<br>

#

<br><br>
<!-- ========= END TEASER ========= -->


<!-- ========= START SPONSOR BADGE ========= -->
<p align="center">
  <a href="https://github.com/sponsors/Mindful-AI-Research">
    <img src="https://img.shields.io/badge/Sponsor-%E0%A5%90%20%E2%8B%86%20Mindful%20AI%20%E2%8B%86%20Research%20%26%20Consulting%20%F0%96%A4%90%20%E2%8B%86-00FFFF?style=for-the-badge&logo=githubsponsors&logoColor=white&labelColor=0a1f44" alt="Sponsor ॐ ⋆ Mindful AI ⋆ Research & Consulting 𖤐 ⋆">
  </a>
</p>


<br><br>
<!-- ========= END SPONSOR BADGE ========= -->

<!-- =========  START PUC HEADER GIF ========= -->
<p align="center">
   <img src="https://github.com/user-attachments/assets/791a69e2-d09a-429f-9257-f6667fff5c04 ">
 </p>

<br><br>
<!-- =========  END PUC HEADER GIF ========= -->
<!-- ======================================= Start nstitucional INFOR ===========================================  -->
[**Institution:**]() Pontifical Catholic University of São Paulo (PUC-SP)  <br>
[**School:**]() FACEI — Computer Science Department  <br>
[**Course:**]() BSc in Human-Centered AI & Data Science •  6th Semester • 2026 <br>
[**Subject:**]() Data Science  & Machine Learning — Model Evaluation, Benchmarking & Selection  <br>
[**Extensionist Projects:**]() Social projects with open-source software for community support <br>
**Professor:** [✨ Giovani Giulio Tristão Thibes Vieira]() <br>
**Author:** [Fabiana ⚡️ Campanari]() 


<br><br>

#

<br><br>
<!-- ======================================= SZEnd Institutional INFO ===========================================  -->

<!-- ========= START NOTE ========= -->
> [!WARNING]
>
> ⚠️ Projects may be publicly shared when permitted.  
> The focus is on applied, hands-on learning with real datasets in AI governance and security contexts.  
> All sensitive content remains protected in private repositories when required.
>

<br><br>

#

<br><br>
<!-- ========= END NOTE ========= -->



<br><br><br><br>

<!-- ========= End Confidentiality statement ========= -->
<!-- ========= START BADGES ========= -->
<p align="center">
  <img src="https://img.shields.io/badge/Focus-Model%20Evaluation-1ABC9C" />
  <img src="https://img.shields.io/badge/Validation-Cross--Validation-16A085" />
  <img src="https://img.shields.io/badge/Tuning-Hyperparameter%20Search-48C9B0" />
  <img src="https://img.shields.io/badge/Selection-Comparative%20Analysis-20B2AA" />
  <img src="https://img.shields.io/badge/Reproducibility-Pipelines%20%26%20Model%20Cards-76D7C4" />
  <img src="https://img.shields.io/badge/Tools-Python%20%7C%20scikit--learn%20%7C%20GitHub-0E7490" />
</p>
<br><br><br><br>

<!-- ========= END BADGES ========= -->

## [Overview]()

This repository is the **master hub** for the 6th-semester course on model evaluation, model comparison, and methodological rigor in Data Science. It centralizes the full semester roadmap, weekly classes, references, extension-oriented project work, and links to future sub-repositories for labs, assignments, experiments, and project deliverables.

The course teaches how to evaluate machine learning models carefully and fairly. In simple words, it asks a very important question: **“How do we know a model is actually good?”** Instead of trusting accuracy alone, we learn how to inspect mistakes, compare alternatives, avoid data leakage, tune models properly, calibrate probabilities, and publish reproducible results.

This hub is designed to:

- organize all weekly course content in one place;
- serve as a navigation center for semester materials;
- link out to sub-repositories created for specific weekly topics or projects;
- document extension activities and public-impact work;
- preserve a professional academic record of the course journey.

<br><br>

<br><br>

## [Table of Contents]()

- [Repository Mission](#repository-mission)
- [Why This Course Matters](#why-this-course-matters)
- [Learning Goals](#learning-goals)
- [Semester Roadmap](#semester-roadmap)
- [Weekly Course Table](#weekly-course-table)
- [Advanced Evaluation — Agentic AI Systems](#advanced-evaluation--agentic-ai-systems)
- [Assessment Structure](#assessment-structure)
- [Extension Project](#extension-project)
- [Repository Architecture](#repository-architecture)
- [Suggested Sub-Repository Pattern](#suggested-sub-repository-pattern)
- [How to Use This Hub](#how-to-use-this-hub)
- [Recommended Tooling](#recommended-tooling)
- [AI Resources](#ai-resources)
- [Bibliographic References](#bibliographic-references)
- [Mermaid Overview](#mermaid-overview)
- [Final Notes](#final-notes)

<br><br>

## [Repository Mission]()

This repository works as the **central semester hub** for the course. Think of it like a map in a museum: it does not replace every room, but it helps you know where everything is, how the ideas connect, and where to go next.

As the semester evolves, each week or major topic can branch into:

- a dedicated sub-repository;
- a notebook collection;
- a practical lab folder;
- a project deliverable repository;
- a case-study or experiment archive.

<br><br>

## [Why This Course Matters]()

A machine learning model is not useful just because it runs. A model is useful when it is evaluated correctly, compared fairly, and deployed responsibly.

This discipline helps students move from “I trained a model” to “I can defend why this model should be trusted.” That difference is what turns experimentation into professional data science practice.

For a child-friendly explanation:

- A model is like a student taking a test.
- Evaluation metrics are the teacher’s grading rules.
- Validation checks whether the student really learned, or just memorized the homework.
- Comparative analysis helps us decide which student performed best in the fairest way.

<br><br>

## [Learning Goals]()

By the end of the semester, this course aims to help students:

- choose appropriate metrics for classification and regression tasks;
- apply hold-out, stratified cross-validation, group-based validation, temporal validation, and nested validation correctly;
- prevent leakage using `Pipeline` and `ColumnTransformer`;
- tune hyperparameters with exhaustive and randomized search;
- interpret validation curves and learning curves;
- compare and select models using principled criteria and statistical tests;
- adjust decision thresholds and calibrate probabilities;
- ensure reproducibility through seeds, serialized artifacts, and model cards;
- publish clear, reusable, and responsible results in public repositories when appropriate.

<br><br>

## [Semester Roadmap]()

[***From metrics to rigorous model selection***] <br>

The semester follows a logical progression:

1. Understand what “good performance” means.
2. Learn which metrics fit which type of problem.
3. Validate models properly.
4. Tune models without fooling yourself.
5. Diagnose underfitting and overfitting.
6. Compare candidate models fairly.
7. Improve decision quality with thresholds and calibration.
8. Extend evaluation principles to adaptive, tool-using, and non-deterministic AI systems.
9. Build reproducible end-to-end evaluation workflows.
10. Communicate findings and publish results professionally.

<br><br>



## [Weekly Course Map](#weekly-course-map)

> [!TIP]
> Use this table as the semester’s living index. Replace each **Planned link** with the future GitHub repository, folder, notebook, slides, or report created for that class. A relative path works well for folders in this hub; an absolute URL works well for independent repositories.

| Week | Topic summary | Core ideas and expected artifact | Notes / files |
|---:|---|---|---|
|  | 🧠 **Part I — Evaluation Foundations and Metrics** |  |  |
| 1 | **Fundamentals of Model Evaluation** | Generalization, overfitting, underfitting, bias–variance trade-off, and data leakage. Create a concept-and-diagnostics notebook. | `[Planned link](#)` |
| 2 | **Classification Metrics I** | Confusion matrix, accuracy, precision, recall, F1-score, and imbalanced classes. Extension project launch and group formation. | `[Planned link](#)` |
| 3 | **Classification Metrics II** | ROC-AUC, precision–recall curves, log loss, macro/micro/weighted averaging. Build a classifier evaluation report. | `[Planned link](#)` |
| 4 | **Regression Metrics** | MAE, MSE, RMSE, R², MAPE, and residual analysis. Deliver a regression diagnostics notebook. | `[Planned link](#)` |
|  | 🧪 **Part II — Validation and Hyperparameter Search** |  |  |
| 5 | **Hold-Out Validation and Safe Pipelines** | Train/validation/test roles; `Pipeline` and `ColumnTransformer` as protections against leakage. | `[Planned link](#)` |
| 6 | **Cross-Validation I** | K-fold, stratified K-fold, leave-one-out, and repeated cross-validation. Compare validation estimates. | `[Planned link](#)` |
| 7 | **Cross-Validation II** | Group-based, time-series, and nested cross-validation. Choose a validation strategy that respects the data-generating process. | `[Planned link](#)` |
| 8 | **Exhaustive Hyperparameter Search** | Define search spaces and use `GridSearchCV`. Extension project data milestone. | `[Planned link](#)` |
| 9 | **Randomized Search and AutoML Overview** | Use `RandomizedSearchCV`; compare exhaustive and random search trade-offs; introduce AutoML. | `[Planned link](#)` |
| 10 | **Validation Curves** | Read performance versus hyperparameter values; diagnose overfitting and underfitting. **Challenge 3 substitute.** | `[Planned link](#)` |
| 11 | **Learning Curves** | Interpret performance versus training-set size; reason about bias and variance. | `[Planned link](#)` |
| — | **Academic Week** | Institutional academic activities; no regular weekly topic listed in the official schedule. | `[Calendar / notes](#)` |
|  | ⚖️ **Part III — Comparative Analysis, Reliability, and Delivery** |  |  |
| 12 | **Model Selection and Comparison** | One-standard-error rule, paired tests, and McNemar-style comparison logic. Produce a fair comparison report. | `[Planned link](#)` |
| 13 | **Decision Thresholds and Calibration** | Threshold tuning, calibration curves, and Brier score. Evaluate probability quality, not only class labels. | `[Planned link](#)` |
| — | **Advanced Evaluation — Agentic AI Systems** | **Supplementary advanced module.** Extend model-evaluation principles to adaptive systems that plan, call tools, retrieve information, and may produce different execution paths across runs. Covers task success, robustness, failure analysis, trace evaluation, LLM-as-a-Judge, human review, and evaluation governance. | `[Planned link](#)` |
| 14 | **Advanced Leakage Prevention and Reproducibility** | Random seeds, `joblib`, experiment consistency, and model cards. | `[Planned link](#)` |
| 15 | **Feature Selection Inside Validation** | Run feature selection within the cross-validation workflow to avoid inflated results. | `[Planned link](#)` |
<br><br>











































<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>


## 💌 [Let the data flow... Ping Me !](mailto:fabicampanari@proton.me)

<br>


#### <p align="center">  🛸๋ My Contacts [Hub](https://linktr.ee/fabianacampanari)


<br>

### <p align="center"> <img src="https://github.com/user-attachments/assets/517fc573-7607-4c5d-82a7-38383cc0537d" />


<br><br>

<p align="center">  ────────────── ⊹🔭๋ ──────────────

<!--
<p align="center">  ────────────── 🛸๋*ੈ✩* 🔭*ੈ₊ ──────────────
-->

<br>

<p align="center"> ➣➢➤ <a href="#top">Back to Top </a>
  

  
#
 
##### <p align="center">Copyright 2026 Mindful-AI-Assistants. Code released under the  [MIT license.](https://github.com/Mindful-AI-Research/model-evaluation-hub/blob/cf8bf0392253d4d1c9b187eac7b55527efae4c9e/LICENSE)
