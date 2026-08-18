

# 🧠 Gabarito — Filtro de Spam e Métricas de Classificação - Answer Key — Spam Filter and Classification Metrics

| # | Questão | ✅ Alternativa correta |
|---|---|---|
| **1** | O que o filtro de spam entrega para cada e-mail, antes de decidir? | **Uma nota (probabilidade) de quão provável é ser spam.** |
| **2** | “O modelo ordena bem” significa que: | **Os spams recebem nota maior que os e-mails bons.** |
| **3** | Baixar o limiar (marcar mais e-mails como spam) tende a: | **Aumentar o Recall e reduzir a Precision.** |
| **4** | No filtro de spam, um Falso Positivo (FP) é: | **Um e-mail bom que foi jogado no spam.** |
| **5** | Uma AUC = 0,5 indica que o modelo: | **Não ordena melhor que o acaso (equivalente a uma classificação aleatória).** |
| **6** | A curva ROC coloca, ao variar o corte: | **Recall (TPR) contra a taxa de falso alarme (FPR).** |
| **7** | Quando a classe positiva é RARA, qual visão é mais informativa? | **A curva PR (Precision-Recall) e a AP.** |
| **8** | O Log-Loss serve principalmente para: | **Medir a qualidade das probabilidades, punindo o erro confiante.** |

<br><br>

## 🇺🇸 English — Answer Key

| # | Question | ✅ Correct Answer |
|---|---|---|
| **1** | What does a spam filter provide for each email before making a decision? | **A score (probability) indicating how likely the email is to be spam.** |
| **2** | What does “the model ranks well” mean? | **Spam emails receive higher scores than legitimate emails.** |
| **3** | Lowering the threshold (marking more emails as spam) tends to: | **Increase Recall and decrease Precision.** |
| **4** | In a spam filter, what is a False Positive (FP)? | **A legitimate email incorrectly classified as spam.** |
| **5** | An AUC = 0.5 indicates that the model: | **Does not rank better than chance (equivalent to random classification).** |
| **6** | As the threshold changes, the ROC curve plots: | **Recall (TPR) against the False Positive Rate (FPR).** |
| **7** | When the positive class is RARE, which view is more informative? | **The PR (Precision-Recall) curve and AP.** |
| **8** | What is Log-Loss mainly used for? | **Measuring the quality of predicted probabilities and penalizing confident errors.** |

<br><br>

## 📌 Resumo para memorizar - Memorization Summary

### 🇧🇷 Português

- **Probabilidade →** o modelo dá uma **nota** antes de decidir.
- **Ordenação →** spam deve receber **nota maior** que e-mails bons.
- **Limiar ↓ →** **Recall ↑** e **Precision ↓** tendencialmente.
- **FP →** e-mail **bom enviado para o spam**.
- **AUC = 0,5 →** desempenho equivalente ao **acaso**.
- **ROC →** **TPR (Recall) × FPR (taxa de falso positivo)**.
- **Classe rara →** **PR + AP** podem ser mais informativas que ROC-AUC.
- **Log-Loss →** avalia a **qualidade das probabilidades**, penalizando previsões confiantes e erradas.

  <br>

### 🇺🇸 English

- **Probability →** the model produces a **score** before making a decision.
- **Ranking →** spam should receive a **higher score** than legitimate emails.
- **Threshold ↓ →** **Recall ↑** and **Precision ↓**, in general.
- **FP →** a **legitimate email incorrectly sent to spam**.
- **AUC = 0.5 →** performance equivalent to **chance**.
- **ROC →** **TPR (Recall) × FPR (False Positive Rate)**.
- **Rare class →** **PR + AP** can be more informative than ROC-AUC.
- **Log-Loss →** evaluates the **quality of predicted probabilities**, strongly penalizing confident errors.

<br><br>

## 🧠 Regra para lembrar - Rule to Remember

### 🇧🇷 Português

$$
Probabilidade
\rightarrow
Ordenação
\rightarrow
Limiar
\rightarrow
Classificação
$$

<br>

### 🇺🇸 English

$$
Probability
\rightarrow
Ranking
\rightarrow
Threshold
\rightarrow
Classification
$$

<br><br>

## 📊 O que cada métrica responde? -  What does each metric answer?

### 🇧🇷 Português

$$
ROC\text{-}AUC
\rightarrow
\text{“O modelo ordena bem?”}
$$

$$
PR/AP
\rightarrow
\text{“O modelo encontra bem os positivos raros?”}
$$

$$
Log\text{-}Loss
\rightarrow
\text{“As probabilidades são confiáveis?”}
$$

$$
Matriz\ de\ Confusão
\rightarrow
\text{“O que aconteceu neste limiar?”}
$$

<br>

### 🇺🇸 English

$$
ROC\text{-}AUC
\rightarrow
\text{“Does the model rank the classes well?”}
$$

$$
PR/AP
\rightarrow
\text{“Does the model identify rare positives effectively?”}
$$

$$
Log\text{-}Loss
\rightarrow
\text{“Are the predicted probabilities reliable?”}
$$

$$
Confusion\ Matrix
\rightarrow
\text{“What happened at this threshold?”}
$$

<br><br>

## 🎯 Mensagem principal - Key Takeaway

> **🇧🇷 Comece sempre pela pergunta. A métrica é uma decisão, não um padrão.**

> **🇺🇸 Always start with the question. The metric is a decision, not a default.**
