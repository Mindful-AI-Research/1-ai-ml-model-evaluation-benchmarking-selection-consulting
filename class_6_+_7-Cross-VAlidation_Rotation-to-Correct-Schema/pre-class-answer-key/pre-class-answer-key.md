
# Gabarito Bilíngue — Validação Cruzada / Bilingual Answer Key — Cross-Validation

## 1) "Avaliar com uma única divisão treino/teste é como jogar a moeda." Por quê?
**"Evaluating with a single train/test split is like flipping a coin." Why?**

* A) porque o modelo responde ao acaso / because the model answers randomly
* B) porque a acurácia é sempre 50% / because accuracy is always 50%
* C) porque a nota varia conforme a sorte do corte; a solução é a média de vários cortes (validação cruzada) / because the score varies depending on the luck of the split; the solution is to average multiple splits (cross-validation) ✅
* D) porque não se deve usar teste / because a test set should not be used

**Resposta / Answer:** C) porque a nota varia conforme a sorte do corte; a solução é a média de vários cortes (validação cruzada) / because the score varies depending on the luck of the split; the solution is to average multiple splits (cross-validation).

<br><br>

## 2) Em `cross_val_score(modelo, X, y, cv=5)`, quem define o que é treino e o que é teste?
**In `cross_val_score(model, X, y, cv=5)`, what defines what is training and what is testing?**

* A) você, separando as linhas à mão antes de chamar a função / you, manually separating the rows before calling the function
* B) o `cv=5`: ele corta o `X, y` em 5 pedaços e faz o rodízio automaticamente / `cv=5`: it splits `X, y` into 5 parts and automatically rotates them ✅
* C) o modelo, durante o `fit` / the model, during `fit`
* D) é preciso passar dois conjuntos `X_treino` e `X_teste` / you need to provide two sets, `X_train` and `X_test`

**Resposta / Answer:** B) o `cv=5`: ele corta o `X, y` em 5 pedaços e faz o rodízio automaticamente / `cv=5`: it splits `X, y` into 5 parts and automatically rotates them.

<br><br>

## 3) O `X, y` que você passa para o `cross_val_score` deve:
**The `X, y` passed to `cross_val_score` should:**

* A) ser apenas o bloco de treino/validação — o teste final fica separado e é tocado uma vez no fim / be only the training/validation block — the final test set remains separate and is used once at the end ✅
* B) ser a base inteira, incluindo o teste final / be the entire dataset, including the final test set
* C) conter só a classe positiva / contain only the positive class
* D) já estar padronizado com o conjunto todo / already be standardized using the entire dataset

**Resposta / Answer:** A) ser apenas o bloco de treino/validação — o teste final fica separado e é tocado uma vez no fim / be only the training/validation block — the final test set remains separate and is used once at the end.

<br><br>

## 4) Um 5-fold deu as acurácias `[0,80; 0,90; 0,70; 0,85; 0,75]`. A média é:
**A 5-fold produced accuracies `[0.80; 0.90; 0.70; 0.85; 0.75]`. What is the mean?**

* A) 0,78 / 0.78
* B) 0,80 / 0.80 ✅
* C) 0,82 / 0.82
* D) 0,85 / 0.85

**Resposta / Answer:** B) 0,80 / 0.80.

<br><br>

## 5) O resultado honesto de uma validação cruzada é:
**The honest result of cross-validation is:**

* A) a maior nota entre as dobras / the highest score among the folds
* B) a nota da primeira dobra / the score from the first fold
* C) a média das dobras E o desvio-padrão (média ± desvio) / the mean across folds AND the standard deviation (mean ± standard deviation) ✅
* D) um único número, sem desvio / a single number, without deviation

**Resposta / Answer:** C) a média das dobras E o desvio-padrão (média ± desvio) / the mean across folds AND the standard deviation (mean ± standard deviation).

<br><br>

## 6) O `StratifiedKFold` serve para:
**`StratifiedKFold` is used to:**

* A) acelerar o treino / speed up training
* B) padronizar as variáveis / standardize the variables
* C) embaralhar melhor as linhas / shuffle the rows better
* D) manter a mesma proporção das classes em cada dobra / maintain the same class proportion in each fold ✅

**Resposta / Answer:** D) manter a mesma proporção das classes em cada dobra / maintain the same class proportion in each fold.

<br><br>

## 7) Você tem 100 exemplos, mas só 3 positivos, e quer 5 dobras estratificadas (`k=5`). O que acontece?
**You have 100 examples, but only 3 positives, and you want 5 stratified folds (`k=5`). What happens?**

* A) funciona: o `StratifiedKFold` cria positivos sintéticos para completar / it works: `StratifiedKFold` creates synthetic positives to fill the folds
* B) pelo menos 2 dobras ficam sem nenhum positivo — `k` não pode passar do nº de positivos / at least 2 folds will have no positive examples — `k` cannot exceed the number of positive examples ✅
* C) o modelo fica mais preciso por ter classe rara / the model becomes more accurate because the class is rare
* D) a acurácia vira sempre 100% / accuracy always becomes 100%

**Resposta / Answer:** B) pelo menos 2 dobras ficam sem nenhum positivo — `k` não pode passar do nº de positivos / at least 2 folds will have no positive examples — `k` cannot exceed the number of positive examples.

<br><br>

## 8) O Leave-One-Out (LOO) é:
**Leave-One-Out (LOO) is:**

* A) o k-fold levado ao extremo, com `k = n`: cada exemplo, sozinho, é o teste uma vez / k-fold taken to the extreme, with `k = n`: each example is used alone as the test set once ✅
* B) um k-fold que usa sempre `k=2` / a k-fold that always uses `k=2`
* C) treinar e testar no mesmo conjunto / training and testing on the same dataset
* D) sortear 1 exemplo e ignorar o resto / randomly selecting 1 example and ignoring the rest

**Resposta / Answer:** A) o k-fold levado ao extremo, com `k = n`: cada exemplo, sozinho, é o teste uma vez / k-fold taken to the extreme, with `k = n`: each example is used alone as the test set once.

<br><br>

## 9) Uma consequência prática do LOO (`n=10` exemplos) é:
**A practical consequence of LOO (`n=10` examples) is:**

* A) faz apenas 2 treinos / it performs only 2 training runs
* B) cada rodada testa 5 exemplos / each round tests 5 examples
* C) faz 10 treinos e a nota de cada rodada é 0 ou 1 (teste de 1 exemplo) — só a média faz sentido / it performs 10 training runs and each round's score is 0 or 1 (testing 1 example) — only the average makes sense ✅
* D) não precisa treinar o modelo / it does not need to train the model

**Resposta / Answer:** C) faz 10 treinos e a nota de cada rodada é 0 ou 1 (teste de 1 exemplo) — só a média faz sentido / it performs 10 training runs and each round's score is 0 or 1 (testing 1 example) — only the average makes sense.

<br><br>

## 10) O `RepeatedKFold` (k-fold repetido) faz:
**`RepeatedKFold` (repeated k-fold) does:**

* A) aumenta o número de dobras dentro de uma rodada / increases the number of folds within one round
* B) repete o k-fold várias vezes, com sorteios diferentes, e tira a média de todas as notas / repeats k-fold several times with different random splits and averages all the scores ✅
* C) usa o mesmo sorteio toda vez / uses the same split every time
* D) remove os exemplos difíceis / removes difficult examples

**Resposta / Answer:** B) repete o k-fold várias vezes, com sorteios diferentes, e tira a média de todas as notas / repeats k-fold several times with different random splits and averages all the scores.

<br><br>

## 11) Você tem vários exames do mesmo paciente. Se exames do MESMO paciente caem no treino E no teste:
**You have several exams from the same patient. If exams from the SAME patient end up in both training AND testing:**

* A) não há problema algum / there is no problem
* B) o teste fica maior que o treino / the test set becomes larger than the training set
* C) o modelo aprende melhor a doença / the model learns the disease better
* D) o modelo "decora" o paciente e acerta por reconhecê-lo — nota inflada que some com pacientes novos / the model "memorizes" the patient and gets the answer right by recognizing them — an inflated score that disappears with new patients ✅

**Resposta / Answer:** D) o modelo "decora" o paciente e acerta por reconhecê-lo — nota inflada que some com pacientes novos / the model "memorizes" the patient and gets the answer right by recognizing them — an inflated score that disappears with new patients.

<br><br>

## 12) O `GroupKFold` resolve isso porque:
**`GroupKFold` solves this because:**

* A) garante que todas as linhas de um mesmo grupo fiquem do mesmo lado (ou todas no treino, ou todas no teste) / it ensures that all rows from the same group stay on the same side (either all in training or all in testing) ✅
* B) remove os grupos com poucas linhas / removes groups with few rows
* C) padroniza cada grupo separadamente / standardizes each group separately
* D) treina um modelo por grupo / trains one model per group

**Resposta / Answer:** A) garante que todas as linhas de um mesmo grupo fiquem do mesmo lado (ou todas no treino, ou todas no teste) / it ensures that all rows from the same group stay on the same side (either all in training or all in testing).

<br><br>

## 13) Quando os dados têm data, a divisão correta é:
**When the data contains dates, the correct splitting strategy is:**

* A) por tempo: treinar no passado e testar no futuro (não sortear as linhas) / by time: train on the past and test on the future (do not randomly shuffle the rows) ✅
* B) sortear aleatoriamente como sempre / randomly shuffle as usual
* C) testar no passado e treinar no futuro / test on the past and train on the future
* D) usar só o mês mais recente / use only the most recent month

**Resposta / Answer:** A) por tempo: treinar no passado e testar no futuro (não sortear as linhas) / by time: train on the past and test on the future (do not randomly shuffle the rows).

<br><br>

## 14) Qual a relação entre segurar 2016–2021 como teste (treino 2000–2015) e o `TimeSeriesSplit`?
**What is the relationship between holding out 2016–2021 as the test set (training on 2000–2015) and `TimeSeriesSplit`?**

* A) são coisas sem relação / they are unrelated
* B) o `TimeSeriesSplit` sorteia as datas / `TimeSeriesSplit` randomly shuffles the dates
* C) o primeiro é um hold-out temporal (1 corte); o `TimeSeriesSplit` é isso repetido em vários cortes, dando uma média mais estável / the first is a temporal hold-out (1 split); `TimeSeriesSplit` repeats this across multiple splits, producing a more stable average ✅
* D) o hold-out temporal usa o futuro no treino / the temporal hold-out uses the future for training

**Resposta / Answer:** C) o primeiro é um hold-out temporal (1 corte); o `TimeSeriesSplit` é isso repetido em vários cortes, dando uma média mais estável / the first is a temporal hold-out (1 split); `TimeSeriesSplit` repeats this across multiple splits, producing a more stable average.

<br><br>

## 15) Por que rodar a validação cruzada DENTRO de um `Pipeline` (pré-processamento + modelo)?
**Why run cross-validation INSIDE a `Pipeline` (preprocessing + model)?**

* A) para usar menos memória / to use less memory
* B) para escolher o `k` sozinho / to choose `k` automatically
* C) para remover valores ausentes / to remove missing values
* D) porque o pré-processamento se refaz em cada dobra, só com o treino daquela dobra — sem vazar / because preprocessing is refit in each fold using only that fold's training data — preventing leakage ✅

**Resposta / Answer:** D) porque o pré-processamento se refaz em cada dobra, só com o treino daquela dobra — sem vazar / because preprocessing is refit in each fold using only that fold's training data — preventing leakage.

<br><br>

## 16) Padronizar (`StandardScaler`) usando o conjunto INTEIRO antes da validação cruzada é:
**Standardizing (`StandardScaler`) using the ENTIRE dataset before cross-validation is:**

* A) vazamento — a média/desvio incluem os dados que serão teste em cada dobra, inflando a nota / data leakage — the mean/standard deviation include data that will be used as test data in each fold, inflating the score ✅
* B) a forma recomendada / the recommended approach
* C) irrelevante para o resultado / irrelevant to the result
* D) necessário para o k-fold funcionar / necessary for k-fold to work

**Resposta / Answer:** A) vazamento — a média/desvio incluem os dados que serão teste em cada dobra, inflando a nota / data leakage — the mean/standard deviation include data that will be used as test data in each fold, inflating the score.

<br><br>

## 17) Ficou alguma dúvida da aula de hoje?
**Did you have any questions about today's class?**

**Uma dúvida que fiquei: na prática, como a gente sabe se o modelo está realmente generalizando bem e não só decorando os dados do treino? E quando os dados mudam com o tempo, como a gente percebe que está na hora de reavaliar ou treinar o modelo de novo?**

**One question I still have is: in practice, how do we know if the model is actually generalizing well and not just memorizing the training data? And when the data changes over time, how can we tell when it is time to reevaluate or retrain the model?**

