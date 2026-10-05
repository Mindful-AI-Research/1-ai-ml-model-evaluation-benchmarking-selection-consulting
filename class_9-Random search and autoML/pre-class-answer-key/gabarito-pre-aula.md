# Gabarito — Busca Aleatória, Otimização e AutoML

## 1. Por que a busca aleatória é motivada pelo custo da grade?

**Resposta correta:**

> **porque seriam 6.400 treinos: cada hiperparâmetro novo MULTIPLICA o custo**

**Explicação:**  
São 1.280 combinações × 5 dobras de validação cruzada = **6.400 ajustes de modelo**. A cada novo eixo de hiperparâmetro, o número de combinações pode crescer multiplicativamente.

---

## 2. Com o mesmo orçamento de 16 medições, por que o sorteio costuma vencer a grade?

**Resposta correta:**

> **porque pontos aleatórios cobrem áreas maiores do espaço do que pontos alinhados**

**Explicação:**  
A busca aleatória tende a explorar o espaço de hiperparâmetros de forma mais eficiente, especialmente quando apenas alguns hiperparâmetros realmente influenciam o desempenho.

---

## 3. O que mostrou a repetição da busca com 100 sementes?

**Resposta correta:**

> **as 100 ficaram acima da grade (média 0,640), e a semente 0 ficou até abaixo da média**

**Explicação:**  
Isso mostra que o ganho da busca aleatória não dependia apenas de uma semente particularmente favorável.

---

## 4. Qual a chance de acertar a região boa com 16 sorteios?

**Resposta correta:**

> **cerca de 93% de chance — e o número de hiperparâmetros não entra na conta**

**Cálculo:**

$$
1 - (1-p)^n
$$

Com $p=0,15$ e $n=16$:

$$
1-(1-0,15)^{16}
=
1-0,85^{16}
\approx 0,926
$$

Ou seja, aproximadamente **92,6%**.

---

## 5. O que significa `uniform(0.1, 0.8)`?

**Resposta correta:**

> **0,1 e 0,9 — o segundo argumento é a LARGURA, somada ao início**

**Explicação:**  
Em `scipy.stats.uniform(loc, scale)`, o primeiro argumento é o início (`loc`) e o segundo é a largura (`scale`).

Portanto:

$$
0,1 + 0,8 = 0,9
$$

A distribuição sorteia valores no intervalo **[0,1; 0,9)**.

---

## 6. Por que usar `loguniform` para o `C` da regressão logística?

**Resposta correta:**

> **porque no uniform cerca de 90% dos sorteios cairiam entre 10 e 100, enquanto o loguniform dá 25% a cada ordem de grandeza**

**Explicação:**  
Para parâmetros que variam em várias ordens de grandeza, como $C$ entre 0,01 e 100, uma distribuição logarítmica oferece uma exploração mais equilibrada.

---

## 7. O que sugere o melhor `ccp_alpha` estar exatamente no menor valor da faixa?

**Resposta correta:**

> **que o pico pode estar FORA da faixa: convém ampliá-la e rodar de novo**

**Explicação:**  
Quando o melhor valor aparece na borda da distribuição, isso pode indicar que a faixa escolhida não contém o verdadeiro ótimo.

---

## 8. Quantos ajustes uma `RandomizedSearchCV` com `n_iter=40` e `cv=5` faz?

**Resposta correta:**

> **200 — n_iter × dobras, independentemente de quantos hiperparâmetros existam**

**Cálculo:**

$$
40 \times 5 = 200
$$

São **200 ajustes de modelo**, sem contar o `refit` final.

---

## 9. Como interpretar o `cv_results_`?

**Resposta correta:**

> **é um platô: a diferença cabe no ruído, e vale preferir a configuração mais simples ou mais estável**

**Explicação:**  
Uma diferença de apenas 0,014 entre candidatos, diante de uma variabilidade de aproximadamente ±0,11 entre dobras, não fornece evidência forte de que o primeiro candidato seja realmente superior.

---

## 10. O que significa a estratégia "grosso ao fino"?

**Resposta correta:**

> **aleatória larga e em log primeiro, depois uma grade pequena em volta dos melhores pontos — parando quando os números viram platô**

**Explicação:**  
Primeiro explora-se amplamente o espaço para localizar regiões promissoras. Depois, concentra-se a busca em torno dessas regiões, sem continuar refinando quando o desempenho já entrou em um platô.

---

## 11. Qual número NÃO deve ser reportado como desempenho?

**Resposta correta:**

> **o best_score_, porque é o MÁXIMO de 40 medições feitas na mesma validação que escolheu**

**Explicação:**  
O `best_score_` é usado para selecionar o melhor candidato durante a otimização. Como ele representa o máximo observado entre várias tentativas, tende a ser **otimista** como estimativa de generalização.

A `nested CV` fornece uma avaliação mais apropriada do processo de seleção, enquanto o **teste guardado** pode fornecer uma avaliação final independente quando foi realmente mantido fora de todo o processo de seleção.

---

## 12. O que a otimização bayesiana faz de diferente?

**Resposta correta:**

> **usa as medições já feitas para ajustar um modelo substituto e decidir onde medir em seguida**

**Explicação:**  
Ao contrário de simplesmente sortear novos pontos, a otimização bayesiana utiliza os resultados anteriores para orientar inteligentemente as próximas avaliações.

---

## 13. Por que uma lista de espaços no mini-AutoML é um espaço condicional?

**Resposta correta:**

> **porque o espaço é condicional: o C só existe se o sorteio cair na logística, o ccp_alpha só se cair na floresta**

**Explicação:**  
Cada modelo possui seus próprios hiperparâmetros. Assim, os parâmetros relevantes dependem de qual modelo foi selecionado.

---

## 14. O que os resultados do mini-AutoML ensinam?

**Resposta correta:**

> **que escolher o MODELO pesou muito mais (0,31 de diferença) do que afinar o vencedor (0,005 entre o melhor e a mediana)**

**Explicação:**

Logística:

$$
0,721
$$

Árvore:

$$
0,413
$$

Diferença:

$$
0,721-0,413=0,308
$$

Enquanto a diferença entre o melhor resultado da logística e sua mediana é:

$$
0,721-0,716=0,005
$$

Portanto, **a escolha da família do modelo teve impacto muito maior do que o refinamento dos hiperparâmetros dentro do modelo vencedor**.

---

## 15. O que acontece ao adicionar sete modelos mantendo `n_iter=30`?

**Resposta correta:**

> **cada modelo receber ~3 sorteios: pouco para cobrir o espaço dele, e a conta 1 − (1 − p)^n não fecha**

**Explicação:**  
Com mais modelos competindo pelo mesmo orçamento de 30 sorteios, cada modelo recebe, em média, poucas oportunidades de exploração.

Isso reduz a capacidade de explorar adequadamente o espaço de hiperparâmetros de cada família.

---

## 16. O que o aumento da AP de 0,721 para 0,936 ao incluir popularidade revela?

**Resposta correta:**

> **que ele otimizaria com prazer uma variável que vaza o alvo: decidir o que entra no modelo continua sendo seu**

**Explicação:**  
O AutoML otimiza a métrica que recebe. Ele não sabe, por conta própria, que determinada variável representa **vazamento de informação do alvo**.

Portanto, a seleção de features e a definição de quais informações são válidas para o modelo continuam sendo responsabilidades do pesquisador.

---

# Pergunta aberta

## Ficou alguma dúvida da aula de hoje?

**Resposta:**

> **Não ficou nenhuma dúvida importante. Entendi que a busca aleatória explora melhor hiperparâmetros relevantes com orçamento limitado, que `loguniform` é útil para faixas em várias ordens de grandeza e que `best_score_` pode ser otimista, sendo importante avaliar com nested CV ou teste guardado.**
