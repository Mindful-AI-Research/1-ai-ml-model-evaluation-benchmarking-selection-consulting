

# ⚠️ Cuidado com o AUC em Bases Desbalanceadas

## ROC-AUC, Precision-Recall, AP, Log-Loss e Matriz de Confusão

Quando avaliamos um modelo de classificação, é importante não olhar apenas para uma métrica. Em problemas nos quais a **classe positiva é rara**, como a identificação de spam, uma métrica pode transmitir uma impressão melhor do que o desempenho real do modelo.

Imagine uma base com **1.000 e-mails**, sendo:

- **980 e-mails normais**
- **20 e-mails spam**

Nesse cenário:

$$
\text{Prevalência} = \frac{20}{1000} = 2\%
$$

Essa distribuição é chamada de **desbalanceamento de classes**.

<br><br>

## 1. Por que a acurácia pode enganar?

Imagine um modelo que simplesmente classifique **todos os 1.000 e-mails como “não spam”**.

Teríamos:

| | Previsto: Spam | Previsto: Não Spam |
|---|---:|---:|
| **Real: Spam** | VP = 0 | FN = 20 |
| **Real: Não Spam** | FP = 0 | VN = 980 |

A acurácia seria:

$$
Accuracy = \frac{VP+VN}{VP+VN+FP+FN}
$$

$$
Accuracy = \frac{0+980}{1000}=98\%
$$

**98% de acurácia parece excelente.**

Mas o modelo não identificou **nenhum spam**.

Seu recall para a classe spam seria:

$$
Recall = \frac{VP}{VP+FN}
$$

$$
Recall = \frac{0}{0+20}=0\%
$$

Ou seja:

> **98% de acurácia, mas 0% de recall para spam.**

Esse exemplo mostra por que, em bases desbalanceadas, a acurácia não deve ser analisada isoladamente.

<br><br>

# 2. O que o ROC-AUC realmente avalia?

A curva **ROC (Receiver Operating Characteristic)** mostra o comportamento do modelo enquanto variamos o threshold.

Ela relaciona:

$$
TPR = \frac{VP}{VP+FN}
$$

com:

$$
FPR = \frac{FP}{FP+VN}
$$

O **ROC-AUC** é a área sob essa curva.

Uma interpretação importante do AUC é sua capacidade de avaliar a **discriminação ou ordenação dos exemplos**.

Imagine que o modelo atribua uma probabilidade de spam para cada e-mail:

| E-mail | Classe real | Probabilidade de spam |
|---|---|---:|
| A | Spam | **0,99** |
| B | Spam | **0,95** |
| C | Spam | **0,90** |
| D | Spam | **0,85** |
| E | Não spam | **0,30** |
| F | Não spam | **0,20** |

O modelo está fazendo uma boa **ordenação**:

$$
Spam > Não\ Spam
$$

Os exemplos que realmente são spam receberam scores maiores.

Essa capacidade de separar e ordenar as classes é justamente uma das coisas capturadas pelo ROC-AUC.

<br><br>

# 3. Mas por que ter cuidado com o ROC-AUC?

Agora imagine que existam **muito poucos spams**.

Por exemplo:

- 990 e-mails normais
- 10 e-mails spam

O modelo pode apresentar um **ROC-AUC alto**, porque consegue ordenar os spams acima de grande parte dos e-mails normais.

Por exemplo:

$$
ROC\text{-}AUC \approx 0.90
$$

Isso indica uma **boa capacidade de discriminação**.

Mas isso não responde necessariamente à pergunta mais importante da aplicação:

> **Quando eu realmente bloquear e-mails como spam, quantos dos e-mails bloqueados serão realmente spam?**

Essa é uma questão relacionada à **Precision**.

<br><br>

# 4. Precision-Recall: olhando para o spam

A curva **Precision-Recall (PR)** pode ser especialmente útil quando a classe positiva é rara.

Ela relaciona:

### Precision

Entre os e-mails classificados como spam, quantos realmente são spam?

$$
Precision = \frac{VP}{VP+FP}
$$

### Recall

Entre todos os spams que realmente existem, quantos o modelo conseguiu encontrar?

$$
Recall = \frac{VP}{VP+FN}
$$

Portanto:

> **Precision → qualidade dos alarmes**

> **Recall → cobertura dos spams**

A curva PR é construída **variando o threshold**.

<br><br>

# 5. O que acontece quando alteramos o threshold?

Imagine que o modelo atribua probabilidades aos e-mails.

Se escolhermos:

$$
Threshold = 0.50
$$

qualquer e-mail com probabilidade de spam ≥ 0,50 será classificado como spam.

Mas podemos mudar esse valor.

### Threshold mais baixo

$$
Threshold = 0.30
$$

Mais e-mails serão classificados como spam.

Isso pode aumentar o:

$$
Recall
$$

porque conseguimos encontrar mais spams.

Porém, também podemos aumentar:

$$
FP
$$

e, consequentemente, reduzir a:

$$
Precision
$$

### Threshold mais alto

$$
Threshold = 0.80
$$

O modelo fica mais seletivo.

Podemos aumentar a Precision, mas alguns spams podem passar despercebidos, aumentando:

$$
FN
$$

e reduzindo o Recall.

<br><br>

# 6. Exemplo prático

Imagine novamente:

**1.000 e-mails**

- 990 não spam
- 10 spam

Agora suponha que, em determinado threshold, o modelo encontre:

- **VP = 9**
- **FN = 1**
- **FP = 90**
- **VN = 900**

O Recall será:

$$
Recall = \frac{9}{9+1}=90\%
$$

Parece excelente.

Mas a Precision será:

$$
Precision = \frac{9}{9+90}
$$

$$
Precision \approx 9,1\%
$$

Ou seja:

> O modelo encontrou **90% dos spams**, mas somente aproximadamente **9% dos e-mails classificados como spam eram realmente spam**.

Isso é extremamente importante em um sistema real.

Imagine que os 90 falsos positivos sejam e-mails legítimos importantes.

<br><br>

# 7. AP — Average Precision

A curva PR possui vários pontos porque estamos variando o threshold.

O **AP (Average Precision)** resume o comportamento da curva Precision-Recall em um único valor.

Uma forma de expressá-lo é:

$$
AP = \sum_n (R_n-R_{n-1})P_n
$$

onde:

- $P_n$ = Precision no ponto $n$
- $R_n$ = Recall no ponto $n$

Quanto mais alto o AP, melhor tende a ser o desempenho do modelo na relação entre **Precision e Recall**.

<br><br>

# 8. A linha de base da Precision-Recall

A linha de base da Precision-Recall corresponde aproximadamente à **prevalência da classe positiva**:

$$
Baseline = \frac{P}{P+N}
$$

Se temos:

- 200 positivos
- 800 negativos

então:

$$
Baseline = \frac{200}{1000}=0.20
$$

Portanto:


$$
\boxed{\text{Baseline} = 20\%}
$$


No exemplo de spam:

- 10 spams
- 990 e-mails normais

temos:

$$
Baseline = \frac{10}{1000}=1\%
$$

Ou seja, a classe spam representa somente **1% da base**.

Isso ajuda a contextualizar o desempenho da curva PR.

<br><br>

# 9. Matriz de Confusão

A matriz de confusão mostra o que aconteceu **depois que escolhemos um determinado threshold**.

| | **Previsto: Spam (1)** | **Previsto: Não Spam (0)** |
|---|---:|---:|
| **Real: Spam (1)** | **VP** | **FN** |
| **Real: Não Spam (0)** | **FP** | **VN** |

### VP — Verdadeiro Positivo

Era spam e o modelo previu spam.

### FN — Falso Negativo

Era spam, mas o modelo previu não spam.

### FP — Falso Positivo

Era um e-mail legítimo, mas o modelo previu spam.

### VN — Verdadeiro Negativo

Era um e-mail legítimo e o modelo previu não spam.

<br><br>

# 10. Diagonal = acertos

Na matriz:

$$
VP + VN
$$

representam os **acertos**.

Enquanto:

$$
FP + FN
$$

representam os **erros**.

No caso do spam:

- **FP:** um e-mail legítimo é bloqueado.
- **FN:** um spam passa para a caixa de entrada.

Dependendo da aplicação, podemos considerar um desses erros mais grave que o outro.

<br><br>

# 11. Log-Loss — quando o erro confiante custa caro

Enquanto ROC-AUC e PR avaliam principalmente a **ordenação e o desempenho da classificação**, o **Log-Loss** avalia a qualidade das **probabilidades previstas**.

A ideia central é:

> **Não basta acertar. O modelo também precisa ser bem calibrado na confiança que atribui à previsão.**

O Log-Loss pune fortemente o modelo quando ele **erra com muita confiança**.

Para uma única amostra:

$$
LogLoss = -\left[y\log(p)+(1-y)\log(1-p)\right]
$$

onde:

- $y$ = classe verdadeira, sendo $1$ para spam e $0$ para não spam
- $p$ = probabilidade atribuída pelo modelo à classe spam

### Se era spam

Nesse caso:

$$
y=1
$$

A fórmula fica:

$$
LogLoss=-\log(p)
$$

### Exemplo 1 — acerto confiante

Era spam e o modelo atribuiu:

$$
p=0.90
$$

Então:

$$
LogLoss=-\log(0.90)\approx0.11
$$

**Penalidade pequena → bom resultado.**

### Exemplo 2 — erro confiante

Era spam, mas o modelo atribuiu:

$$
p=0.10
$$

Então:

$$
LogLoss=-\log(0.10)\approx2.30
$$

**Penalidade muito maior → resultado ruim.**

A diferença é importante:

| Situação | $p$ para spam | Log-Loss |
|---|---:|---:|
| Spam, previsão muito confiante | 0,99 | ≈ 0,01 |
| Spam, previsão correta | 0,90 | ≈ 0,11 |
| Spam, previsão incerta | 0,50 | ≈ 0,69 |
| Spam, previsão errada | 0,10 | ≈ 2,30 |
| Spam, previsão errada e muito confiante | 0,01 | ≈ 4,61 |

Quanto menor o Log-Loss, melhor:

$$
\boxed{\text{Menor Log-Loss} = \text{melhor}}
$$

### 📌 O que o Log-Loss está medindo?

Podemos pensar assim:

> **ROC-AUC:** o modelo ordenou bem os casos?

> **PR/AP:** o modelo consegue encontrar a classe positiva rara mantendo uma boa relação entre Precision e Recall?

> **Log-Loss:** as probabilidades atribuídas pelo modelo são confiáveis ou ele está errando com excesso de confiança?

<br><br>

# 12. Por que o Log-Loss é importante?

Imagine dois modelos:

### Modelo A

Diz:

> “Este e-mail tem **51% de chance de ser spam**.”

E era spam.

### Modelo B

Diz:

> “Este e-mail tem **99% de chance de ser spam**.”

E também era spam.

Os dois acertaram a classe, mas o Modelo B demonstrou muito mais confiança.

Agora imagine o contrário:

### Modelo C

Diz:

> “Este e-mail tem **1% de chance de ser spam**.”

Mas ele era spam.

O erro é muito mais grave porque o modelo estava **extremamente confiante na previsão errada**.

É justamente esse tipo de comportamento que o Log-Loss penaliza fortemente.

<br><br>

# 13. Log-Loss em várias amostras

Para um conjunto de $N$ observações, o Log-Loss médio é:

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

Quanto menor o valor:

$$
\boxed{\text{melhor}}
$$

Um modelo que fornece probabilidades bem calibradas tende a apresentar um Log-Loss menor.

<br><br>

# 14. Uma comparação simples entre as métricas

| Métrica | Pergunta principal |
|---|---|
| **Accuracy** | Quantos casos foram classificados corretamente? |
| **ROC-AUC** | O modelo consegue separar e ordenar bem as classes? |
| **Precision** | Dos casos classificados como spam, quantos realmente são spam? |
| **Recall** | Dos spams existentes, quantos foram encontrados? |
| **PR-AUC / AP** | Como o modelo se comporta entre Precision e Recall para a classe positiva? |
| **Log-Loss** | As probabilidades previstas são boas ou o modelo erra com muita confiança? |
| **Matriz de Confusão** | Quantos VP, FP, FN e VN ocorreram no threshold escolhido? |

---

# 15. A ideia central

Não estamos avaliando exatamente a mesma coisa com todas essas métricas.

**ROC-AUC:**

$$
\boxed{\text{discriminação / ordenação dos scores}}
$$

**PR:**

$$
\boxed{\text{Precision} \times \text{Recall}}
$$

**AP:**

$$
\boxed{\text{resumo da curva Precision-Recall}}
$$

**Log-Loss:**

$$
\boxed{\text{qualidade das probabilidades e penalização do erro confiante}}
$$

**Matriz de Confusão:**

$$
\boxed{VP,\ FP,\ FN,\ VN\ no\ threshold\ escolhido}
$$

Por isso, um **ROC-AUC alto, como 0,90, não significa automaticamente que o modelo seja excelente para a aplicação de spam**.

Ele pode estar fazendo uma boa ordenação das probabilidades, mas ainda apresentar uma Precision baixa quando precisamos efetivamente classificar os e-mails.

Da mesma forma, um modelo pode acertar a classe final, mas atribuir probabilidades muito ruins. Nesse caso, o **Log-Loss** pode revelar problemas que a acurácia ou o ROC-AUC não mostram.

> **ROC-AUC ajuda a responder se o modelo sabe separar e ordenar as classes. PR/AP ajuda a entender se essa capacidade é útil para identificar uma classe positiva rara. Log-Loss verifica a qualidade das probabilidades e pune fortemente erros confiantes. A matriz de confusão mostra o que acontece quando escolhemos um threshold específico.**

<br><br>

## 🎯 Regra prática

Para uma classe positiva rara, como spam:

$$
\boxed{
ROC\text{-}AUC
+
PR\text{-}AUC/AP
+
Precision
+
Recall
+
F1
+
Log\text{-}Loss
+
\text{Matriz de Confusão}
}
$$

é uma análise muito mais completa do que observar apenas a acurácia ou apenas o ROC-AUC.

### 📌 Base de comparação

Para a PR, a **prevalência da classe positiva** funciona como uma linha de base:

$$
Baseline_{PR} = \frac{N_{positivos}}{N_{total}}
$$

Para o Log-Loss, uma referência simples é o modelo que prevê sempre a **prevalência da classe positiva**:

$$
p_i = \pi
$$

onde:

$$
\pi = \frac{N_{positivos}}{N_{total}}
$$

Assim, podemos avaliar se o modelo está realmente oferecendo probabilidades melhores do que simplesmente repetir a frequência observada da classe.

<br><br>

# 🎯 16. Qual é o seu objetivo?

A escolha da métrica deve começar pela **pergunta que queremos responder**, e não pela métrica que parece mais conhecida ou mais alta.

| **O que você quer avaliar?** | **Métrica indicada** |
|---|---|
| **Ordenar / comparar modelos sem um limiar fixo** | **ROC-AUC** |
| **Encontrar positivos raros e caros de perder** | **PR-AUC / AP + Recall** |
| **Garantir que os positivos previstos sejam realmente positivos** | **Precision** |
| **Encontrar a maior parte dos positivos** | **Recall** |
| **Equilibrar Precision e Recall** | **F1-score** |
| **Obter probabilidades confiáveis e bem calibradas** | **Log-Loss** |
| **Entender exatamente os acertos e erros em um threshold específico** | **Matriz de Confusão** |

### 🧭 Como pensar sobre a escolha?

#### 1. Quero comparar a capacidade de discriminação dos modelos

> **“Qual modelo consegue ordenar melhor os casos positivos acima dos negativos?”**

Use:

$$
\boxed{\text{ROC-AUC}}
$$

É especialmente útil quando ainda não existe um threshold fixo definido.

<br><br>

#### 2. Quero encontrar uma classe positiva rara

> **“Preciso encontrar os positivos, mas não quero gerar uma quantidade enorme de falsos positivos.”**

Use:

$$
\boxed{\text{PR-AUC / AP + Recall}}
$$

Isso é particularmente importante em problemas como:

- Spam
- Fraude
- Detecção de anomalias
- Falhas raras
- Doenças raras
- Eventos de alto risco

Quando a classe positiva é rara, a curva Precision-Recall permite observar melhor o equilíbrio entre:

$$
\text{Precision} \leftrightarrow \text{Recall}
$$

<br><br>

#### 3. Quero probabilidades confiáveis

> **“Quando o modelo diz 90%, essa probabilidade realmente representa aproximadamente 90% de chance?”**

Nesse caso, precisamos avaliar a **qualidade das probabilidades**, e não apenas se a classe final foi acertada.

Uma métrica importante é:

$$
\boxed{\text{Log-Loss}}
$$

O Log-Loss penaliza fortemente **erros confiantes**.

Por exemplo:

$$
-\log(0.90)\approx0.11
$$

é uma penalidade pequena quando a previsão está correta e confiante.

Já:

$$
-\log(0.10)\approx2.30
$$

é uma penalidade muito maior quando o modelo atribui uma probabilidade muito baixa à classe que realmente ocorreu.

<br><br>

## 🧠 A métrica depende da pergunta

Não existe uma métrica universalmente melhor.

A escolha depende do que queremos otimizar:

$$
\boxed{\text{Objetivo} \rightarrow \text{Métrica} \rightarrow \text{Threshold} \rightarrow \text{Decisão}}
$$

Por isso:

> **Comece sempre pela pergunta. A métrica é uma decisão, não um padrão.**

### 📌 Exemplo: filtro de spam

Se a pergunta for:

> **“O modelo consegue ordenar os e-mails de acordo com a probabilidade de serem spam?”**

→ **ROC-AUC**

Se a pergunta for:

> **“O modelo consegue encontrar os poucos spams sem bloquear muitos e-mails legítimos?”**

→ **PR-AUC / AP + Recall + Precision**

Se a pergunta for:

> **“As probabilidades de spam fornecidas pelo modelo são confiáveis?”**

→ **Log-Loss + calibração**

Se a pergunta for:

> **“O que realmente aconteceu depois de escolher o threshold 0,50?”**

→ **Matriz de Confusão + Precision + Recall + F1-score**

<br><br>

## 🔑 Regra final

$$
\boxed{
\text{Não escolha a métrica primeiro.}
}
$$

$$
\boxed{
\text{Defina primeiro o que significa “bom” para o problema.}
}
$$

**A métrica deve responder à pergunta do negócio ou da aplicação — e não o contrário.**
