# Guia de Métricas de Avaliação e Documentação no GitHub

Este guia apresenta de forma clara e estruturada os conceitos essenciais para a avaliação de modelos de classificação em ciência de dados e a criação de documentações de projetos no GitHub utilizando Markdown e LaTeX.

<br><br>

## 1. Avaliação de Modelos de Classificação

Para determinar a qualidade e a eficiência de um classificador, analisa-se a distribuição de suas predições em relação aos valores reais observados.

### 1.1. A Matriz de Confusão

A Matriz de Confusão categoriza os resultados do modelo em quatro cenários fundamentais:

| | Predição Positiva (Modelo disse Sim) | Predição Negativa (Modelo disse Não) |
| :--- | :--- | :--- |
| **Classe Real Positiva** | **Verdadeiro Positivo (VP)** - Classificação correta da classe positiva. | **Falso Negativo (FN)** - Omissão da classe positiva. |
| **Classe Real Negativa** | **Falso Positivo (FP)** - Alarme falso da classe positiva. | **Verdadeiro Negativo (VN)** - Classificação correta da classe negativa. |

*   **Verdadeiro Positivo ($VP$):** Instância positiva prevista corretamente.
*   **Verdadeiro Negativo ($VN$):** Instância negativa prevista corretamente.
*   **Falso Positivo ($FP$):** Instância negativa incorretamente prevista como positiva.
*   **Falso Negativo ($FN$):** Instância positiva incorretamente prevista como negativa.

<br><br>

### 1.2. Principais Métricas de Desempenho

A partir da Matriz de Confusão, derivam-se as métricas fundamentais de desempenho.

#### Acurácia

A acurácia mede a proporção de todas as classificações corretas em relação ao total de amostras.

$$
\text{Acurácia} = \frac{VP + VN}{VP + VN + FP + FN}
$$

> Em conjuntos de dados desbalanceados, a acurácia isolada pode mascarar o baixo desempenho na classe minoritária.

#### Precisão

A precisão mede a proporção de instâncias verdadeiramente positivas entre todas aquelas rotuladas como positivas pelo modelo.

$$
\text{Precisão} = \frac{VP}{VP + FP}
$$

A precisão é prioritária em cenários onde o custo do **Falso Positivo ($FP$)** é elevado.

#### Recall (Sensibilidade)

O recall mede a proporção de instâncias positivas reais que foram identificadas corretamente pelo modelo.

$$
\text{Recall} = \frac{VP}{VP + FN}
$$

O recall é prioritário em aplicações onde o custo do **Falso Negativo ($FN$)** é elevado.

#### F1-Score

O F1-Score é a média harmônica entre a precisão e o recall, garantindo equilíbrio na avaliação.

$$
F_1 = 2 \cdot \frac{\text{Precisão} \cdot \text{Recall}}{\text{Precisão} + \text{Recall}}
$$

<br><br>

### 1.3. A Métrica F-beta ($F_\beta$)

O $F_\beta$ permite ajustar a importância relativa entre a precisão e o recall por meio do parâmetro $\beta$.

$$
F_\beta = (1 + \beta^2) \cdot \frac{\text{Precisão} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precisão}) + \text{Recall}}
$$

*   **$\beta = 1$ ($F_1$-Score):** Pesos iguais para precisão e recall.
*   **$\beta > 1$ (ex: $F_2$):** Confere maior peso ao **recall**, priorizando a redução de Falsos Negativos ($FN$).
*   **$\beta < 1$ (ex: $F_{0.5}$):** Confere maior peso à **precisão**, priorizando a redução de Falsos Positivos ($FP$).

<br><br>

### 1.4. Limiar de Decisão (Decision Threshold)

O limiar de decisão define o valor de probabilidade acima do qual a classe positiva é atribuída.

*   **Reduzir o limiar:** Aumenta o **recall** e reduz os Falsos Negativos, mas reduz a **precisão**.
*   **Aumentar o limiar:** Aumenta a **precisão** e reduz os Falsos Positivos, mas reduz o **recall**.

