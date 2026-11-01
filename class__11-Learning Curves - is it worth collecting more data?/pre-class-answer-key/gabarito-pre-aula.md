# CD1 — Aula 11: Curvas de aprendizado

## Gabarito

Registro do gabarito apresentado na conversa. As alternativas seguem a ordem informada anteriormente e não foram verificadas novamente no formulário.

| Questão | Alternativa | Resposta |
| --- | --- | --- |
| 1 | 2ª | Na curva de validação varia um hiperparâmetro; na curva de aprendizado varia o tamanho do conjunto de treino. |
| 2 | 3ª | O score de treino cai um pouco, o de validação sobe e a distância entre eles diminui. |
| 3 | 1ª | A linha de validação, pois estima o desempenho em dados novos. |
| 4 | 4ª | Viés alto (subajuste): falta capacidade ao modelo. |
| 5 | 1ª | Variância alta (sobreajuste): mais dados ou regularização devem ajudar. |
| 6 | 2ª | A diferença entre treino e validação diminui e a validação ainda sobe: vale coletar mais dados. |
| 7 | 3ª | Não priorizar mais dados: as curvas estão próximas e estabilizadas num patamar baixo, indicando viés alto. |
| 8 | 4ª | Saturação: aumentar a quantidade de dados traz ganhos cada vez menores. |
| 9 | 1ª | São 40 treinos: 8 tamanhos × 5 dobras, contra 5 no cross_val_score. |
| 10 | 2ª | Evitar vazamento: ajustar o pré-processamento somente no treino de cada dobra e tamanho. |
| 11 | 3ª | O primeiro ponto usa pouquíssimos exemplos, tornando o resultado pouco confiável. |
| 12 | 4ª | A melhora não está demonstrada apenas pelas médias, diante da grande sobreposição das faixas. |
| 13 | 1ª | Mais dobras deixam uma fração maior dos dados disponível para cada treino. |
| 14 | 2ª | Usar modelo mais rico, novos atributos ou menos regularização. |
| 15 | 3ª | Observar a inclinação final da validação: se ainda sobe, mais dados podem render; se estabilizou, o ganho tende a ser pequeno. |
| 16 | 4ª | Pedir mais dados quando as curvas estão juntas num patamar baixo, indicando viés alto. |

## Observação sobre a questão 16

O item acima reproduz o gabarito anterior. Sem o enunciado completo, não é possível determinar se a questão pede uma conduta correta ou a identificação de um erro.

## Sugestão para o campo de dúvida

> Como podemos verificar se a melhora na validação ao aumentar os dados é real ou apenas variação entre as dobras, especialmente quando as faixas de desvio-padrão se sobrepõem?

## Origem

Material transcrito da resposta anterior nesta conversa, referente ao formulário “CD1 — Aula 11: Curvas de aprendizado”. Não constitui gabarito oficial.
