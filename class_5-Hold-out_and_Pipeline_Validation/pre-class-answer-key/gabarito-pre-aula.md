1. Por que NÃO se deve avaliar um modelo usando os mesmos dados em que ele foi treinado?

* A) porque o treinamento demora mais
* B) porque ele memoriza os dados, e a nota não mede a generalização (fica alta demais) ✅
* C) porque o conjunto de teste precisa ser maior que o conjunto de treinamento
* D) porque, caso contrário, não é possível usar árvores

Resposta: B) porque ele memoriza os dados, e a nota não mede a generalização (fica alta demais).

#

2. Em um hold-out com três conjuntos, qual é o papel de cada conjunto?

* A) o treinamento ajusta o modelo, a validação não é utilizada, e o teste escolhe o modelo
* B) os três conjuntos são usados para treinamento
* C) o treinamento ajusta o modelo, a validação ajuda nas decisões (modelo/limiar), e o teste fornece a nota final sem ser utilizado nas decisões ✅
* D) a validação treina o modelo e o teste faz os ajustes

Resposta: C) o treinamento ajusta o modelo, a validação ajuda nas decisões (modelo/limiar), e o teste fornece a nota final sem ser utilizado nas decisões.

#

3. Como obter os conjuntos de treinamento / validação / teste na prática?

* A) duas divisões sequenciais: primeiro separa-se o conjunto de teste; depois divide-se o restante em treinamento e validação ✅
* B) uma única divisão em três partes iguais e sempre selecionadas aleatoriamente
* C) não é possível ter três conjuntos
* D) sorteando aleatoriamente as linhas a cada época

Resposta: A) duas divisões sequenciais: primeiro separa-se o conjunto de teste; depois divide-se o restante em treinamento e validação.

#

4. Para que serve o `stratify=y` no `train_test_split`?

* A) para embaralhar melhor os dados
* B) para padronizar as variáveis
* C) para acelerar o treinamento
* D) para manter a mesma proporção das classes nos conjuntos de treinamento e teste ✅

Resposta: D) para manter a mesma proporção das classes nos conjuntos de treinamento e teste.

#

5. Para que serve o `random_state` (a semente)?

* A) para melhorar a acurácia do modelo
* B) para tornar a divisão aleatória reprodutível — mesma semente, mesma divisão ✅
* C) para aumentar o tamanho do conjunto de teste
* D) para evitar valores ausentes

Resposta: B) para tornar a divisão aleatória reprodutível — mesma semente, mesma divisão.

#

6. "Avaliar com uma única divisão treino/teste é como jogar uma moeda." O que isso significa?

* A) que o modelo escolhe a resposta aleatoriamente
* B) que a acurácia é sempre 50%
* C) que a nota varia dependendo da sorte da divisão; a solução é calcular a média de várias divisões (validação cruzada) ✅
* D) que não se deve utilizar um conjunto de teste

Resposta: C) que a nota varia dependendo da sorte da divisão; a solução é calcular a média de várias divisões (validação cruzada).

#

7. Padronizar (`StandardScaler`) usando o conjunto INTEIRO antes de separar os conjuntos de treinamento e teste é:

* A) vazamento de dados — o conjunto de teste "espia" os dados de treinamento porque sua média e seu desvio padrão estão incluídos ✅
* B) a abordagem recomendada
* C) irrelevante para o resultado
* D) necessário para o `OneHotEncoder`

Resposta: A) vazamento de dados — o conjunto de teste "espia" os dados de treinamento porque sua média e seu desvio padrão estão incluídos.

#

8. Qual é a forma correta de aplicar o `StandardScaler`?

* A) `fit_transform` no conjunto de treinamento e `fit_transform` no conjunto de teste
* B) `fit` no conjunto de teste e `transform` no conjunto de treinamento
* C) `fit` no conjunto de treinamento; `transform` (sem `fit`) no conjunto de teste ✅
* D) `fit_transform` no conjunto inteiro

Resposta: C) `fit` no conjunto de treinamento; `transform` (sem `fit`) no conjunto de teste.

#

9. Para que serve o `ColumnTransformer`?

* A) para treinar vários modelos ao mesmo tempo
* B) para aplicar uma transformação diferente a cada tipo de coluna (numérica × categórica) de uma só vez ✅
* C) para dividir os conjuntos de treinamento e teste
* D) para calcular a acurácia

Resposta: B) para aplicar uma transformação diferente a cada tipo de coluna (numérica × categórica) de uma só vez.

#

10. Por que um `Pipeline` (pré-processamento + modelo) evita vazamento durante a validação cruzada?

* A) porque utiliza menos memória
* B) porque remove os valores ausentes
* C) porque escolhe o `k` automaticamente
* D) porque refaz o pré-processamento DENTRO de cada dobra, usando apenas os dados de treinamento daquela dobra ✅

Resposta: D) porque refaz o pré-processamento DENTRO de cada dobra, usando apenas os dados de treinamento daquela dobra.

#

11. Quando os dados contêm datas, qual é a estratégia correta de divisão?

* A) dividir por tempo: treinar com o passado e testar com o futuro (não embaralhar aleatoriamente) ✅
* B) embaralhar aleatoriamente as linhas como sempre
* C) testar no passado e treinar no futuro
* D) usar apenas o mês mais antigo

Resposta: A) dividir por tempo: treinar com o passado e testar com o futuro (não embaralhar aleatoriamente).

#

12. Você tem dados de vendas de 2000 a 2025 e quer fazer uma previsão para dezembro de 2026 (já possui os dados de janeiro a novembro de 2026). Em relação ao treinamento:

* A) remover TODOS os meses de dezembro de 2000–2025 do conjunto de treinamento
* B) treinar somente com os dados de 2026
* C) manter os dezembros anteriores no conjunto de treinamento (eles ensinam a sazonalidade); nunca usar dados posteriores ao ponto de previsão ✅
* D) usar apenas dados de dezembro no treinamento

Resposta: C) manter os dezembros anteriores no conjunto de treinamento (eles ensinam a sazonalidade); nunca usar dados posteriores ao ponto de previsão.

#

13. Para estimar se o modelo de série temporal funciona (sem ainda ter o valor real de 2026), qual é a melhor prática?

* A) nunca testar; confiar nos resultados do treinamento
* B) manter os anos mais recentes conhecidos separados como conjunto de teste (por exemplo, treinar de 2000–2023 e prever 2024–2025) e, se aprovado, treinar novamente usando todos os dados ✅
* C) testar em anos selecionados aleatoriamente
* D) usar o mesmo ano para treinamento e teste

Resposta: B) manter os anos mais recentes conhecidos separados como conjunto de teste (por exemplo, treinar de 2000–2023 e prever 2024–2025) e, se aprovado, treinar novamente usando todos os dados.

#

14. Para que serve um "global holdout" (holdout de produto)?

* A) para ajustar os parâmetros do modelo
* B) para padronizar as variáveis
* C) para substituir o conjunto de teste do modelo
* D) para manter um grupo de usuários fora de TODAS as mudanças, medindo o impacto real acumulado no negócio ✅

Resposta: D) para manter um grupo de usuários fora de TODAS as mudanças, medindo o impacto real acumulado no negócio.

#

15. Qual é a diferença entre um teste A/B e um global holdout?

* A) o teste A/B mede UMA mudança (curto prazo); o global holdout mede o efeito de TUDO (longo prazo) ✅
* B) são exatamente a mesma coisa
* C) o teste A/B não utiliza um grupo de controle
* D) o global holdout serve para treinar o modelo

Resposta: A) o teste A/B mede UMA mudança (curto prazo); o global holdout mede o efeito de TUDO (longo prazo).

#

16. Na produção do modelo, o que são skew e drift?

* A) o mesmo problema com nomes diferentes
* B) erros de digitação no código
* C) skew = treinamento ≠ produção AGORA (defesa: usar o mesmo Pipeline); drift = os dados mudam AO LONGO DO TEMPO (defesa: monitorar e treinar novamente) ✅
* D) skew acontece ao longo do tempo e drift acontece durante a implantação

Resposta: C) skew = treinamento ≠ produção AGORA (defesa: usar o mesmo Pipeline); drift = os dados mudam AO LONGO DO TEMPO (defesa: monitorar e treinar novamente).

#

17. Você ficou com alguma dúvida sobre a aula de hoje?

Uma dúvida que ainda tenho é:

Na prática, como a gente sabe se o modelo está realmente generalizando bem e não apenas memorizando os dados de treinamento? E quando os dados mudam ao longo do tempo, como podemos perceber que chegou o momento de reavaliar ou treinar o modelo novamente?

Isso é algo que eu gostaria de entender melhor na prática, principalmente como identificar quando um modelo deixou de generalizar bem e quando é necessário fazer um novo treinamento.
