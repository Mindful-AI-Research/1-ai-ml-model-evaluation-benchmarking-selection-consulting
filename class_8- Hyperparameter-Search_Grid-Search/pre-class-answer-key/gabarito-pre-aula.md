
# Gabarito — Ajuste de Hiperparâmetros e GridSearchCV

<br><br>

## 1) Qual destes é um HIPERPARÂMETRO (e não um parâmetro)?

A) Os pesos aprendidos por uma regressão logística

B) **A profundidade máxima de uma árvore de decisão ✅**

C) Os pontos de corte que a árvore escolhe em cada nó

D) O coeficiente angular estimado numa regressão linear

**Resposta:** B — A profundidade máxima (`max_depth`) é definida antes do treinamento e controla como o modelo será treinado.

<br><br>

## 2) A regra de ouro do ajuste de hiperparâmetros é:

A) Treino aprende, teste escolhe, validação confirma
B) Escolher pelo teste e confirmar na validação, que é maior
C) Usar o mesmo conjunto para escolher e para medir, desde que a CV tenha muitas dobras
D) **Treino aprende, VALIDAÇÃO escolhe, TESTE confirma — uma vez só ✅**

**Resposta:** D

<br><br>

## 3) O GridSearchCV é chamado de busca EXAUSTIVA porque:

A) **Testa a grade inteira, sem deixar nenhuma combinação de fora ✅**
B) Roda até o desempenho parar de melhorar
C) Esgota a memória da máquina em grades grandes
D) Sorteia combinações até cobrir a maior parte do espaço

**Resposta:** A

<br><br>

## 4) Na grade do bolo (3 tempos × 3 temperaturas), quantas combinações são testadas e qual venceu?

A) 6 combinações; venceu 30 min × 180 °C
B) 3 combinações; venceu 50 min × 200 °C
C) **9 combinações; venceu 40 min × 180 °C, com nota 9,0 ✅**
D) 9 combinações; venceu 40 min × 200 °C, com nota 7,0

**Resposta:** C

<br><br>

## 5) Com a grade `C ∈ {0,1; 1; 10}` × `class_weight ∈ {None, balanced}` e uma CV de 5 dobras, quantos modelos são TREINADOS?

A) 6, um por combinação
B) 5, um por dobra
C) 11, a soma das opções com as dobras
D) **30 — 3 × 2 = 6 combinações, cada uma avaliada em 5 dobras ✅**

**Resposta:** D

<br><br>

## 6) Acrescentando à mesma grade um terceiro hiperparâmetro com 4 opções, o número de treinos passa de 30 para:

A) 34 — soma-se o novo eixo
B) **120 — cada eixo novo MULTIPLICA o total ✅**
C) 38 — somam-se as 4 opções às 5 dobras
D) 30 — o número de dobras não muda

**Resposta:** B

<br><br>

## 7) A tabela da aula deu estas médias de CV (AP): C=0,1/None 0,61 · C=0,1/balanced 0,68 · C=1/None 0,71 · C=1/balanced 0,79 · C=10/None 0,70 · C=10/balanced 0,74. O que o GridSearchCV guarda em `best_params_` e `best_score_`?

A) **`best_params_ = {C: 1, class_weight: balanced}` e `best_score_ = 0,79` ✅**
B) `best_params_ = {C: 10, class_weight: balanced}` e `best_score_ = 0,74`
C) `best_params_ = {C: 1, class_weight: balanced}` e `best_score_ = 0,71`
D) `best_params_ = a média das seis combinações, 0,705`

**Resposta:** A

<br><br>

## 8) Numa busca sobre um Pipeline, por que a grade é escrita como `{"clf__C": [0.1, 1, 10]}` e não `{"C": [0.1, 1, 10]}`?

A) Porque o underscore duplo indica que o valor é uma lista
B) Porque `"clf"` é o nome da métrica usada no scoring
C) **Porque o prefixo `nome_do_passo__parametro` diz a QUAL passo do Pipeline o hiperparâmetro pertence ✅**
D) Porque o sklearn exige nomes com prefixo em qualquer busca, com ou sem Pipeline

**Resposta:** C

<br><br>

## 9) Num problema com classe rara, definir `scoring="accuracy"` na busca é um erro porque:

A) A acurácia não é aceita pelo GridSearchCV em problemas binários
B) **Prever sempre a classe majoritária já dá acurácia alta — a busca escolheria o modelo que ignora a classe rara ✅**
C) A acurácia só pode ser usada com validação cruzada estratificada
D) A acurácia é sempre menor que o F1, então subestima o modelo

**Resposta:** B

<br><br>

## 10) Em classificação, o cv recomendado na busca é o StratifiedKFold porque ele:

A) É mais rápido que o KFold comum
B) Embaralha os dados, o que o KFold não faz
C) Usa todas as dobras para treino e nenhuma para validação
D) **Preserva a proporção das classes em cada dobra — sem isso a classe rara pode sumir de uma delas ✅**

**Resposta:** D

<br><br>

## 11) Por que o GridSearchCV deve receber o PIPELINE inteiro, e não só o classificador com os dados já pré-processados?

A) Porque o Pipeline deixa a busca mais rápida
B) Porque só o Pipeline aceita o parâmetro scoring
C) **Porque assim o pré-processamento é reajustado dentro de cada dobra, usando só o treino dela — pré-processar antes vaza informação da validação ✅**
D) Porque fora do Pipeline o sklearn não consegue calcular `best_score_`

**Resposta:** C

<br><br>

## 12) A busca terminou com `best_score_ = 0,79` e o teste, tocado uma vez com o `best_estimator_`, deu 0,76. O número que você reporta é:

A) **0,76 — o teste é a palavra final; o `best_score_` serviu para escolher, não para reportar ✅**
B) 0,79 — é a melhor estimativa, obtida com validação cruzada em várias dobras
C) A média dos dois, 0,775
D) 0,79, informando que o teste foi apenas uma verificação

**Resposta:** A

<br><br>

## 13) Por que o `best_score_` tende a ser um pouco OTIMISTA?

A) Porque a validação cruzada sempre superestima o desempenho
B) Porque ele é calculado no conjunto de treino
C) Porque o sklearn arredonda o valor para cima
D) **Porque é o MÁXIMO de muitas medições ruidosas — quem ficou no topo teve, em parte, sorte nas dobras ✅**

**Resposta:** D

<br><br>

## 14) Olhando o `cv_results_`, três combinações empatam dentro do desvio-padrão. O que fazer?

A) Aumentar o número de dobras até uma delas se destacar
B) **Reconhecer o platô e escolher a combinação mais SIMPLES entre as empatadas ✅**
C) Ficar com a de maior média, mesmo que a diferença esteja dentro do desvio
D) Descartar as três e ampliar a grade em volta delas

**Resposta:** B

<br><br>

## 15) Sobre o `busca.best_estimator_`, é CORRETO afirmar:

A) **É o Pipeline com a melhor combinação, JÁ TREINADO — é ele que você leva ao teste ✅**

B) É apenas um dicionário com os melhores valores encontrados

C) É o modelo com os hiperparâmetros certos, mas ainda sem treino: é preciso chamar `fit` antes de usar

D) É a tabela completa com a média e o desvio de cada combinação

**Resposta:** A

<br><br>

## 16) O teste ficou MUITO abaixo do `best_score_` da validação. A leitura mais provável é:

A) O teste está pequeno demais e deve ser ampliado com dados da validação

B) O modelo está subajustado e a grade precisa de valores mais extremos

C) **A busca se sobreajustou — grade grande demais para o tamanho dos dados, e o topo veio de sorte nas dobras ✅**

D) Houve erro de código: os dois números têm de coincidir

**Resposta:** C

<br><br>

## 17) Alguma dúvida ?

**Uma dúvida que fiquei é: como a gente percebe, na prática, que uma busca de hiperparâmetros está começando a “forçar” demais os dados da validação? 
E, quando várias combinações ficam muito próximas, como decidir se vale a pena escolher a mais simples?**
