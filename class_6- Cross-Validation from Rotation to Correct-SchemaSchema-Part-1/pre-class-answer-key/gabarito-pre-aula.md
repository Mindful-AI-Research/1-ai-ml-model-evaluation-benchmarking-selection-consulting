Gabarito  - Validação Cruzada

1) "Avaliar com uma única divisão treino/teste é como jogar a moeda." Por quê?

A)  Porque o modelo responde aleatoriamente

B)  Porque a acurácia é sempre 50%

C)  Porque o resultado pode variar dependendo da divisão dos dados; por
isso usamos várias divisões e tiramos uma média (validação cruzada)
✅

D)  Porque não precisamos de conjunto de teste

Resposta: C

<br>

2) No código cross_val_score(modelo, X, y, cv=5), o que cv=5 define?

A)  Cinco modelos diferentes

B)  A divisão dos dados em 5 partes, alternando os conjuntos de treino e
validação automaticamente ✅

C)  Cinco épocas de treinamento

D)  Cinco métricas diferentes

Resposta: B

<br>

3) Quais dados devem ser passados para cross_val_score?

A)  Apenas o bloco de treino/validação, mantendo o teste final separado
para ser usado uma única vez no final ✅

B)  Todo o dataset, incluindo o teste final

C)  Apenas o conjunto de teste

D)  Apenas uma amostra dos dados

Resposta: A

<br>

4) Em uma validação cruzada de 5 folds, as acurácias foram [0.80, 0.90, 0.70, 0.85, 0.75]. Qual é a média?

A)  0.75

B)  0.80 ✅

C)  0.85

D)  0.90

Resposta: B

<br>

5) Qual é uma forma mais honesta de apresentar o resultado da validação cruzada?

A)  Apenas a maior acurácia

B)  Apenas a menor acurácia

C)  A média das métricas junto com o desvio-padrão (média ±

desvio-padrão) ✅
D)  Apenas o primeiro fold

Resposta: C

<br>

6) O que o StratifiedKFold procura manter em cada fold?

A)  O mesmo número total de exemplos

B)  A mesma ordem dos dados

C)  A mesma quantidade de features

D)  Aproximadamente a mesma proporção entre as classes ✅

Resposta: D

<br>

7) Temos 100 exemplos, sendo apenas 3 positivos, e queremos usar 5 folds estratificados. O que acontece?

A)  Todos os folds terão exatamente o mesmo número de positivos

B)  Pelo menos 2 folds ficarão sem exemplos positivos; o número de folds
não deve ser maior que o número de exemplos da classe minoritária
✅

C)  O modelo ficará automaticamente balanceado

D)  A acurácia será necessariamente 100%

Resposta: B

<br>

8) O que é Leave-One-Out Cross-Validation (LOO)?

A)  Um caso de k-fold em que k = n; cada exemplo é usado sozinho como
teste uma vez ✅

B)  Uma validação sem conjunto de teste

C)  Uma validação com apenas um fold

D)  Uma técnica que elimina o treinamento

Resposta: A

<br>

9) Em LOO com n=10, quantos treinamentos são realizados?

A)  1

B)  5

C)  10; em cada rodada há apenas um exemplo no teste, então a média dos
resultados é mais informativa ✅

D)  100

Resposta: C

<br>


10) O que o RepeatedKFold faz?

A)  Usa sempre exatamente a mesma divisão

B)  Repete o k-fold várias vezes, usando diferentes divisões aleatórias,

e considera todos os resultados ✅

C)  Remove os folds com pior resultado

D)  Treina apenas uma vez

Resposta: B

<br>

11) Por que é um problema ter exames do mesmo paciente tanto no treino quanto no teste?

A)  Porque aumenta o número de features

B)  Porque reduz a quantidade de dados

C)  Porque impede o modelo de aprender

D)  Porque o modelo pode memorizar características daquele paciente e
gerar uma avaliação artificialmente alta, mas falhar em pacientes
novos ✅

Resposta: D

<br>

12) O que o GroupKFold garante?

A)  Todas as linhas pertencentes ao mesmo grupo ficam no mesmo lado da
divisão, treino ou teste ✅

B)  Cada grupo aparece em todos os folds

C)  Os grupos são escolhidos aleatoriamente dentro de cada fold

D)  O número de features é igual em todos os grupos

Resposta: A

<br>

13) Para dados que dependem de datas, qual é a estratégia mais adequada?

A)  Treinar com dados do passado e testar com dados do futuro, evitando
embaralhar aleatoriamente os dados ✅

B)  Embaralhar todos os dados antes da divisão

C)  Usar apenas os dados mais recentes no treino

D)  Misturar passado e futuro em todos os folds

Resposta: A

<br>

14) Se treinamos com dados de 2000--2015 e usamos 2016--2021 como holdout temporal, o que isso representa?

A)  Um erro de validação

B)  Um exemplo de LOO


C)  Uma única divisão temporal; o TimeSeriesSplit permite repetir
divisões cronológicas para obter uma avaliação mais estável ✅

D)  Um exemplo de GroupKFold

Resposta: C

<br>

15) Por que colocar o pré-processamento dentro de um Pipeline?

A)  Para deixar o código maior

B)  Para evitar o treinamento do modelo

C)  Para usar mais features

D)  Para que o pré-processamento seja ajustado novamente em cada fold
usando apenas os dados de treino daquele fold, evitando vazamento ✅

Resposta: D

<br>

16) O que acontece quando usamos StandardScaler em todo o dataset antes da validação cruzada?

A)  Pode ocorrer vazamento de dados, porque média e desvio-padrão são
calculados também usando informações das partes que deveriam
funcionar como teste em cada fold, podendo inflar o resultado ✅

B)  O modelo fica automaticamente mais robusto

C)  A validação cruzada deixa de existir

D)  O conjunto de teste desaparece

Resposta: A

<br>

17) Pergunta aberta

Uma dúvida que fiquei: na prática, como a gente sabe se o modelo está
realmente generalizando bem e não só decorando os dados do treino? E
quando os dados mudam com o tempo, como a gente percebe que está na hora
de reavaliar ou treinar o modelo de novo?
