# [CD1 — Aula 12: Limiar de decisão e calibração]()

## [Gabarito]()

> Respostas esperadas pelo formulário, conforme o enunciado fornecido. A numeração das alternativas segue a ordem de cima para baixo (1 a 4). Este documento reproduz o gabarito esperado da aula, não uma revisão técnica das afirmações.

## [Consulta rápida]()

| Questão | Alternativa |
| :--- | :---: |
| 1 | 3 |
| 2 | 2 |
| 3 | 4 |
| 4 | 1 |
| 5 | 2 |
| 6 | 3 |
| 7 | 1 |
| 8 | 4 |
| 9 | 2 |
| 10 | 3 |
| 11 | 1 |
| 12 | 4 |
| 13 | 2 |
| 14 | 1 |
| 15 | 3 |
| 16 | 2 |
| 17 | 4 |

<br><br>

## [1. Para comparar os quatro candidatos, a aula usou ROC-AUC em vez de acurácia. Por quê?]()

### [Resposta correta — alternativa 3]()

> porque acurácia é acurácia NO CORTE 0,50 — comparar por ela seria escolher o modelo já com um corte embutido, justo o corte que a aula vai mostrar que está errado

<br><br>

## [2. Logística 0,960 ± 0,014 e floresta 0,945 ± 0,025, nas mesmas 10 dobras. A faixa de empate é ≥ 0,946. O que se faz?]()

### [Resposta correta — alternativa 2]()

> as duas são equivalentes na prática (a floresta fica fora por um milésimo, com as barras se sobrepondo), e fica a logística por ser a mais simples e explicável

<br><br>

## [3. A árvore de profundidade 4 deu 0,803 ± 0,135. O que esse desvio enorme está avisando?]()

### [Resposta correta — alternativa 4]()

> que o modelo muda de ideia a cada dobra: com 90 positivos em 595 linhas, trocar a amostra troca a árvore — o 0,803 é uma média de resultados que discordam entre si

<br><br>

## [4. O modelo devolve 0,80 para um paciente. O que exatamente esse número afirma?]()

### [Resposta correta — alternativa 1]()

> que entre todos os pacientes a quem o modelo deu ~0,80, cerca de 80% deveriam ser de risco alto — é uma afirmação que dá para conferir contra o que aconteceu

<br><br>

## [5. O que `modelo.predict(X)` faz, num problema de duas classes?]()

### [Resposta correta — alternativa 2]()

> compara a probabilidade com 0,50: `pred = (proba >= 0.50).astype(int)` devolve exatamente o mesmo vetor

<br><br>

## [6. Em que situação o corte 0,50 é de fato o corte certo?]()

### [Resposta correta — alternativa 3]()

> quando os dois tipos de erro custam igual E as classes estão equilibradas — nenhuma das duas condições vale nos nossos pacientes

<br><br>

## [7. Na tabela de cortes, do 0,15 ao 0,70 a precisão vai de 0,567 a 0,909 e o recall vai de 0,889 a 0,556. Qual é a leitura?]()

### [Resposta correta — alternativa 1]()

> subir o corte troca um erro pelo outro: não existe corte que melhore a precisão e o recall ao mesmo tempo, e por isso a escolha precisa de um critério externo

<br><br>

## [8. Com FN valendo 5 FP, o corte 0,50 custa 184 e o mínimo da curva, em 0,14, custa 103. Confira a conta do 0,50 e diga o que se ganha.]()

### [Resposta correta — alternativa 4]()

> 184 = 19 + 33·5 ; o 0,14 economiza 44%, sem treinar nada de novo

<br><br>

## [9. A área diz: "tenho de pegar pelo menos 90% dos casos". Como isso vira um limiar?]()

### [Resposta correta — alternativa 2]()

> varra a curva e pegue o MAIOR limiar que ainda cumpre a meta — assim você maximiza a precisão dentro da restrição; na nossa base, 0,148, com recall 0,900 e precisão 0,570

<br><br>

## [10. Qual pergunta precisa ser respondida ANTES de escolher qualquer limiar?]()

### [Resposta correta — alternativa 3]()

> quanto custa, para o cliente, cada tipo de erro

<br><br>

## [11. Na faixa de 0,50 a 1,00, a floresta prometeu 0,677 e aconteceu 0,806. O que isso significa — e por quê?]()

### [Resposta correta — alternativa 1]()

> ela subestima justamente os piores casos; 0,677 é o maior número que ela consegue dizer, porque é a média de 200 árvores e seriam precisas ~190 concordando para chegar a 0,95

<br><br>

## [12. Por que a logística sai calibrada "de fábrica" neste problema?]()

### [Resposta correta — alternativa 4]()

> porque ela é treinada minimizando o log-loss, que é justamente a métrica que pune probabilidade desonesta — ela foi otimizada para isso

<br><br>

## [13. Sobre o reliability diagram (previsto no eixo x, observado no eixo y), é CORRETO afirmar:]()

### [Resposta correta — alternativa 2]()

> não há corte nenhum na conta: separa-se por faixa de probabilidade e conta-se a fração de positivos em cada faixa; a diagonal é a honestidade perfeita

<br><br>

## [14. O número do seu modelo saiu torto. Qual é a primeira coisa a fazer?]()

### [Resposta correta — alternativa 1]()

> ver se existe outro candidato que ordena igual e já é honesto — se existe, use ele e acabou: nenhuma etapa nova, nenhum dado reservado

<br><br>

## [15. O corte de custo mínimo da floresta é 0,23; o da logística é 0,14; e na floresta, depois de corrigir a escala, o 0,23 vira 0,11. Qual é a consequência prática?]()

### [Resposta correta — alternativa 3]()

> o corte vive na escala do modelo: trocar de modelo ou corrigir a escala obriga a REVARRER o limiar — a ordem é escolher, conferir o número e só então cortar

<br><br>

## [16. Em qual destas situações a honestidade da probabilidade NÃO é obrigatória?]()

### [Resposta correta — alternativa 2]()

> quando você só precisa ordenar a fila — pegar os 100 pacientes de maior risco para ligar exige ordem certa, não valor honesto, e o ROC-AUC já respondeu isso

<br><br>

## [17. No teste, aberto uma vez: no corte 0,50 a validação previa 0,309 por paciente e o teste deu 0,310; no corte escolhido, 0,173 previsto contra 0,137 entregue. Como se lê isso?]()

### [Resposta correta — alternativa 4]()

> a linha do 0,50 é o controle e mostra que a validação estima bem (0,309 contra 0,310); já o 0,173 → 0,137 saiu melhor que o prometido, e isso é sorte do sorteio, não mérito

<br><br>

## [Campo aberto — Ficou alguma dúvida da aula de hoje?]()

Escreva o que não ficou claro (a faixa de empate, o que a probabilidade afirma, o 0,50 por baixo dos panos, a curva de custo, a meta de recall, o previsto × observado, por que a floresta trava em 0,677, trocar de modelo em vez de calibrar, a ordem das três decisões, a auditoria...). Vamos revisar na próxima aula.

### [Sugestão de pergunta relevante]()

> Como podemos verificar se o limiar de custo mínimo escolhido na validação é estável, em vez de ser apenas consequência da amostra, sem usar o conjunto de teste para ajustá-lo?

Este campo é aberto e não possui uma única resposta correta.
