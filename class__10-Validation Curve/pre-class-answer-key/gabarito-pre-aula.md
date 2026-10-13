# Gabarito --- CD1 \| Aula 10: Curvas de validação

**Quiz da Aula 10 --- respostas corretas com justificativas**

  -------------------------------------------------------------------------------------------
                  Questão      Alternativa correta      Justificativa
  ----------------------- ----------------------------- -------------------------------------
                        1             **B**             O eixo x apresenta os valores de um
                                                        único hiperparâmetro, mantendo os
                                                        demais componentes do modelo
                                                        constantes.

                        2             **B**             A curva de validação permite isolar o
                                                        efeito de um hiperparâmetro. Em uma
                                                        busca com vários eixos, os efeitos
                                                        podem ficar misturados.

                        3             **C**             O score de treino geralmente aumenta
                                                        com a complexidade. Escolher com base
                                                        apenas nele favoreceria modelos cada
                                                        vez mais complexos, mesmo com
                                                        sobreajuste.

                        4             **A**             A árvore de profundidade 1 tem
                                                        acurácia de validação (0,839)
                                                        inferior à linha de base (0,849),
                                                        indicando subajuste.

                        5             **B**             O score de treino aumenta de 0,943
                                                        para 1,000, enquanto a validação cai
                                                        de 0,888 para 0,875. O gap cresce de
                                                        0,055 para 0,125, sinal de
                                                        sobreajuste.

                        6             **C**             O gap é a diferença entre o score de
                                                        treino e o score de validação para o
                                                        mesmo valor do hiperparâmetro.

                        7             **A**             Deve-se maximizar a validação; quando
                                                        os resultados são semelhantes dentro
                                                        da variação observada, prefere-se o
                                                        modelo mais simples.

                        8             **B**             A diferença entre as médias é pequena
                                                        (0,002), dentro do desvio informado.
                                                        A profundidade 2 é mais simples e
                                                        apresenta gap menor (0,018 contra
                                                        0,055).

                        9             **C**             Em um `Pipeline`, usa-se
                                                        `nome_do_passo__nome_do_parâmetro`;
                                                        neste caso, `clf__max_depth`.

                       10             **B**             `validation_curve` devolve duas
                                                        matrizes, com uma linha por valor
                                                        testado e uma coluna por dobra.
                                                        `mean(axis=1)` calcula a média entre
                                                        as dobras para cada valor.

                       11             **D**             O `Pipeline` faz a padronização
                                                        dentro de cada dobra de treino,
                                                        evitando que dados da dobra de
                                                        validação influenciem o
                                                        pré-processamento e tornem a
                                                        avaliação otimista.

                       12             **A**             Como `C` varia por várias ordens de
                                                        grandeza, a escala logarítmica
                                                        permite visualizar melhor os valores
                                                        pequenos e grandes.

                       13             **B**             Com `C` muito pequeno, a
                                                        regularização é muito forte e os
                                                        coeficientes ficam próximos de zero.
                                                        O modelo praticamente não identifica
                                                        positivos e empata com a linha de
                                                        base.

                       14             **C**             A regressão logística é um modelo
                                                        linear e, nessa curva, não apresenta
                                                        sinais de sobreajuste na faixa
                                                        observada. Nem toda curva exibe as
                                                        três regiões clássicas.

                       15             **D**             Em uma base com 15,1% de positivos,
                                                        prever sempre a classe negativa gera
                                                        acurácia de 0,849, mas F1 igual a
                                                        zero. A acurácia pode esconder o
                                                        subajuste.

                       16             **A**             Primeiro, usam-se curvas para
                                                        delimitar faixas promissoras; depois,
                                                        uma busca escolhe a combinação de
                                                        hiperparâmetros. A avaliação final
                                                        deve usar nested cross-validation ou
                                                        um conjunto de teste guardado,
                                                        utilizado uma única vez.
  -------------------------------------------------------------------------------------------

## Resumo das alternativas

**1-B · 2-B · 3-C · 4-A · 5-B · 6-C · 7-A · 8-B · 9-C · 10-B · 11-D ·
12-A · 13-B · 14-C · 15-D · 16-A**


