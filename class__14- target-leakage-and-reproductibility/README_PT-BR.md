# Aula 14 — Vazamento de Alvo e Reprodutibilidade

Instituição: Pontifícia Universidade Católica de São Paulo (PUC-SP)  
Faculdade: FACEI — Departamento de Ciência da Computação  
Curso: Bacharelado em IA Centrada no Humano & Ciência de Dados · 6º semestre · 2026  
Disciplina: Ciência de Dados & Machine Learning — Avaliação, Benchmarking e Seleção de Modelos  
Professor: Giovani Giulio Tristão Thibes Vieira  
Documentação: Fabiana Campanari

[Mindful AI Research](https://github.com/Mindful-AI-Research)

> Numeração alterada a pedido da autora. O handout original identifica este conteúdo como Aula 13. Metadados acadêmicos fornecidos pela autora.

## 1. Visão geral

Vazamento de alvo, vazamento temporal e reprodutibilidade: avaliar se uma métrica elevada representa uma tarefa preditiva válida.

## 2. A pergunta decisiva

Eu teria esta coluna, deste jeito, no momento da previsão? Se não, exclua. Em caso de dúvida, investigue a procedência. A disponibilidade importa mesmo quando a coluna quase não altera a acurácia.

## 3. Prever reprovação na semana 6

Alvo: reprovou. Disponíveis: curso, semestre_ingresso, nota_P1 (semana 5), faltas_ate_semana_6. Investigar frequentou_monitoria: a participação ocorreu antes ou depois da semana 6?

## 4. Entradas indisponíveis

Excluir faltas_no_semestre, nota_P2, fez_substitutiva, media_final e sala_prova_final. Contêm informação posterior ou representam de perto o resultado. O caso é de reprovação, não uma tarefa demonstrada de evasão.

## 5. Experimento relatado

Base simulada: 1.000 estudantes; validação cruzada de 10 dobras. Acurácia com atributos do fim do semestre: 0,953; somente semana 6: 0,744; prever sempre “não reprovou”: 0,628. Resultados relatados no handout, não reproduzidos independentemente.

## 6. Um atributo por vez

Adicionados às entradas da semana 6: nota_P2 → 0,861; faltas_no_semestre → 0,746; fez_substitutiva → 0,806; media_final → 0,955; sala_prova_final → 0,745. A sala da prova continua inválida apesar do impacto desprezível.

## 7. Validação temporal

Em previsões futuras, embaralhar datas pode colocar observações futuras no treino. O handout propõe validação para frente com TimeSeriesSplit: treinar em observações anteriores e avaliar posteriores. A linha do tempo é conceitual.

## 8. Agregados podem vazar

Ao prever no dia 3, a média do mês inteiro inclui dias futuros. Uma média restrita aos dias 1 e 2 respeita o corte ilustrado. Uma divisão válida não torna todos os atributos válidos.

## 9. Reprodutibilidade

Mantenha a divisão aleatória fixa nas comparações repetidas. O handout apresenta train_test_split(X, y, test_size=0.2, random_state=42). O trecho pressupõe importações e dados existentes. A semente não deve ser buscada para melhorar a métrica.

## 10. Audite → Remova → Documente

Defina alvo e momento da previsão; audite disponibilidade; remova entradas indisponíveis; confira agregados temporais; escolha validação adequada; fixe a semente; salve o Pipeline completo e prepare um model card. São recomendações, não artefatos salvos verificados.

## 11. Evidências e limitações

Fonte principal: 2-Handout.pdf, datado de 10/11/2026, identificado como Aula 13 no original. Esta edição usa Aula 14 a pedido da autora. O original tem 21 páginas e a numeração passa de 19 para 23. Nenhum conteúdo ausente foi reconstruído. Código, versões do ambiente, resultados holdout e implantação não são comprovados apenas pelo PDF.

## Referências e escopo

Fonte principal: `2-Handout.pdf`. Referências mencionadas no handout: Kapoor e Narayanan, Patterns, 2023; H2O.ai (antibiótico); IBM (estorno); Larsen e Becker, 2019 (crédito). Dados bibliográficos completos não foram verificados independentemente. Os notebooks complementares estão fora do escopo de evidências de implementação deste documento.
