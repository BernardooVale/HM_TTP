# 🛠️ Scripts do Framework TTP

Este diretório organiza todos os scripts utilitários, orquestradores experimentais e ferramentas de dados do projeto de forma categorizada e intuitiva.

---

## 📁 1. `benchmark/` (Teste Global e Benchmarking)
Destinado a execuções em larga escala comparando todos os solucionadores cadastrados no framework e gerando os relatórios estatísticos consolidados.

- **`executar_super_bateria.py`**: Orquestrador unificado que executa a bateria completa de solucionadores (com ordenação por tamanho, paralelismo e retomada automática) e opcionalmente a análise de sensibilidade ao parâmetro de aluguel $R$.
  - *Modelo padrão:* **TTP1** (benchmark padrão da literatura). O modelo **TTP2** é opcional via `--incluir-ttp2` ou `--modelos TTP1,TTP2`.
- **`normalizar_estatisticas.py`**: Normaliza todas as métricas no intervalo $[0, 1]$ (onde o melhor solver recebe `1.0000` e os demais valores relativos) para comparações justas de desempenho.

---

## 🔥 2. `experimentos_queimada/` (Hipóteses da Heurística da Queimada)
Destinado a validar cientificamente as hipóteses da Heurística da Queimada (temperatura, dissipação, 2-opt, afinidade térmica, backward edges, buracos).

- **`comparar_queimadas.py`**: Exibe no terminal uma tabela comparativa com medalhas (🥇, 🥈, 🥉) confrontando as diferentes variantes da Queimada contra o estado da arte (`heuristic_37_refinada`, `pack_iterative`).
- **`grid_search_paralelo.py`**: Varredura paralela massiva de hiperparâmetros com suporte aos espaços `completo` (2.880 configs), `padrao` (288 configs) ou `rapido` (sem 2-opt), cache de instâncias, retomada automática e análise marginal de parâmetros.
- **`grid_search_queimada.py`**: Implementação clássica de referência de varredura serial em cenários sintéticos.
- **`executar_comparativo.py`**: Bateria comparativa serial com consolidação de quartis e desvio padrão.

---

## 🧹 3. `instancias/` (Filtros e Gerenciamento Interno de Instâncias)
Destinado a pré-processar, auditar e organizar a base de instâncias (`bateria_testes/`).

- **`separar_alto_prejuizo.py`**: Triagem econômica que classifica instâncias em `EQUILIBRADA` ($\text{Razão} \ge 0.5$) ou `ALTO_PREJUIZO` ($\text{Razão} < 0.5$), movendo instâncias desbalanceadas e gerando consolidados sem reexecução.
- **`filtrar_instancias_lucro.py`**: Avalia o Limitante Superior Teórico ($UB > 0$) e executa bateria dedicada no subconjunto de instâncias potencialmente lucrativas.
- **`quarentena.py`**: Identifica instâncias que não foram concluídas no CSV e move para a pasta `quarentena` (ou as restaura).
- **`gerar_cenarios_teste.py`**: Gerador sintético de grafos com distribuições geométricas e clusters de cidades.
