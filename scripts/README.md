# 🛠️ Scripts do Framework TTP

Este diretório organiza todos os scripts utilitários, orquestradores experimentais e ferramentas de dados do projeto de forma categorizada e intuitiva.

---

## 📁 1. `benchmark/` (Teste Global e Benchmarking)
Destinado a execuções em larga escala comparando todos os solucionadores cadastrados no framework e gerando os relatórios estatísticos consolidados.

- **`executar_super_bateria.py`**: Orquestrador unificado que executa a bateria completa de solucionadores (com ordenação por tamanho, paralelismo e retomada automática) e opcionalmente a análise de sensibilidade ao parâmetro de aluguel $R$.
  - *Modelo padrão:* **TTP1** (benchmark padrão da literatura). O modelo **TTP2** é opcional via `--incluir-ttp2` ou `--modelos TTP1,TTP2`.
- **`executar_comparativo.py`**: Bateria comparativa serial com consolidação de quartis, desvio padrão e exportação detalhada.
- **`normalizar_estatisticas.py`**: Normaliza todas as métricas no intervalo $[0, 1]$ (onde o melhor solver recebe `1.0000` e os demais valores relativos) para comparações justas de desempenho.

---

## 🔥 2. `experimentos_queimada/` (Hipóteses da Heurística da Queimada)
Destinado a avaliar cientificamente as opções e configurações estruturais da Heurística da Queimada (temperatura, dissipação, 2-opt, afinidade térmica, backward edges, busca direcionada e otimização de buracos).

- **`grid_search_paralelo.py`**: Motor oficial de avaliação das variantes da Queimada. Varre combinações estruturais (com suporte aos espaços `completo`, `direcionado`, `padrao` e `rapido`), aproveita resultados prévios com retomada automática (Resume sem recalcular instâncias já concluídas), gera ranking estatístico e atualiza automaticamente a melhor configuração no `config_manager.py` para os benchmarks globais.

---

## 🧹 3. `instancias/` (Filtros e Gerenciamento Interno de Instâncias)
Destinado a pré-processar, auditar e organizar a base de instâncias (`bateria_testes/`).

- **`separar_alto_prejuizo.py`**: Triagem econômica que classifica instâncias em `EQUILIBRADA` ($\text{Razão} \ge 0.5$) ou `ALTO_PREJUIZO` ($\text{Razão} < 0.5$), movendo instâncias desbalanceadas e gerando consolidados sem reexecução.
- **`filtrar_instancias_lucro.py`**: Avalia o Limitante Superior Teórico ($UB > 0$) e executa bateria dedicada no subconjunto de instâncias potencialmente lucrativas.
- **`quarentena.py`**: Identifica instâncias que não foram concluídas no CSV e move para a pasta `quarentena` (ou as restaura).
- **`gerar_cenarios_teste.py`**: Gerador sintético de grafos com distribuições geométricas e clusters de cidades.
