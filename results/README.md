# 📊 Resultados Experimentais do TTP (Traveling Thief Problem)

Este diretório centraliza todos os dados de execução do framework em uma arquitetura simétrica **$3 \times 3$** (3 famílias de solucionadores $\times$ 3 cenários de instâncias).

---

## 🗂️ Estrutura de Diretórios ($3 \times 3$)

```text
results/
├── todos/                  # Todos os solucionadores (Literatura, Baselines, 37% e Queimada)
│   ├── completo.csv        # Todas as 550+ instâncias da bateria
│   ├── equilibradas.csv    # Subconjunto equilibrado (Receita Potencial / Custo MST ≥ 0.5)
│   └── lucrativas.csv      # Subconjunto matematicamente lucrativo (Upper Bound > 0)
│
├── queimada/               # Exclusivo das variantes da Heurística da Queimada
│   ├── completo.csv        # Histórico completo da Queimada em todas as instâncias
│   ├── equilibradas.csv    # Desempenho da Queimada nas instâncias equilibradas
│   ├── lucrativas.csv      # Desempenho da Queimada nas instâncias lucrativas
│   └── grid_search/        # Avaliações combinatórias estruturais (grid_search_paralelo.py)
│
├── heuristica_37/          # Exclusivo das variantes da Heurística dos 37%
│   ├── completo.csv        # Histórico completo dos 37% em todas as instâncias
│   ├── equilibradas.csv    # Desempenho dos 37% nas instâncias equilibradas
│   └── lucrativas.csv      # Desempenho dos 37% nas instâncias lucrativas
│
└── triagem/                # Diagnósticos estruturais das instâncias
    ├── triagem_alto_prejuizo.csv       # Classificação EQUILIBRADA vs ALTO_PREJUIZO
    └── filtro_instancias_lucrativas.csv # Avaliação do Limitante Superior Teórico (UB)
```

---

## 🔍 Dicionário de Colunas (CSVs de Execução)
- `id_execucao`: Identificador único da execução.
- `arquivo_instancia`: Nome do arquivo de instância (ex: `instancia_01.txt`, `512.ttp`).
- `modelo_ttp`: `TTP1` (padrão da literatura) ou `TTP2` (com decaimento temporal).
- `solucionador`: Nome registrado do solver no `SolverRegistry`.
- `lucro_liquido_G`: Valor final da função objetivo $G$.
- `tempo_viagem_f`: Tempo total gasto para percorrer o ciclo de cidades.
- `valor_bruto_itens_g`: Soma do valor dos itens coletados.
- `peso_acumulado_mochila`: Peso total transportado na mochila.
- `pct_ocupacao_mochila`: Percentual da capacidade máxima utilizado.
- `tempo_computacional_seg`: Tempo de execução do algoritmo em segundos.
