# 📊 Resultados Experimentais do TTP (Travelling Thief Problem)

Este diretório centraliza todos os dados brutos, relatórios detalhados e tabelas estatísticas geradas pelo framework.

---

## 📑 Índice dos Arquivos de Resultados

### 1. Bateria Global (Todas as 550 Instâncias)
| Arquivo | Descrição |
| :--- | :--- |
| `bateria_solucoes_detalhada.csv` | Registro de cada execução individual (instância × solver × modelo × seed). |
| `comparativo_estatisticas_ttp1_ttp2.csv` | Consolidação estatística com Média, Desvio Padrão, Quartis (Q1, Mediana, Q3), Mín, Máx e Tempo de CPU. |
| `comparativo_estatisticas_normalizadas.csv` | Desempenho relativo normalizado no intervalo $[0, 1]$ (o melhor solver recebe 1.0000). |
| `bateria_completa.csv` | Registro de execuções das instâncias sintéticas preliminares. |
| `comparativo_heuristicas.csv` | Comparativo resumido de execuções preliminares. |

### 2. Instâncias Equilibradas (Recomendado para Benchmarking Científico)
Subconjunto selecionado com base no critério econômico $\text{Razão} = \frac{\text{Receita Potencial Mochila}}{\text{Custo Mínimo Aluguel MST}} \ge 0.5$, onde o problema não degenera em TSP puro:
| Arquivo | Descrição |
| :--- | :--- |
| `bateria_equilibradas_detalhada.csv` | Execuções detalhadas apenas nas instâncias equilibradas. |
| `comparativo_estatisticas_equilibradas.csv` | Estatísticas consolidadas das instâncias equilibradas. |
| `comparativo_estatisticas_equilibradas_normalizadas.csv` | Scores normalizados $[0, 1]$ para análise justa de pódios. |

### 3. Triagem Econômica e Diagnóstico de Lucro
| Arquivo | Descrição |
| :--- | :--- |
| `triagem_alto_prejuizo.csv` | Diagnóstico de todas as 550 instâncias classificadas como `EQUILIBRADA` ou `ALTO_PREJUIZO`. |
| `filtro_instancias_lucrativas.csv` | Diagnóstico pelo Limitante Superior Teórico ($UB = \text{Mochila Fracionária} - R \cdot L_{\text{MST}} / v_{\max}$). |
| `bateria_lucrativas_detalhada.csv` | Execuções detalhadas no subconjunto matematicamente lucrativo ($UB > 0$). |
| `comparativo_estatisticas_lucrativas.csv` | Estatísticas consolidadas do subconjunto lucrativo. |

### 4. Experimentos e Grid Search da Queimada
| Arquivo | Descrição |
| :--- | :--- |
| `grid_search_queimada.csv` | Varredura clássica das parametrizações da Queimada em cenários de teste sintéticos. |
| `grid_search_queimada_padrao.csv` | Espaço clássico com 288 configurações avaliadas. |
| `grid_search_queimada_rapido.csv` | Espaço construtivo rápido (sem 2-opt e sem buscas locais pesadas). |
| `grid_search_queimada_completo.csv` | Espaço exaustivo completo de 2.880 configurações explorando afinidade térmica, backward edges e preenchimento de buracos. |

---

## 🔍 Dicionário de Colunas Comuns (CSVs Detalhados)
- `id_execucao`: Identificador alfanumérico único da execução.
- `arquivo_instancia`: Nome do arquivo de instância (ex: `instancia_01.txt`, `512.ttp`).
- `modelo_ttp`: `TTP1` (padrão da literatura) ou `TTP2` (com decaimento temporal).
- `solucionador`: Nome registrado do solver no `SolverRegistry`.
- `lucro_liquido_G`: Valor final da função objetivo $G$.
- `tempo_viagem_f`: Tempo total gasto para percorrer o ciclo de cidades.
- `valor_bruto_itens_g`: Soma do valor dos itens coletados.
- `peso_acumulado_mochila`: Peso total transportado na mochila.
- `pct_ocupacao_mochila`: Percentual da capacidade máxima utilizado.
- `tempo_computacional_seg`: Tempo de execução do algoritmo em segundos.
