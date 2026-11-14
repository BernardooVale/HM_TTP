# Componentes e Configurações Extraídos da Literatura

Este documento sumariza todas as funcionalidades, parâmetros e operadores incorporados à **Heurística da Queimada** (`QueimadaSolver` em [`queimada.py`](queimada.py)) a partir de artigos da literatura, bem como os solucionadores de referência implementados no simulador para comparação justa nos modos **construtivo** e **iterativo/híbrido**.

---

## 1. Parâmetros da `QueimadaSolver` Originados da Literatura

| Parâmetro / Valor no `QueimadaSolver` | Origem na Literatura | Papel na Queimada | Modo de Uso |
| :--- | :--- | :--- | :--- |
| `refinamento=True` | **Faulkner et al. (2015)** (*GECCO/CEC 2015*) | Executa busca local *Drop & Add* (1-bit-flip) sobre o plano de coleta após a seleção inicial (`refinar_plano_coleta`). | Iterativo (refinamento de mochila) |
| `refinamento_rota=True` | **2-OPT clássico do TSP** | Remove cruzamentos na rota térmica antes da seleção de itens (`_refinar_rota_tsp_inicial`). | Pré-processamento TSP |
| `usar_delaunay_2opt=True` | **El Yafrani & Ahiod (2015)** (*Cosolver2B*) / **Namazi et al. (2019)** (*CTTP / CoCo*) | Restringe os candidatos de troca do 2-OPT inicial às arestas da **Triangulação de Delaunay** (`gerar_grafo_delaunay`), reduzindo a vizinhança de $O(n^2)$ para $O(n)$. | Pré-processamento TSP (escalável) |
| `usar_pack_iterative=True` | **Faulkner et al. (2015)** (`Pack` + `PackIterative`) | Substitui a seleção de itens padrão pela busca ternária intervalar sobre o expoente $\alpha$ em $s_{ik} = (p_{ik}/w_{ik})^\alpha / d_i$ com empacotamento em lotes $\mu = \lceil m/r \rceil$ (`pack_iterative`). | Construtivo (2 estágios, análogo ao `S1`) |
| `pos_otimizacao="insertion_bitflip"` | **Faulkner et al. (2015)** (heurística `C1`) | Após a construção da Queimada, alterna 1 passagem de `BitFlip` (atualização diferencial de sufixo) e 1 passagem de `Insertion` (recálculo parcial de segmento) até convergir ou atingir `time_limit_pos`. | Iterativo / Híbrido |
| `pos_otimizacao="boundary_bitflip"` | **Namazi et al.** (*CoCo*) | Após a construção da Queimada, refina apenas a mochila testando aleatoriamente os **itens de fronteira** das curvas $\Pi(P_{c,p}, k)$ e $\Omega(Q_{c,p}, k)$ (`boundary_bit_flip`). | Iterativo (refinamento focado de mochila) |
| `pos_otimizacao="pgch_coco"` | **Namazi et al.** (*PGCH Estendido* + *CoCo*) | Após a construção da Queimada, alterna `TSPSolver` *steepest-ascent* com **PGCH Estendido** ($\Pi$ para descarte, $\Omega$ para coleta compensatória) podado por Delaunay e `KPSolver` via **Boundary Bit-Flip** até convergir ou atingir `time_limit_pos`. | Iterativo / Híbrido |

---

## 2. Aliases Registrados da Queimada (`SolverRegistry`)

Para facilitar experimentos e comparações sem alterar a configuração padrão:

| Nome no `SolverRegistry` | Configuração Carregada / Fixada | Categoria Experimental |
| :--- | :--- | :--- |
| `queimada_construtiva` | Carrega `"queimada_construtiva"` de `melhores_configuracoes.json` com `refinamento=False`, `refinamento_rota=False`, `otimizar_buracos=False`, `pos_otimizacao="nenhuma"`. | **Totalmente Construtiva** |
| `queimada` | Carrega `"queimada"` (campeã geral/híbrida de `melhores_configuracoes.json`). | **Híbrida / Iterativa** |
| `queimada_pack_iterative` | `QueimadaSolver` com `usar_pack_iterative=True`. | **Construtiva / Híbrida (2 estágios)** |
| `queimada_pgch_coco` | `QueimadaSolver` com `pos_otimizacao="pgch_coco"`. | **Híbrida / Iterativa** |
| `queimada_insertion_bitflip` | `QueimadaSolver` com `pos_otimizacao="insertion_bitflip"`. | **Híbrida / Iterativa** |
| `queimada_boundary_bitflip` | `QueimadaSolver` com `pos_otimizacao="boundary_bitflip"`. | **Híbrida / Iterativa** |

---

## 3. Pares da Literatura Implementados para Comparação

### 3.1 Grupo Construtivo (para comparar com `queimada_construtiva`)
- **Polyakovskiy et al. (2014):** `s2_heuristic`, `s5_heuristic` (`src/solvers/constructive/s2_s5_solvers.py`).
- **Faulkner et al. (2015):** `faulkner_s1` (`CLK` + `PackIterative` em `src/solvers/constructive/faulkner_s1.py`).

### 3.2 Grupo Iterativo / Busca Local (para comparar com `queimada` híbrida)
- **Faulkner et al. (2015):** `faulkner_s2` (`BitFlip`), `faulkner_s3` (`(1+1)-EA`), `faulkner_s4` (`Insertion`), `faulkner_s5` (reamostragem `CLK + PackIterative`), `faulkner_c1`–`faulkner_c6` (`src/solvers/iterative/faulkner_solvers.py`).
- **El Yafrani & Ahiod (2015) — Cosolver2B:** `cosolver2b_first_fit`, `cosolver2b_best_fit` (`src/solvers/iterative/cosolver2b_solvers.py`).
- **Namazi et al. (2019) — CTTP / PGCH:** `cttp_2opt`, `cttp_pgch` (`src/solvers/iterative/pgch_solvers.py`).
- **Namazi et al. — CoCo:** `coco_solver` (`src/solvers/iterative/coco_solvers.py`).
- **Chagas & Wagner (2021) — WSM:** `wsm_solver` (`src/solvers/iterative/wsm_solvers.py`).

---

## 4. Calibração no `grid_search_paralelo.py`

Ao executar o script com `--espaco hibrido`:
- O grid search avalia tanto configurações estritamente construtivas quanto combinações com `usar_pack_iterative`, `usar_delaunay_2opt` e `pos_otimizacao` (`"nenhuma"`, `"pgch_coco"`, `"insertion_bitflip"`, `"boundary_bitflip"`).
- Ao final, persiste **duas** campeãs separadas em `ttp_simulator/src/config/melhores_configuracoes.json`:
  - **`"queimada"`**: melhor configuração geral (iterativa/híbrida).
  - **`"queimada_construtiva"`**: melhor configuração estritamente construtiva (`refinamento=False`, `refinamento_rota=False`, `otimizar_buracos=False`, `pos_otimizacao="nenhuma"`).
