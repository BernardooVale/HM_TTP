# 🚚 Framework Experimental do Travelling Thief Problem (TTP)

Ambiente computacional de alto desempenho, modular e reproduzível para simulação, experimentação em lote, benchmarking estatístico, exploração de hiperparâmetros da **Heurística da Queimada** e visualização interativa de resultados para o **Travelling Thief Problem**.

> [!NOTE]
> **Padrão da Literatura (TTP1):** Quase 100% dos benchmarks da literatura científica utilizam o modelo monobjetivo **TTP1** (Polyakovskiy et al., 2014; Faulkner et al., 2015). Por padrão, todo o framework opera estritamente sob o **TTP1**. O modelo multiobjetivo **TTP2** é suportado integralmente, porém configurado como **opcional** (ativado via flag `--incluir-ttp2` ou `--modelo TTP2`).

---

## 📋 Sumário
1. [Visão Geral e Arquitetura](#1-visão-geral-e-arquitetura)
2. [Guia Rápido: Como Criar e Rodar sua Heurística Sem Fofoca](#2-guia-rápido-como-criar-e-rodar-sua-heurística-sem-fofoca)
3. [Instalação e Configuração](#3-instalação-e-configuração)
4. [Estrutura do Repositório](#4-estrutura-do-repositório)
5. [Como Rodar os Experimentos](#5-como-rodar-os-experimentos)
   - [5.1. CLI Rápido de Solucionadores (`testar_heuristica.py`)](#51-cli-rápido-de-solucionadores-testar_heuristicapy)
   - [5.2. Teste Global e Benchmarking (`scripts/benchmark/`)](#52-teste-global-e-benchmarking-scriptsbenchmark)
   - [5.3. Experimentos e Hipóteses da Queimada (`scripts/experimentos_queimada/`)](#53-experimentos-e-hipóteses-da-queimada-scriptsexperimentos_queimada)
   - [5.4. Filtros e Gerenciamento Interno de Instâncias (`scripts/instancias/`)](#54-filtros-e-gerenciamento-interno-de-instâncias-scriptsinstancias)
   - [5.5. Estudo de Sensibilidade ao Parâmetro $R$ (`avaliacao_r/`)](#55-estudo-de-sensibilidade-ao-parâmetro-r-avaliacao_r)
6. [Painel Web Interativo (`visualizador/`)](#6-painel-web-interativo-visualizador)
7. [Dicionário de Resultados (`results/`)](#7-dicionário-de-resultados-results)
8. [Testes Automatizados](#8-testes-automatizados)

---

## ⚡ 1. Visão Geral e Arquitetura

O Travelling Thief Problem combina dois problemas clássicos NP-difíceis:
1. **Traveling Salesperson Problem (TSP):** Encontrar a rota ótima passando por todas as cidades e retornando à cidade de partida.
2. **Knapsack Problem (KP):** Selecionar itens de alto valor distribuídos nas cidades sem estourar a capacidade máxima de peso $W$.

A interdependência surge porque o peso coletado na mochila reduz a velocidade do ladrão ao longo do trajeto, aumentando o tempo total de viagem $f$ e o custo de aluguel por unidade de tempo $R$, impactando a função objetivo de lucro líquido:
$$G = g - R \cdot f$$
onde $g$ é a soma dos valores dos itens coletados e $f$ é o tempo de percurso.

---

## 🚀 2. Guia Rápido: Como Criar e Rodar sua Heurística Sem Fofoca

Adicionar um novo algoritmo ao framework leva **menos de 2 minutos** em 3 passos simples:

### Passo 1: Criar o arquivo do solver
Crie um arquivo em [`ttp_simulator/src/solvers/`](file:///ttp_simulator/src/solvers/), por exemplo `meu_solver.py` (ou copie o template pronto [`ttp_simulator/src/solvers/template_heuristica.py`](file:///ttp_simulator/src/solvers/template_heuristica.py)).

### Passo 2: Implementar a classe herdando de `BaseSolver`
```python
from typing import Dict, List
from src.core.models import SolucaoTTP, TTPInstance
from src.solvers.base import BaseSolver, SolverRegistry

@SolverRegistry.register("meu_solver")
class MeuSolver(BaseSolver):
    @property
    def nome(self) -> str:
        return "meu_solver"

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        # 1. Rota pelas cidades (deve iniciar na cidade 1)
        tour = list(range(1, instance.num_cities + 1))

        # 2. Plano de coleta de itens (cidade -> lista de item_ids)
        plano = {c: [] for c in tour}
        peso_atual = 0.0

        for item in sorted(instance.itens, key=lambda it: it.valor / it.peso, reverse=True):
            if peso_atual + item.peso <= instance.capacidade_mochila:
                plano[item.cidade_origem].append(item.id)
                peso_atual += item.peso

        return SolucaoTTP(rota=tour, plano_coleta=plano)
```

> [!TIP]
> Graças à descoberta dinâmica de módulos, você **não precisa editar nenhum arquivo de configuração ou `__init__.py`**. Basta salvar o arquivo e o solver estará registrado automaticamente!

### Passo 3: Testar e comparar imediatamente no terminal
```bash
# Testar seu solver em uma instância rápida e comparar com os melhores da literatura:
python testar_heuristica.py --solver meu_solver --comparar

# Executar na bateria rápida de instâncias:
python testar_heuristica.py --solver meu_solver --bateria

# Ver todos os solvers registrados:
python testar_heuristica.py --listar
```

---

## 📦 3. Instalação e Configuração

### Pré-requisitos
- **Python 3.10+** (recomendado Python 3.11 ou 3.13)
- Gerenciador de pacotes `pip`

```bash
# 1. Clonar o repositório
git clone <URL_DO_REPOSITORIO>
cd HM_TTP

# 2. Criar e ativar ambiente virtual
# No Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# No Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar dependências
pip install -r requirements.txt
```

---

## 📂 4. Estrutura do Repositório

O projeto possui uma organização limpa e modular:

```text
HM_TTP/
├── testar_heuristica.py          # 🎯 CLI amigável para testar qualquer solver sem fofoca
├── requirements.txt              # Dependências do ecossistema Python
├── README.md                     # Documentação central do projeto
│
├── scripts/                      # 🛠️ Scripts utilitários e experimentais categorizados
│   ├── benchmark/                # 🚀 Teste global de todos os solvers e consolidação
│   │   ├── executar_super_bateria.py
│   │   └── normalizar_estatisticas.py
│   ├── experimentos_queimada/    # 🔥 Testes de hipóteses e grid search da Queimada
│   │   ├── comparar_queimadas.py
│   │   ├── grid_search_paralelo.py
│   │   ├── grid_search_queimada.py
│   │   └── executar_comparativo.py
│   └── instancias/               # 🧹 Scripts internos de triagem e quarentena
│       ├── separar_alto_prejuizo.py
│       ├── filtrar_instancias_lucro.py
│       ├── quarentena.py
│       └── gerar_cenarios_teste.py
│
├── results/                      # 📊 Pasta ÚNICA de saída para todos os CSVs
│   ├── README.md                 # Dicionário de arquivos e colunas dos resultados
│   ├── bateria_solucoes_detalhada.csv
│   ├── comparativo_estatisticas_ttp1_ttp2.csv
│   ├── comparativo_estatisticas_normalizadas.csv
│   ├── bateria_equilibradas_detalhada.csv
│   ├── comparativo_estatisticas_equilibradas.csv
│   ├── triagem_alto_prejuizo.csv
│   └── grid_search_queimada_completo.csv
│
├── ttp_simulator/                # 🚚 Motor central do TTP
│   ├── bateria_testes/           # Instâncias de teste (.txt e .ttp)
│   │   └── alto_prejuizo/        # Subpasta para instâncias desbalanceadas (Razão < 0.5)
│   ├── src/
│   │   ├── core/                 # Modelos matemáticos (models, evaluators)
│   │   ├── solvers/              # Implementações de solucionadores e template
│   │   ├── scenarios/            # Leitores de instâncias (.txt / .ttp)
│   │   ├── runners/              # Motores paralelos BatchRunner com Resume
│   │   └── utils/                # Loggers CSV e métricas
│   └── tests/                    # Bateria de testes automatizados com pytest
│
├── avaliacao_r/                  # 📐 Estudo de sensibilidade ao parâmetro de aluguel R
└── visualizador/                 # 🌐 Painel web analítico e visualizador interativo
```

---

## 💻 5. Como Rodar os Experimentos

### 5.1. CLI Rápido de Solucionadores (`testar_heuristica.py`)

Localizado na raiz, é o ponto de entrada rápido para quem desenvolve novos solvers:

```bash
# Listar todos os solucionadores cadastrados
python testar_heuristica.py --listar

# Testar seu solver em instância padrão (cenario_pequeno_10c.txt)
python testar_heuristica.py --solver queimada

# Testar e comparar lado a lado com baselines (com pódio 🥇, 🥈, 🥉)
python testar_heuristica.py --solver queimada --comparar

# Testar em instância específica
python testar_heuristica.py --solver pack_iterative --instancia 512.ttp

# Executar bateria rápida em instâncias representativas
python testar_heuristica.py --solver heuristic_37_refinada --bateria

# Exibir detalhes da rota e lista de itens roubados
python testar_heuristica.py --solver queimada --detalhes

# Testar sob o modelo TTP2 (opcional)
python testar_heuristica.py --solver queimada --modelo TTP2
```

---

### 5.2. Teste Global e Benchmarking (`scripts/benchmark/`)

Executa todos os solucionadores cadastrados em todas as instâncias da bateria de testes, com suporte a **multiprocessamento paralelo**, **ordenação inteligente por tamanho** (instâncias menores rodam primeiro) e **retomada automática (*Resume*)**.

```bash
# Execução padrão completa (Etapa 1: Soluções + Etapa 2: Avaliação de R)
python scripts/benchmark/executar_super_bateria.py

# Executar apenas a bateria de soluções (Etapa 1) com 8 workers
python scripts/benchmark/executar_super_bateria.py --etapa solucoes --workers 8

# Filtrar instâncias específicas por padrão glob
python scripts/benchmark/executar_super_bateria.py --padrao "cenario_*"

# Executar solvers específicos
python scripts/benchmark/executar_super_bateria.py --solvers "queimada,heuristic_37_refinada,pack_iterative"

# Incluir o modelo TTP2 na bateria (por padrão roda apenas TTP1)
python scripts/benchmark/executar_super_bateria.py --incluir-ttp2

# Normalizar as estatísticas geradas no intervalo [0, 1] (melhor = 1.0000)
python scripts/benchmark/normalizar_estatisticas.py
```

---

### 5.3. Experimentos e Hipóteses da Queimada (`scripts/experimentos_queimada/`)

Scripts dedicados a validar hipóteses científicas e afinar hiperparâmetros da Heurística da Queimada:

```bash
# 1. Comparação direta visual entre variantes da Queimada vs Estado da Arte
python scripts/experimentos_queimada/comparar_queimadas.py

# Comparar em instâncias selecionadas
python scripts/experimentos_queimada/comparar_queimadas.py --instancias 512.ttp cenario_medio_20c.txt

# 2. Grid Search Paralelo de Alto Desempenho (com Resume automático)
# Espaço padrão (288 combinações) em instâncias leves:
python scripts/experimentos_queimada/grid_search_paralelo.py --espaco padrao --instancias leves

# Espaço rápido (construtivo sem 2-opt) em todas as instâncias:
python scripts/experimentos_queimada/grid_search_paralelo.py --espaco rapido --workers 8

# Espaço completo (2.880 combinações com backward edges e afinidade térmica):
python scripts/experimentos_queimada/grid_search_paralelo.py --espaco completo --workers 12

# 3. Grid Search clássico serial de referência
python scripts/experimentos_queimada/grid_search_queimada.py

# 4. Bateria comparativa serial com quartis e desvio padrão
python scripts/experimentos_queimada/executar_comparativo.py
```

---

### 5.4. Filtros e Gerenciamento Interno de Instâncias (`scripts/instancias/`)

Módulos internos para auditar, balancear e isolar instâncias da base:

```bash
# 1. Triagem econômica (Receita Máxima da Mochila / Custo Mínimo na MST >= 0.5)
# Classifica, move instâncias desbalanceadas para 'alto_prejuizo' e extrai resultados:
python scripts/instancias/separar_alto_prejuizo.py

# Apenas auditar sem mover arquivos (dry-run):
python scripts/instancias/separar_alto_prejuizo.py --listar

# Restaurar instâncias de volta para a pasta principal:
python scripts/instancias/separar_alto_prejuizo.py --restaurar

# 2. Filtro teórico de lucro matemático (Limitante Superior Dantzig - MST > 0)
python scripts/instancias/filtrar_instancias_lucro.py --apenas-filtrar

# Rodar bateria dedicada nas instâncias com UB > 0:
python scripts/instancias/filtrar_instancias_lucro.py --workers 6

# 3. Gerenciamento de quarentena de instâncias incompletas
python scripts/instancias/quarentena.py --listar
python scripts/instancias/quarentena.py --restaurar

# 4. Gerar novos cenários sintéticos de teste
python scripts/instancias/gerar_cenarios_teste.py
```

---

### 5.5. Estudo de Sensibilidade ao Parâmetro $R$ (`avaliacao_r/`)

Mapeia o comportamento das heurísticas sob variação sistemática da taxa de aluguel por unidade de tempo ($R$), identificando o ponto de transição ($R_c$) e as zonas de acoplamento forte:

```bash
python -m avaliacao_r.runner --workers 4
```

---

## 🌐 6. Painel Web Interativo (`visualizador/`)

O repositório inclui um dashboard interativo completo para visualização científica de dados, matrizes de desempenho (*Heat Grid* com pódios 🥇, 🥈, 🥉), gráficos de radar e dispersão.

### Como abrir o painel:
1. **Diretamente:** Abra o arquivo [`visualizador/index.html`](file:///visualizador/index.html) no navegador (funciona 100% offline).
2. **Atualizar o site com novos resultados:**
   ```bash
   python visualizador/gerar_site.py
   # Carregar um CSV específico no dashboard:
   python visualizador/gerar_site.py --csv results/grid_search_queimada_completo.csv
   ```
3. **Servir localmente via HTTP (opcional):**
   ```bash
   python -m http.server 8000 --directory visualizador
   ```
   Acesse: `http://localhost:8000`

---

## 📊 7. Dicionário de Resultados (`results/`)

Todos os dados exportados ficam centralizados em [`results/`](file:///results/) (consulte [`results/README.md`](file:///results/README.md) para detalhes):

| Arquivo | Descrição |
| :--- | :--- |
| `bateria_solucoes_detalhada.csv` | Todas as execuções run-a-run da bateria global (todas as 550 instâncias). |
| `comparativo_estatisticas_ttp1_ttp2.csv` | Consolidação estatística com Média, Desvio Padrão, Mediana, Q1, Q3, Mín, Máx e CPU. |
| `comparativo_estatisticas_normalizadas.csv` | Desempenho relativo normalizado no intervalo $[0, 1]$ (melhor = `1.0000`). |
| `bateria_equilibradas_detalhada.csv` | Execuções detalhadas apenas nas 34 instâncias equilibradas ($\text{Razão} \ge 0.5$). |
| `comparativo_estatisticas_equilibradas.csv` | Estatísticas consolidadas das 34 instâncias equilibradas. |
| `comparativo_estatisticas_equilibradas_normalizadas.csv` | Dados normalizados $[0, 1]$ do subconjunto equilibrado. |
| `triagem_alto_prejuizo.csv` | Classificação teórica de todas as 550 instâncias (`EQUILIBRADA` vs `ALTO_PREJUIZO`). |
| `filtro_instancias_lucrativas.csv` | Classificação matemática por Limitante Superior ($UB > 0$). |
| `bateria_lucrativas_detalhada.csv` | Execuções no subconjunto comprovadamente lucrativo. |
| `grid_search_queimada_completo.csv` | 2.880 configurações avaliadas com afinidade térmica e backward edges. |
| `grid_search_queimada_padrao.csv` | 288 configurações do espaço clássico. |
| `grid_search_queimada_rapido.csv` | Espaço construtivo rápido sem 2-opt. |

---

## 🛡️ 8. Testes Automatizados

A suíte de testes unitários cobre modelos matemáticos, parsers, avaliadores e solvers:

```bash
# Executar todos os testes com o pytest do ambiente virtual
.\.venv\Scripts\pytest -v
```

Todos os 97 testes automatizados devem passar com sucesso em frações de segundo.
