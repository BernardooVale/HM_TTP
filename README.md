# 🚚 Simulador e Framework Experimental do Travelling Thief Problem (TTP)

Ambiente computacional integrado, modular e reproduzível para simulação, experimentação em lote, benchmarking estatístico, avaliação de sensibilidade do parâmetro de aluguel ($R$) e visualização interativa de resultados para o **Travelling Thief Problem (TTP1 e TTP2)**.

---

## 📋 Sumário
1. [Configuração e Instalação](#1-configuração-e-instalação)
2. [Estrutura do Repositório](#2-estrutura-do-repositório)
3. [Como Rodar Simulações e Experimentos](#3-como-rodar-simulações-e-experimentos)
   - [3.1. Orquestrador Unificado (`executar_super_bateria.py`)](#31-orquestrador-unificado-executar_super_bateriapy)
   - [3.2. Triagem e Separação por Equilíbrio Econômico (`separar_alto_prejuizo.py`)](#32-triagem-e-separação-por-equilíbrio-econômico-separar_alto_prejuizopy)
   - [3.3. Gerenciador de Quarentena de Instâncias (`quarentena.py`)](#33-gerenciador-de-quarentena-de-instâncias-quarentenapy)
   - [3.4. Bateria de Instâncias Potencialmente Lucrativas (`filtrar_instancias_lucro.py`)](#34-bateria-de-instâncias-potencialmente-lucrativas-filtrar_instancias_lucropy)
   - [3.5. Normalizador de Estatísticas (`normalizar_estatisticas.py`)](#35-normalizador-de-estatísticas-normalizar_estatisticaspy)
   - [3.6. Avaliação de Sensibilidade de $R$ (`avaliacao_r/runner.py`)](#36-avaliação-de-sensibilidade-de-r-avaliacao_rrunnerpy)
   - [3.7. Execuções Rápidas e Simulações Individuais (`run_simulation.py` e `run_batch.py`)](#37-execuções-rápidas-e-simulações-individuais-run_simulationpy-e-run_batchpy)
4. [Painel Web / Visualizador Interativo (`visualizador/`)](#4-painel-web--visualizador-interativo-visualizador)
5. [Como Adicionar um Novo Solver / Heurística](#5-como-adicionar-um-novo-solver--heurística)
6. [Como Gerar Novos Cenários Sintéticos](#6-como-gerar-novos-cenários-sintéticos)
7. [Dicionário de Arquivos e Colunas de Resultados (`results/`)](#7-dicionário-de-arquivos-e-colunas-de-resultados-results)
8. [Boas Práticas e Verificação com Testes](#8-boas-práticas-e-verificação-com-testes)

---

## 🚀 1. Configuração e Instalação

### Pré-requisitos
- **Python 3.10+** (recomendado Python 3.11 ou superior)
- Gerenciador de pacotes `pip`

### Passo a Passo

1. **Clone ou baixe o repositório:**
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd HM_TTP
   ```

2. **Crie e ative o ambiente virtual:**
   - **No Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **No Linux / macOS (Bash):**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📂 2. Estrutura do Repositório

```text
HM_TTP/
├── executar_super_bateria.py      # Orquestrador central: bateria completa + avaliação de R
├── separar_alto_prejuizo.py       # Triagem econômica (Receita/Custo >= 0.5) e separação de instâncias
├── quarentena.py                  # Gerenciador de quarentena para instâncias não finalizadas
├── filtrar_instancias_lucro.py    # Filtro teórico de lucro (UB > 0) e bateria dedicada
├── normalizar_estatisticas.py     # Normalizador estatístico [0, 1] (melhor = 1.0000)
├── requirements.txt               # Dependências do ecossistema Python
│
├── visualizador/                  # Dashboard Web interativo
│   ├── index.html                 # Aplicação web estática completa (offline, gráficos Chart.js)
│   └── gerar_site.py              # Script que atualiza o index.html com os dados mais recentes de results/
│
├── avaliacao_r/                   # Módulo de sensibilidade e acoplamento do aluguel R
│   ├── config.py                  # Parâmetros e limites da avaliação de R
│   ├── evaluator.py               # Motor de varredura de R e cálculo de gap
│   ├── exact_solvers.py           # Resolvedores exatos para instâncias reduzidas (B&B, DP)
│   ├── heuristic_solvers.py       # Heurísticas adaptadas para varredura de R
│   ├── relatorios.py              # Exportação de tabelas e métricas analíticas
│   ├── runner.py                  # CLI da bateria de avaliação de R
│   ├── resultados/                # CSVs e JSONs gerados pela avaliação de R
│   └── tests/                     # Testes de unidade da avaliação de R
│
├── ttp_simulator/                 # Simulador central do TTP
│   ├── bateria_testes/            # Pasta com os arquivos de instâncias equilibradas (.txt / .ttp)
│   │   ├── alto_prejuizo/         # Subpasta contendo instâncias com prejuízo desproporcional
│   │   └── quarentena/            # Subpasta para instâncias isoladas temporariamente
│   ├── gerar_cenarios_teste.py    # Gerador de cenários sintéticos
│   ├── run_simulation.py          # Execução de simulação unitária
│   ├── run_batch.py               # Execução de simulações em lote
│   ├── src/
│   │   ├── core/                  # Modelos matemáticos (TTP1/TTP2) e avaliadores de rota/mochila
│   │   ├── scenarios/             # Leitores de instâncias (.txt/.ttp) e gerador sintético
│   │   ├── solvers/               # Registro central (SolverRegistry) e implementações de algoritmos
│   │   ├── runners/               # Motores BatchRunner e SimulationRunner com ordenação por tamanho
│   │   └── utils/                 # Métricas estatísticas, exportadores CSV e formatação
│   └── tests/                     # Bateria de testes automatizados com pytest
│
└── results/                       # Diretório central de exportação dos resultados
    ├── bateria_solucoes_detalhada.csv                     # Registro completo execução a execução (todas)
    ├── comparativo_estatisticas_ttp1_ttp2.csv             # Consolidação estatística de todas as instâncias
    ├── comparativo_estatisticas_normalizadas.csv          # Estatísticas normalizadas [0, 1] (todas)
    ├── bateria_equilibradas_detalhada.csv                 # Registro detalhado das instâncias equilibradas (34)
    ├── comparativo_estatisticas_equilibradas.csv          # Consolidação estatística das instâncias equilibradas
    ├── comparativo_estatisticas_equilibradas_normalizadas.csv # Estatísticas normalizadas [0, 1] equilibradas
    ├── triagem_alto_prejuizo.csv                          # Relatório de triagem teórica e classificação das 550
    ├── filtro_instancias_lucrativas.csv                   # Diagnóstico teórico de lucro UB > 0
    ├── bateria_lucrativas_detalhada.csv                   # Execuções exclusivas em instâncias com UB > 0
    └── comparativo_estatisticas_lucrativas.csv            # Resumo estatístico do subconjunto lucrativo
```

---

## 💻 3. Como Rodar Simulações e Experimentos

O framework oferece ferramentas especializadas para diferentes etapas da pesquisa experimental.

### 3.1. Orquestrador Unificado (`executar_super_bateria.py`)

O orquestrador unificado executa de ponta a ponta as duas etapas fundamentais do projeto:
1. **Etapa 1 (`solucoes`)**: Executa todas as instâncias para todos os solucionadores cadastrados nos modelos TTP1 e TTP2, gerando os resultados detalhados e a consolidação estatística (médias, medianas, quartis Q1/Q3, desvio padrão, tempos de CPU).
2. **Etapa 2 (`avaliacao_r`)**: Avalia a sensibilidade ao parâmetro de aluguel $R$ para mapear o ponto de cruzamento ($R_c$), a faixa de 90% e a zona de forte acoplamento.

#### Características operacionais
- **Ordenação por tamanho**: O runner ordena automaticamente as instâncias das menores para as maiores (cidades $\rightarrow$ itens $\rightarrow$ tamanho em bytes), garantindo resultados rápidos logo no início.
- **Retomada Automática (*Resume*)**: Ao reiniciar o comando, ele lê o CSV existente e processa apenas as instâncias e solvers faltantes, sem retrabalho.
- **Interrupção Segura (Ctrl+C)**: Ao pressionar Ctrl+C uma vez, os processos em andamento finalizam atomicamente a linha atual e gravam o CSV no disco de forma íntegra.
- **Isolamento de Subpastas**: Por padrão, o runner ignora subpastas como `quarentena` e `alto_prejuizo`, executando apenas as instâncias da raiz de `bateria_testes/`.

#### Comandos:

- **Execução padrão sobre a bateria ativa (Etapa 1 + Etapa 2 em paralelo):**
  ```bash
  python executar_super_bateria.py
  ```

- **Executar explicitamente sobre as instâncias de `alto_prejuizo`:**
  ```bash
  python executar_super_bateria.py --incluir-alto-prejuizo
  ```

- **Executar sobre diretório customizado:**
  ```bash
  python executar_super_bateria.py --diretorio-bateria ttp_simulator/bateria_testes/alto_prejuizo
  ```

- **Executar com número específico de processos concorrentes (ex: 4 workers):**
  ```bash
  python executar_super_bateria.py --workers 4
  ```

- **Executar somente a bateria de soluções (Etapa 1):**
  ```bash
  python executar_super_bateria.py --etapa solucoes
  ```

- **Executar somente a avaliação de $R$ (Etapa 2):**
  ```bash
  python executar_super_bateria.py --etapa avaliacao_r
  ```

---

### 3.2. Triagem e Separação por Equilíbrio Econômico (`separar_alto_prejuizo.py`)

Em muitas instâncias sintéticas, a taxa de aluguel por unidade de tempo ($R$) é tão elevada em relação ao valor dos itens que o tempo de viagem domina mais de 99% da função objetivo, transformando o problema artificialmente em um TSP puro. Para isolar esses casos e garantir comparações equilibradas e justas entre mochila e rota:

1. **Critério de Classificação:**
   $$\text{Razão} = \frac{\text{Receita Potencial Máxima (Mochila Fracionária)}}{\text{Custo Estimado Mínimo de Transporte } (R \cdot L_{\text{MST}} / v_{\max})}$$
   - Se $\text{Razão} \ge 0.5$: A instância é classificada como **Equilibrada** e permanece em `ttp_simulator/bateria_testes/`.
   - Se $\text{Razão} < 0.5$: A instância é classificada como **Alto Prejuízo** e seus arquivos são movidos para `ttp_simulator/bateria_testes/alto_prejuizo/`.
2. **Aproveitamento de Resultados Sem Reexecução:**
   O script extrai as execuções das instâncias equilibradas a partir de `bateria_solucoes_detalhada.csv`, gerando imediatamente:
   - `results/bateria_equilibradas_detalhada.csv`
   - `results/comparativo_estatisticas_equilibradas.csv`
   - `results/comparativo_estatisticas_equilibradas_normalizadas.csv`
   - `results/triagem_alto_prejuizo.csv`

#### Comandos:

- **Executar triagem, mover arquivos e gerar consolidados filtrados:**
  ```bash
  python separar_alto_prejuizo.py
  ```

- **Apenas auditar status das instâncias sem mover arquivos (*dry-run*):**
  ```bash
  python separar_alto_prejuizo.py --listar
  ```

- **Restaurar todas as instâncias de `alto_prejuizo` de volta para `bateria_testes`:**
  ```bash
  python separar_alto_prejuizo.py --restaurar
  ```

---

### 3.3. Gerenciador de Quarentena de Instâncias (`quarentena.py`)

Identifica instâncias incompletas ou pendentes no CSV detalhado e as move temporariamente para `ttp_simulator/bateria_testes/quarentena/`.

#### Comandos:

- **Auditar status das instâncias sem mover nada (*dry-run*):**
  ```bash
  python quarentena.py --listar
  ```

- **Mover instâncias incompletas para quarentena:**
  ```bash
  python quarentena.py
  ```

- **Restaurar todas as instâncias da quarentena de volta para a bateria de testes:**
  ```bash
  python quarentena.py --restaurar
  ```

---

### 3.4. Bateria de Instâncias Potencialmente Lucrativas (`filtrar_instancias_lucro.py`)

Filtra e executa exclusivamente as instâncias com Limitante Superior Teórico positivo ($UB > 0$), gerando relatórios dedicados em `results/`.

#### Comandos:

- **Executar filtro e rodar a bateria lucrativa:**
  ```bash
  python filtrar_instancias_lucro.py --workers 6
  ```

- **Apenas gerar o relatório do filtro teórico (sem executar simulações):**
  ```bash
  python filtrar_instancias_lucro.py --apenas-filtrar
  ```

---

### 3.5. Normalizador de Estatísticas (`normalizar_estatisticas.py`)

Transforma as métricas para a escala $[0, 1]$ para cada modelo separadamente, onde **o melhor resultado recebe 1.0000** e os demais recebem valores $0.XXXX$ proporcionais:
- **Métricas de Ganho (maior é melhor)**: `media_G`, `min_G`, `q1_G`, `mediana_G`, `q3_G`, `max_G`.
- **Métricas de Custo/Tempo (menor é melhor)**: `media_tempo_cpu_ms`, `std_G`.

```bash
python normalizar_estatisticas.py
```

---

### 3.6. Avaliação de Sensibilidade de $R$ (`avaliacao_r/runner.py`)

Executa o estudo de sensibilidade ao aluguel por unidade de tempo ($R$):

```bash
python -m avaliacao_r.runner --workers 4
```

---

### 3.7. Execuções Rápidas e Simulações Individuais (`run_simulation.py` e `run_batch.py`)

- **Simulação Individual (`ttp_simulator/run_simulation.py`):**
  ```bash
  python ttp_simulator/run_simulation.py \
    --instance ttp_simulator/bateria_testes/instancia_01.txt \
    --solver queimada \
    --model TTP1 \
    --seed 42 \
    --output results/teste_rapido.csv
  ```

- **Bateria em Lote via CLI Simples (`ttp_simulator/run_batch.py`):**
  ```bash
  python ttp_simulator/run_batch.py \
    --test-dir ttp_simulator/bateria_testes/ \
    --solvers queimada \
    --models TTP1 \
    --runs 1 \
    --output results/lote_rapido.csv
  ```

---

## 🌐 4. Painel Web / Visualizador Interativo (`visualizador/`)

O repositório inclui um dashboard interativo completo para visualização, exploração de dados, gráficos dinâmicos e filtragem de soluções.

### Como abrir o Dashboard

1. **Abertura direta no navegador:**
   - Dê um duplo clique no arquivo [`visualizador/index.html`](file:///visualizador/index.html) ou abra-o em qualquer navegador web.
   - O arquivo funciona de maneira autônoma e offline.

2. **Servindo via servidor HTTP local (opcional):**
   ```bash
   python -m http.server 8000 --directory visualizador
   ```
   Em seguida, acesse no navegador: `http://localhost:8000`

### Como Atualizar e Alternar Datasets no Site

Por padrão, o gerador do site carrega as **instâncias equilibradas** (com $\text{Receita} / \text{Custo} \ge 0.5$). Você pode configurar o modo padrão via terminal:

- **Gerar com dataset equilibrado por padrão (recomendado):**
  ```bash
  python visualizador/gerar_site.py
  # ou explicitamente:
  python visualizador/gerar_site.py --modo equilibradas
  ```

- **Gerar com todas as 550 instâncias:**
  ```bash
  python visualizador/gerar_site.py --modo todas
  ```

### Funcionalidades do Painel Analítico:
- **Design Sóbrio e Acadêmico (*Data-First*):** Interface neutra voltada para análise técnica rigorosa, sem elementos visuais distrativos ou saturação excessiva.
- **Heat Grid (Matriz de Desempenho):**
  - **Destaque Especial para o Top 3 por Coluna:** Os três melhores colocados em cada métrica recebem cores de pódio exclusivas e inconfundíveis, fora do gradiente:
    - 🥇 **Ouro:** Amarelo ouro vivo (`🥇`)
    - 🥈 **Prata:** Cinza ardósia / metálico neutro (`🥈`)
    - 🥉 **Bronze:** Cobre / terracota avermelhado quente (`🥉`)
  - **Gradiente Monocromático:** Os quadrantes do 4º lugar em diante utilizam um gradiente suave em azul cobalto técnico proporcional ao score normalizado ($0.00 \to 1.00$).
- **Aba Exclusiva para Dados Brutos:** Tabela separada com valores absolutos de função objetivo ($G$), tempos de computação em milissegundos e métricas de dispersão.
- **Ordenação Numérica Rigorosa:** Cabeçalhos clicáveis com alternância crescente/decrescente (`▲` / `▼`) baseada em atributos de ponto flutuante nativos (`data-val`), imune a ambiguidades de pontuação regional (`.` de milhar e `,` decimal).
- **Modo Noturno Confortável:** Paleta de ardósia profunda neutra com divisores translúcidos suaves, eliminando bordas rígidas ou efeito quadriculado pesado, com persistência automática de tema.
- **Alternância Instantânea de Datasets:** Seletor no cabeçalho para alternar entre as 34 instâncias equilibradas (padrão) e as 550 instâncias completas em tempo real.

---

## 🛠️ 5. Como Adicionar um Novo Solver / Heurística

1. **Criar o arquivo:** em `ttp_simulator/src/solvers/`, ex: `meu_novo_solver.py`.
2. **Implementar a classe herdando de `BaseSolver`:**
   ```python
   from src.core.models import SolucaoTTP, TTPInstance
   from src.solvers.base import BaseSolver
   from src.solvers.registry import SolverRegistry

   @SolverRegistry.register("meu_novo_solver")
   class MeuNovoSolver(BaseSolver):
       def solve(self, instance: TTPInstance) -> SolucaoTTP:
           # Implementação da rota e do plano de coleta
           tour = list(range(1, instance.num_cities + 1))
           plano = {}
           return SolucaoTTP(tour=tour, picking_plan=plano)
   ```
3. **Importar em `ttp_simulator/src/solvers/__init__.py`:**
   ```python
   from src.solvers.meu_novo_solver import MeuNovoSolver
   ```

---

## 🎲 6. Como Gerar Novos Cenários Sintéticos

```bash
python ttp_simulator/gerar_cenarios_teste.py
```

---

## 📊 7. Dicionário de Arquivos e Colunas de Resultados (`results/`)

### Arquivo Detalhado: `bateria_equilibradas_detalhada.csv` / `bateria_solucoes_detalhada.csv`
Contém o registro de cada simulação individual executada:

| Coluna | Descrição |
| :--- | :--- |
| `id_execucao` | Identificador único alfanumérico da execução |
| `timestamp` | Data e hora em formato ISO |
| `arquivo_instancia` | Nome do arquivo de instância avaliado |
| `modelo_ttp` | Variante matemática (`TTP1` ou `TTP2`) |
| `solucionador` | Nome identificador do solver registrado |
| `seed` | Semente pseudoaleatória utilizada |
| `tempo_computacional_seg` | Tempo total de execução do método em segundos |
| `lucro_liquido_G` | Valor da função objetivo final $G$ calculada |
| `tempo_viagem_f` | Tempo total consumido no percurso pelas cidades |
| `valor_bruto_itens_g` | Soma dos valores dos itens recolhidos |
| `peso_acumulado_mochila` | Peso final transportado na mochila |
| `capacidade_mochila_W` | Capacidade limite de peso da mochila |
| `pct_ocupacao_mochila` | Percentual de peso utilizado da mochila ($\%$) |
| `qtd_itens_coletados` | Total de itens coletados pelo plano |
| `total_itens_instancia` | Total de itens existentes na instância |
| `qtd_cidades` | Quantidade de cidades no grafo da instância |

### Arquivos Consolidados e Normalizados
- `comparativo_estatisticas_equilibradas.csv` e `comparativo_estatisticas_ttp1_ttp2.csv`: `media_G`, `std_G`, `min_G`, `q1_G`, `mediana_G`, `q3_G`, `max_G`, `media_tempo_cpu_ms`, `media_tempo_viagem`, etc.
- `comparativo_estatisticas_equilibradas_normalizadas.csv`: Valores normalizados em $[0, 1]$ com melhor $= 1.0000$.
- `triagem_alto_prejuizo.csv`: Lista das 550 instâncias com cidades, itens, aluguel $R$, receita máxima da mochila, custo mínimo de aluguel e classificação (`EQUILIBRADA` vs `ALTO_PREJUIZO`).

---

## 🛡️ 8. Boas Práticas e Verificação com Testes

Execute a suite de testes antes de efetuar commits:
```bash
pytest -v
```
Todos os 94 testes unitários do framework devem passar com sucesso.