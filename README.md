# 🚚 Simulador Modular do Travelling Thief Problem (TTP) em Python

Bem-vindo ao repositório do ambiente de simulação e experimentação computacional do **Travelling Thief Problem (TTP)**. Este ambiente foi projetado para ser **modular, extensível e reprodutível**, permitindo que cada integrante do grupo desenvolva e acople sua própria heurística sem interferir no código dos demais.

---

## 🚀 1. Configuração e Instalação

### Pré-requisitos

- **Python 3.10+**
- Gerenciador de pacotes `pip`

### Passo a Passo de Inicialização

1. **Clone ou baixe este repositório** para sua máquina local.
2. **Crie e ative um ambiente virtual (recomendado):**

```bash
   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate

   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1
```

3. **Instale as dependências do projeto:**

```bash
   pip install -r requirements.txt
```

---

## 📂 2. Estrutura do Projeto

```text
ttp_simulator/
├── PROGRESS.md            # Arquivo de controle mantido pelo Agente de IA
├── README.md              # Este guia de uso e desenvolvimento
├── run_simulation.py      # CLI para rodar uma simulação individual
├── run_batch.py           # CLI para executar baterias de testes em lote
├── requirements.txt       # Dependências do projeto
├── bateria_testes/        # Pasta contendo os arquivos .txt de instâncias
├── results/               # Arquivos .csv gerados pelas execuções
├── src/
│   ├── core/              # Modelos matemáticos (TTP1, TTP2), Dataclasses e Avaliadores
│   ├── scenarios/         # Parsers de arquivo .txt e Gerador Sintético
│   ├── solvers/           # Interface BaseSolver, Registry e Implementações Heurísticas
│   ├── runners/           # Motores de execução individual e em lote
│   └── utils/             # Exportador e Logger CSV
└── tests/                 # Testes unitários com PyTest
```

---

## 🛠️ 3. Como Implementar uma Nova Heurística (Passo a Passo)

Para criar sua heurística, você só precisa criar um novo arquivo em `src/solvers/` e registrar sua classe no `SolverRegistry`.

### Passo 1: Criar o arquivo do seu algoritmo

Crie um arquivo Python em `src/solvers/`, por exemplo: `src/solvers/minha_heuristica.py`.

### Passo 2: Implementar a classe herdando de `BaseSolver`

Exemplo de código completo para o seu arquivo:

```python
from src.core.models import Item, SolucaoTTP, TTPInstance
from src.solvers.base import BaseSolver
from src.solvers.registry import SolverRegistry


# O decorator @SolverRegistry.register registra seu algoritmo para uso no terminal
@SolverRegistry.register("minha_heuristica")
class MinhaHeuristicaCustomizada(BaseSolver):
  """Descrição breve da sua estratégia heurística."""

  def solve(self, instance: TTPInstance) -> SolucaoTTP:
    # 1. Obter dados da instância
    num_cities = instance.num_cities
    items = instance.items
    capacity = instance.capacity_W

    # 2. Lógica da Rota (Exemplo: Rota em ordem sequencial 1 -> 2 -> ... -> N -> 1)
    rota_proposta = list(range(1, num_cities + 1))

    # 3. Lógica do Plano de Coleta de Itens
    # z é um dicionário {id_item: cidade_de_coleta} ou lista binária
    plano_coleta = {}
    peso_atual = 0.0

    # Exemplo: Seleciona itens de forma gulosa baseando-se no valor/peso
    items_ordenados = sorted(
        items, key=lambda x: x.profit / x.weight, reverse=True
    )

    for item in items_ordenados:
      if peso_atual + item.weight <= capacity:
        # Coleta o item na cidade de origem dele
        plano_coleta[item.id] = item.assigned_node
        peso_atual += item.weight

    # 4. Retornar a estrutura de solução obrigatória
    return SolucaoTTP(tour=rota_proposta, picking_plan=plano_coleta)
```

### Passo 3: Importar seu arquivo no `__init__.py`

Abra o arquivo `src/solvers/__init__.py` e adicione uma linha importando o seu novo arquivo para que o registro automático funcione:

```python
from src.solvers.minha_heuristica import MinhaHeuristicaCustomizada
```

Pronto! Sua heurística já está integrada ao sistema e pronta para ser chamada via linha de comando pelo nome `"minha_heuristica"`.

---

## 🎲 4. Como Criar Novas Funções/Tipos de Cenários

Se você quiser criar novos cenários sintéticos (novas topologias de grafos, novas distribuições de itens, etc.) para testar suas hipóteses:

### Passo 1: Adicionar a função geradora em `src/scenarios/generator.py`

Exemplo de nova função dentro da classe `ScenarioGenerator`:

```python
class ScenarioGenerator:

  def generate_custom_scenario(
      self, num_cities: int, seed: int = 42
  ) -> TTPInstance:
    """Gera um cenário sintético com lógica customizada."""
    import random

    random.seed(seed)

    # Exemplo: Criar cidades dispostas em círculo
    # ... código de geração ...

    return instance
```

### Passo 2: Exportar o cenário para a pasta `bateria_testes/`

Você pode rodar um script rápido em Python para gerar arquivos `.txt` prontos para reprodução:

```python
from src.scenarios.generator import ScenarioGenerator

generator = ScenarioGenerator()

# Gera 50 cenários variando a semente de 1 a 50
for seed in range(1, 51):
  generator.export_to_txt(
      instance_type="cluster_esparso",
      seed=seed,
      output_filepath=f"bateria_testes/instancia_cluster_seed_{seed}.txt",
  )
```

---

## 💻 5. Como Rodar Simulações e Experimentos

O projeto possui dois scripts principais na raiz para execução via terminal.

### Opção A: Executar uma Simulação Única (`run_simulation.py`)

Utilize para testar rapidamente se a sua heurística está funcionando em uma instância específica.

```bash
python run_simulation.py \
  --instance bateria_testes/instancia_01.txt \
  --solver minha_heuristica \
  --model TTP1 \
  --seed 42 \
  --output results/teste_individual.csv
```

### Opção B: Executar a Bateria de Testes Completa em Lote (`run_batch.py`)

Utilize para rodar seus experimentos científicos sobre **todos** os arquivos da pasta `bateria_testes/`.

- **Rodar apenas a SUA heurística em todas as instâncias:**

```bash
  python run_batch.py \
    --test-dir bateria_testes/ \
    --solvers minha_heuristica \
    --models TTP1 \
    --runs 5 \
    --output results/resultado_minha_heuristica.csv
```

- **Rodar TODAS as heurísticas cadastradas no projeto para comparação:**

```bash
  python run_batch.py \
    --test-dir bateria_testes/ \
    --solvers all \
    --models TTP1,TTP2 \
    --runs 10 \
    --output results/bateria_comparativa_completa.csv
```

---

## 📊 6. Formato do Arquivo de Saída (`.csv`)

Todas as execuções salvam automaticamente os resultados na pasta `results/`. As colunas gravadas no `.csv` são:

| Coluna | Descrição |
| --- | --- |
| `id_execucao` | ID único da simulação |
| `timestamp` | Data e hora da execução |
| `arquivo_instancia` | Nome da instância `.txt` testada |
| `modelo_ttp` | Modelo avaliado (`TTP1` ou `TTP2`) |
| `solucionador` | Nome do algoritmo utilizado |
| `seed` | Semente aleatória usada |
| `tempo_computacional_seg` | Tempo em segundos que a heurística levou para rodar |
| `lucro_liquido_G` | Função Objetivo $G$ final calculada pelo simulador |
| `tempo_viagem_f` | Tempo total de viagem $f(x, z)$ |
| `valor_bruto_itens_g` | Valor somado dos itens $g(z)$ |
| `peso_acumulado_mochila` | Peso total carregado pela mochila |
| `capacidade_mochila_W` | Capacidade máxima $W$ da mochila |
| `pct_ocupacao_mochila` | Taxa de ocupação da mochila ($\%$) |
| `qtd_itens_coletados` | Quantidade de itens coletados na solução |
| `total_itens_instancia` | Total de itens disponíveis na instância |
| `qtd_cidades` | Quantidade de cidades da instância |

---

## 🛡️ 7. Boas Práticas e Dicas para o Grupo

1. **Nunca altere os motores de avaliação (`src/core/evaluators.py`):** O cálculo da velocidade dinâmica e das funções objetivo de TTP1 e TTP2 é padronizado e mantido pelo simulador central.
2. **Use Sempre a Seed:** Em funções estocásticas dentro da sua heurística, utilize a semente (`seed`) recebida para garantir que seus resultados sejam 100% reprodutíveis.
3. **Não Faça Hardcode de Parâmetros:** Sempre obtenha os dados de capacidade da mochila, distâncias e itens a partir do objeto `TTPInstance` passado ao método `solve()`.
4. **Isolamento de Código:** Mantenha a implementação da sua heurística contida no seu próprio arquivo dentro de `src/solvers/`.