# 📊 Módulo de Avaliação do Impacto de $R$ no Travelling Thief Problem (TTP)

Este módulo foi desenvolvido para investigar rigorosamente o impacto da taxa de aluguel $R$ (*renting rate*) no equilíbrio entre o **Knapsack Problem (KP)** e o **Travelling Salesperson Problem (TSP)** dentro do TTP.

---

## 🎯 Fundamentação Teórica e Metodologia

No TTP padrão (TTP1), a função objetivo a ser maximizada é:
$$Z(P, z) = G(z) - R \cdot T(P, z)$$

- Se $R \to 0$: a penalidade de tempo é nula e o problema degenera para o **0-1 Knapsack Problem (KP)** puro.
- Se $R \to \infty$: o custo de viagem domina e o ladrão deve minimizar $T(P, z)$, degenerando para o **TSP com mochila quase vazia**.

### Abordagens Comparadas
Para cada instância e para cada valor de $R$ em uma grade log-linear ($f \times R_{nominal}$), comparamos:

1. **$Z^*(TTP)$ (Ótimo Global do TTP):** Encontrado via Branch & Bound com poda por limitante superior $UB(P) = KP_{opt} - R \cdot \frac{D(P)}{v_{max}}$.
2. **KP-First Ótimo Condicionado:** 0-1 Knapsack resolvido à otimalidade global + Rota ótima encontrada via Bitmask DP minimizando o tempo sob esses itens fixos.
3. **KP-First Guloso:** 0-1 Knapsack ótimo + Rota construída via heurística gulosa (Nearest Neighbor).
4. **TSP-First Ótimo Condicionado:** Rota TSP ótima encontrada via Held-Karp Bitmask DP + Empacotamento ótimo sob a rota fixa resolvido via Branch & Bound para o $R$ atual.
5. **TSP-First Guloso:** Rota TSP ótima + Empacotamento guloso sob a rota (heurística S2 ajustada à distância restante).

### Métricas Calculadas
- **Razão Relativa ($\rho$):** Proporção da pontuação da abordagem em relação ao ótimo global: $\rho = \frac{Z}{Z^*(TTP)}$.
- **Preço do Desacoplamento / Gap de Interdependência ($\Delta$):**
  $$\Delta(R) = 1 - \max(\rho_{KP}, \rho_{TSP})$$
  Indica quanto se perde ao resolver os subproblemas de forma separada versus resolvê-los integradamente.
- **Ponto de Cruzamento ($R_c$):** Valor de $R$ onde $|Z(S_{KP}) - Z(S_{TSP})|$ é mínimo (o ponto de virada de dominância).
- **Ponto de Máxima Interdependência ($R_{\Delta max}$):** Ponto onde $\Delta(R)$ atinge seu pico.
- **Intervalos de Limiar (ex: $\ge 0{,}90$):**
  - Faixa de $R$ onde a abordagem focada em KP atinge $\ge 90\%$ do ótimo do TTP.
  - Faixa de $R$ onde a abordagem focada em TSP atinge $\ge 90\%$ do ótimo do TTP.
  - **Zona de Acoplamento Forte:** Faixa de $R$ onde *nenhuma* das abordagens desacopladas atinge 90%, evidenciando a necessidade estrita de resolução integrada.

---

## 🛡️ Robustez Operacional

1. **Interrupção Segura (Ctrl+C):**
   - Ao pressionar `Ctrl+C`, o programa intercepta o sinal `SIGINT`, conclui a instância que está em execução no momento, salva atomicamente todos os resultados em disco e encerra graciosamente.
   - Pressionar `Ctrl+C` uma segunda vez força a saída imediata.
2. **Retomada Automática (Resume):**
   - O runner verifica automaticamente quais instâncias em `ttp_simulator/bateria_testes` já foram processadas e salvas em `avaliacao_r/resultados/detalhes/`.
   - Ao executar novamente, ele pula as já calculadas e retoma da primeira pendente.
3. **Escrita Atômica:**
   - Todos os arquivos JSON e CSV são gravados em arquivos temporários `.tmp` e substituídos atomicamente via `os.replace`, prevenindo qualquer corrupção em caso de desligamento.
4. **Proteção contra Explosão Combinatória:**
   - Para instâncias com mais de 10 cidades ou que excedam o timeout (configurável via `--timeout`), o solver utiliza a melhor cota incumbente entre os ótimos condicionados e sinaliza o status `TIMEOUT_HEURISTICO`.

---

## 🚀 Como Executar

### Execução Padrão de Toda a Bateria
```bash
python -m avaliacao_r.runner
```
ou:
```bash
python avaliacao_r/runner.py
```

### Executar Apenas uma Instância Específica
```bash
python -m avaliacao_r.runner --instancia instancia_01
```

### Configurar Timeout e Limiar Customizado
```bash
python -m avaliacao_r.runner --timeout 60.0 --limiar 0.85
```

### Forçar Recálculo (Ignorar Histórico Salvo)
```bash
python -m avaliacao_r.runner --forcar
```

---

## 📂 Estrutura de Saída (`avaliacao_r/resultados/`)

- `relatorio_geral_r.csv`: Tabela consolidada com uma linha por instância contendo dimensões, $R_c$, $R_{\Delta max}$, faixas de $\ge 90\%$ e zona de forte acoplamento.
- `curvas_r_detalhadas.csv`: Dados de todos os pontos de $R$ avaliados para cada instância (para geração de gráficos no Excel, Python ou R).
- `detalhes/*.json`: Arquivo JSON estruturado contendo todos os dados e soluções de cada instância individualmente.

---

## 🧪 Testes Automatizados

Para rodar a suíte de testes do módulo:
```bash
python -m pytest avaliacao_r/tests -v
```
