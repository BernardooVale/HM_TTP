# CoCo: Cooperative Coordination Solver para o Travelling Thief Problem

O artigo *"A Cooperative Coordination Solver for Travelling Thief Problems"* (Namazi, Sanderson, Newton e Sattar) introduz o **CoCo (*Cooperative Coordination Solver*)**, um framework de otimização projetado para solucionar a interdependência entre os componentes de rota (TSP) e de empacotamento (KP) no *Travelling Thief Problem* (TTP). O resolvedor combina duas contribuições complementares: uma versão estendida do **PGCH (*Profit Guided Coordination Heuristic*)** para o subproblema de rota e uma busca local direcionada denominada **Boundary Bit-Flip** para o subproblema da mochila.

Este documento detalha o funcionamento matemático e algorítmico do CoCo, suas estruturas de vizinhança, critérios de aceitação e pseudocódigos formais para fins de implementação e avaliação experimental.

---

## 1. Visão Geral e Arquitetura do CoCo

A maioria dos solucionadores do estado da arte para o TTP atua de forma intercalada (*interleaved*): otimiza-se a rota do TSP mantendo o plano de coleta fixo e, em seguida, otimiza-se o plano de coleta do KP mantendo a rota fixa. Essa abordagem sofre de dois gargalos severos de coordenação:

1. **Rejeição Prematura de Movimentos 2-OPT:** Ao inverter um segmento da rota pelo operador 2-OPT tradicional mantendo o plano de coleta inalterado, os itens de menor rentabilidade coletados no fim do segmento original passam a ser transportados logo no início do segmento invertido. Isso derruba a velocidade do ladrão por quase todo o percurso e gera uma explosão de custo de aluguel ($R \times T$), fazendo com que melhorias topológicas na rota sejam descartadas sumariamente por gerarem benefício líquido negativo.
2. **Lentidão do Bit-Flip Tradicional:** O operador convencional de *bit-flip* (utilizado no *Random Local Search* e em algoritmos meméticos) seleciona itens de maneira aleatória ou cega. Como qualquer alteração no bit de um item exige recalcular o tempo de viagem das cidades subsequentes até o fim da rota ($O(n)$), explorar o espaço de busca da mochila torna-se lento e disperso, consumindo a maior parte do orçamento de tempo computacional.

O framework CoCo resolve ambos os problemas por meio de uma cooperação síncrona entre dois módulos especializados:

* **`TSPSolver` (Otimização da Rota com PGCH Estendido):** Substitui o 2-OPT cego por um movimento coordenado que inverte o trecho de cidades e, simultaneamente, substitui itens de baixa rentabilidade no início do trecho por itens de alta rentabilidade no final dele.
* **`KPSolver` (Otimização da Mochila com Boundary Bit-Flip):** Restringe as inversões de bits estritamente aos *itens de fronteira* (itens que moldam os perfis extremos de rentabilidade acumulada). Essa restrição acelera significativamente o tempo de processamento por ciclo do KP, transferindo tempo de CPU para que o `TSPSolver` possa avaliar mais rotas.

---

## 2. PGCH Estendido (Extended Profit Guided Coordination Heuristic)

No artigo original de conferência (2019), o PGCH utilizava apenas a curva dos itens coletados de menor rentabilidade para efetuar o descarte (*unpicking*). No artigo estendido (*coco.pdf*), os autores introduzem o **PGCH Estendido**, que sincroniza tanto o descarte quanto a nova coleta de itens utilizando duas funções matemáticas: o **prefixo-mínimo** e o **posfixo-máximo**.

### 2.1 Funções Matemáticas de Rentabilidade

Considere uma solução atual $\langle c, p \rangle$, onde $c = \langle c_1, c_2, \dots, c_n \rangle$ representa a permutação das cidades visitadas e $p = \langle p_1, p_2, \dots, p_m \rangle$ representa o vetor binário de itens coletados ($p_j \in \{0, 1\}$). Cada item $j$ possui peso $w_j$, lucro $\pi_j$ e razão de rentabilidade $r_j = \pi_j / w_j$.

Para cada cidade $c_k$ na posição $k$ da rota ($1 \le k \le n$):
* Seja $p(k)$ o item coletado em $c_k$ com a **menor** razão de rentabilidade. A sequência correspondente de razões é denotada por $P_{c,p}(k) = r_{p(k)}$. Caso nenhum item seja coletado na cidade $c_k$, adota-se o valor padrão superior:
  $$P_{c,p}(k) = 1 + \max_{1 \le i \le m} r_i$$
* Seja $q(k)$ o item não coletado em $c_k$ com a **maior** razão de rentabilidade. A sequência correspondente de razões é denotada por $Q_{c,p}(k) = r_{q(k)}$. Caso todos os itens da cidade $c_k$ estejam coletados (ou a cidade não possua itens), adota-se o valor padrão inferior:
  $$Q_{c,p}(k) = 0$$

A partir dessas sequências, definem-se as funções acumuladas:
* **Função Prefixo-Mínimo $\Pi(S, k)$:** Retorna o menor valor da sequência $S$ até a posição $k$:
  $$\Pi(S, k) = \min(\Pi(S, k - 1), S(k)), \quad \text{com } \Pi(S, 1) = S(1)$$
* **Função Posfixo-Máximo $\Omega(S, k)$:** Retorna o maior valor da sequência $S$ a partir da posição $k$ até o final da sequência ($n$):
  $$\Omega(S, k) = \max(S(k), \Omega(S, k + 1)), \quad \text{com } \Omega(S, n) = S(n)$$

Nas soluções de alta qualidade do TTP, a sequência $\Pi(P_{c,p}, k)$ tende a decrescer ao longo do percurso (itens de menor rentabilidade só são tolerados no final da rota), enquanto a sequência $\Omega(Q_{c,p}, k)$ tende a crescer quando analisada em ordem reversa (do fim para o início).

### 2.2 Mecânica Operacional do PGCH Estendido

Dado um par de posições de corte $k'$ e $k''$ (com $1 < k' < k'' \le n$):

1. **Reversão Topológica:** Inverte-se o segmento de cidades entre $k'$ e $k''$:
   $$c'_{k' + k} = c_{k'' - k}, \quad \forall k \in [0, k'' - k']$$
2. **Fase de Descarte (*Uncollection / Unpicking*):** Percorre-se o trecho invertido na ordem direta (de $k = k'$ até $k''$). Para cada cidade $c'_k$, identifica-se qualquer item $j$ atualmente coletado ($p'_j = 1$). O item é desmarcado da mochila ($p'_j \leftarrow 0$) se sua rentabilidade for estritamente menor que a referência histórica de prefixo-mínimo original daquela posição:
   $$r_j < \Pi(P_{c,p}, k)$$
   Isso remove imediatamente os itens pesados e pouco rentáveis que caíram no início do segmento invertido.
3. **Fase de Coleta Compensatória (*Collection / Picking*):** Percorre-se o trecho invertido em ordem reversa (de $k = k''$ descendo até $k'$). Para cada cidade $c'_k$, examina-se qualquer item $j$ não coletado ($p'_j = 0$). O item é inserido na mochila ($p'_j \leftarrow 1$) se sua rentabilidade for estritamente maior que a referência de posfixo-máximo original daquela posição:
   $$r_j > \Omega(Q_{c,p}, k)$$
   **Restrição Estrita de Capacidade e Peso:** A inserção de novos itens só é permitida desde que o peso acumulado dos itens adicionados **não ultrapasse** o peso total dos itens descartados na etapa anterior:
   $$\sum_{j \in \text{adicionados}} w_j \le \sum_{j \in \text{descartados}} w_j$$
   Essa regra preserva a viabilidade da mochila e impede que a velocidade do ladrão seja penalizada além da condição inicial.

### 2.3 Pseudocódigo do PGCH Estendido

```text
Algoritmo: PGCH(c, p, k', k'')
Entrada: Rota atual c, Plano de coleta p, Índices de corte k', k'' (1 < k' < k'' <= n)
Saída: Rota candidata c', Novo plano de coleta p'

1.  c' <- c
2.  p' <- p
3.  W_uncollected <- 0
4.  W_collected <- 0

5.  // 1. Reversão do segmento de rota (2-OPT)
6.  para k de 0 até (k'' - k') faça:
7.      c'[k' + k] <- c[k'' - k]

8.  // 2. Fase de Descarte (Uncollection) usando Prefixo-Mínimo
9.  para k de k' até k'' faça:
10.     cidade <- c'[k]
11.     para cada item j localizado na cidade com p'[j] == 1 faça:
12.         se r[j] < Pi(P_{c,p}, k) então:
13.             p'[j] <- 0
14.             W_uncollected <- W_uncollected + w[j]

15. // 3. Fase de Coleta (Collection) usando Posfixo-Máximo
16. para k de k'' decrescendo até k' faça:
17.     cidade <- c'[k]
18.     para cada item j localizado na cidade com p'[j] == 0 faça:
19.         se r[j] > Omega(Q_{c,p}, k) e (W_collected + w[j] <= W_uncollected) então:
20.             p'[j] <- 1
21.             W_collected <- W_collected + w[j]

22. retornar <c', p'>
```

---

## 3. Boundary Bit-Flip Search (KPSolver)

O subproblema de empacotamento com rota fixa (KRP) requer a determinação de um plano binário $p \in \{0, 1\}^m$ que maximize a função objetivo global $G(c, p) = P(p) - R \cdot T(c, p)$.

### 3.1 Definição Matemática dos Itens de Fronteira

Em vez de testar todos os $m$ itens da instância de maneira uniforme, o algoritmo restringe as alterações aos **itens de fronteira (*Boundary Items*)**, definidos como os elementos que sustentam os valores extremos das funções de prefixo-mínimo e posfixo-máximo:

1. **Item Coletado de Fronteira:** O item $p(k)$ coletado na cidade $c_k$ é um item de fronteira se ele for o item de menor rentabilidade entre todos os itens coletados até a posição $k$:
   $$\Pi(P_{c,p}, k) = P_{c,p}(k) = r_{p(k)}$$
   *Interpretação:* Este item representa o "pior gargalo" de rentabilidade até aquele momento; retirá-lo da mochila é a alternativa mais provável para restaurar a velocidade do ladrão sem perder rentabilidade proporcional.
2. **Item Não-Coletado de Fronteira:** O item $q(k)$ não coletado na cidade $c_k$ é um item de fronteira se ele for o item de maior rentabilidade entre todos os itens não coletados desde a posição $k$ até o término da rota ($n$):
   $$\Omega(Q_{c,p}, k) = Q_{c,p}(k) = r_{q(k)}$$
   *Interpretação:* Este item é o candidato mais lucrativo disponível na porção final do percurso; adicioná-lo à mochila proporciona alto ganho bruto com penalidade temporal reduzida.

### 3.2 Estrutura da Busca Local e Gestão da Bolsa de Itens

O procedimento do `KPSolver` gerencia uma estrutura dinâmica (uma bolsa ou lista $b$) que armazena os itens de fronteira da solução corrente:

1. Todos os itens de fronteira são identificados pela rotina `BoundaryItems(c, p)` e inseridos em $b$.
2. Todos os itens em $b$ são marcados como **não verificados** (`unchecked`).
3. Enquanto houver itens não verificados em $b$:
   * Sorteia-se aleatoriamente um item não verificado $j \in b$ e marca-se como verificado (`checked`).
   * Aplica-se a inversão de estado: $p'_j \leftarrow 1 - p_j$ via operador `Flip(p, j)`.
   * Se o novo plano $p'$ for viável ($\sum w_i p'_i \le W$) e produzir uma melhora estrita no benefício global ($G(c, p') > G(c, p)$):
     * A alteração é aceita: $p \leftarrow p'$.
     * O conjunto de itens de fronteira é recalculado via `Update(b)`, e todos os elementos de $b$ voltam ao estado de **não verificados**.
4. Quando todos os itens de $b$ tiverem sido avaliados sem que nenhuma melhora seja encontrada, o `KPSolver` atinge um ótimo local e retorna o plano refinado $p$.

O custo para atualizar as funções $\Pi(P_{c,p}, k)$, $\Omega(Q_{c,p}, k)$ e o conjunto $b$ após uma inversão aceita é de $O(n)$, idêntico ao custo de avaliação do operador bit-flip isolado.

```text
Algoritmo: KPSolver(c, p)
Entrada: Rota fixa c, Plano de coleta p
Saída: Plano de coleta melhorado p

1.  b <- BoundaryItems(c, p)
2.  MarkUnCheckedAll(b)

3.  enquanto not AllChecked(b) faça:
4.      j <- RandUnCheckedItem(b)
5.      MarkChecked(j)
6.      p' <- Flip(p, j)
7.      se PesoTotal(p') <= W e G(c, p') > G(c, p) então:
8.          p <- p'
9.          b <- BoundaryItems(c, p)
10.         MarkUnCheckedAll(b)

11. retornar p
```

---

## 4. O Solucionador CoCo Completo (CoCoSolver)

O algoritmo **CoCo** integra o `TSPSolver` (baseado em subida de encosta mais íngreme com PGCH Estendido) e o `KPSolver` (baseado em Boundary Bit-Flip) sob um laço intercalado de reinícios múltiplos (*multi-start*).

### 4.1 Estrutura do TSPSolver com Triangulação de Delaunay

O `TSPSolver` utiliza a estratégia de **subida de encosta de maior inclinação (*steepest ascent hill-climbing*)**:
* Em vez de testar todas as $O(n^2)$ inversões possíveis, o espaço de busca é restrito pela **Triangulação de Delaunay** das coordenadas das cidades. Para cada cidade $c_{k'}$, apenas as cidades $c_{k''}$ conectadas a ela no grafo de Delaunay ($k' < k'' \le n$) são avaliadas.
* O algoritmo examina todas as posições $k'$ e seus vizinhos de Delaunay $k''$, aplicando `PGCH(c, p, k', k'')`. Dentre todos os movimentos que melhoram a solução, **apenas o movimento de maior ganho global ($\langle c^\ddagger, p^\ddagger \rangle$) é efetivamente aplicado**.
* **Critério de Parada com Limiar $\alpha$:** O ciclo de varredura do `TSPSolver` repete-se enquanto o ganho relativo obtido for significativo:
  $$G(c, p) - G_{prev} \ge \alpha \cdot G_{prev}, \quad \text{onde } \alpha = 0{,}01\% \ (0{,}1 \times 10^{-3})$$
  Esse limiar evita desperdício de processamento em ganhos residuais infundados em instâncias de grande escala.

### 4.2 Lógica do CoCoSolver e Critério de Aborto

O algoritmo executa os seguintes passos principais:

1. **Inicialização:** Gera uma rota inicial através da heurística *Chained Lin-Kernighan* (CLK). O plano de coleta inicial é obtido executando as heurísticas construtivas *PackIterative* e *Insertion*, escolhendo aquele que fornecer o maior ganho global $G(c, p)$.
2. **Loop de Refinamento Intercalado:**
   * Executa $\langle c, p \rangle \leftarrow \text{TSPSolver}(c, p)$.
   * Registra o ganho após a otimização da rota: $G_{TSP} \leftarrow G(c, p)$.
   * Executa $p \leftarrow \text{KPSolver}(c, p)$.
   * **Critério de Escape:** Se após a execução do `KPSolver` o ganho não aumentar ($G(c, p) = G_{TSP}$), a busca local para a solução atual converge e o loop interno é abortado.
3. **Reinício Global (*Multi-Start*):** A melhor solução global $\langle c^*, p^* \rangle$ é atualizada e o resolvedor reinicia com uma nova rota CLK enquanto houver tempo disponível no orçamento global.

```text
Algoritmo: CoCoSolver(tempo_limite)
Entrada: Instância do TTP, Limite global de tempo
Saída: Melhor solução global <c*, p*>

1.  <c*, p*> <- nulo
2.  enquanto tempo_decorrido() < tempo_limite faça:
3.      c <- ChainedLKTour()
4.      p <- InitCollectionPlan(c)  // Melhor entre PackIterative e Insertion
5.      
6.      enquanto tempo_decorrido() < tempo_limite faça:
7.          <c, p> <- TSPSolver(c, p)
8.          G_TSP <- G(c, p)
9.          p <- KPSolver(c, p)
10.         se G(c, p) == G_TSP então:
11.             quebrar  // Escape do loop interno por falta de progresso
12.     
13.     se <c*, p*> == nulo ou G(c, p) > G(c*, p*) então:
14.         <c*, p*> <- <c, p>

15. retornar <c*, p*>
```

---

## 5. Avaliação Experimental e Ablação dos Componentes

Para comprovar formalmente a contribuição isolada do PGCH e do Boundary Bit-Flip, os autores de *coco.pdf* avaliaram quatro configurações variantes do algoritmo sob o mesmo protocolo de teste (tempo limite de 10 minutos por instância, 10 execuções independentes, métrica RDI - *Relative Deviation Index*):

| Variante | Operador TSP | Operador KP | Descrição Metodológica |
| :--- | :--- | :--- | :--- |
| **Solver 1** | 2-OPT Tradicional | Bit-Flip Padrão | Baseline clássica com fraca coordenação. |
| **Solver 2** | 2-OPT Tradicional | Boundary Bit-Flip | Isola o efeito do filtro de fronteira no KP. |
| **Solver 3** | PGCH Estendido | Bit-Flip Padrão | Isola o efeito da coordenação de rota/plano. |
| **Solver 4 (CoCo)** | PGCH Estendido | Boundary Bit-Flip | Solucionador completo com co-otimização coordenada. |

### Principais Conclusões dos Experimentos

1. **Superioridade do PGCH (Solver 3 e 4 vs. Solver 1 e 2):** Solvers com PGCH apresentaram RDI médio expressivamente superior em todas as categorias (A, B e C). Nas análises de rastreamento de movimentos, o PGCH obteve uma taxa muito superior de aceitação de trocas e o comprimento médio dos segmentos invertidos aceitos subiu de 5%–16% (com 2-OPT comum) para 27%–31% das cidades da rota.
2. **Impacto do Boundary Bit-Flip por Categoria:**
   * **Categorias B e C (5 a 10 itens por cidade):** O Boundary Bit-Flip proporcionou ganhos estatisticamente significantes (RDI subiu de 79,5% para 88,7% na categoria B, e de 75,7% para 82,3% na categoria C). A redução do espaço de busca no KP permitiu ao algoritmo executar significativamente mais iterações do `TSPSolver`.
   * **Categoria A (1 item por cidade, capacidade restrita):** O Boundary Bit-Flip causou uma pequena degradação (RDI médio de 71,6% contra 74,8% do bit-flip padrão). Como há extrema escassez de itens na categoria A, podar a busca exclusivamente para itens de fronteira restringiu excessivamente o espaço viável.
3. **Comparação com o Estado da Arte:** O CoCo completo superou solvers consolidados como MATLS, S5, CS2SA* e MEA2P em praticamente todas as instâncias de médio e grande porte avaliadas, estabelecendo dezenas de novos recordes de lucro (*best known solutions*) para o benchmark de Adelaide.

---

## 6. Considerações para Implementação

Ao codificar o CoCo, três otimizações estruturais devem ser observadas:

1. **Pré-cálculo da Vizinhança de Delaunay:** O grafo de Delaunay das cidades deve ser gerado uma única vez na inicialização (utilizando bibliotecas geométricas padrão como `scipy.spatial.Delaunay`). A lista de adjacências de cada cidade serve como filtro direto para os pares $(k', k'')$ no `TSPSolver`.
2. **Atualização Incremental de Tempo e Peso:** O recálculo da função objetivo $G(c', p')$ após um movimento PGCH não deve refazer a rota inteira a partir do início. Como as cidades antes de $k'$ não sofrem alteração, o acumulador de tempo $T(c', p', k')$ e peso $W(c', p', k')$ até a posição $k'$ permanece idêntico, bastando propagar a soma dos tempos da posição $k'$ até $n + 1$.
3. **Indexação dos Itens de Fronteira:** Manter ponteiros diretos para os itens que satisfazem as condições extremas em cada cidade reduz a identificação de $\Pi(P_{c,p}, k)$ e $\Omega(Q_{c,p}, k)$ para um custo amortizado de $O(n)$, viabilizando execuções de milissegundos mesmo em problemas com milhares de cidades.
