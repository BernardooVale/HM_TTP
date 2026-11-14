# Weighted-Sum Method (WSM) para o Traveling Thief Problem

Este documento detalha a arquitetura algorítmica, as formulações matemáticas e os procedimentos operacionais propostos por Jonatas B. C. Chagas e Markus Wagner no artigo *"A weighted-sum method for solving the bi-objective traveling thief problem"* (publicado na *Computers & Operations Research*, 2021). O foco deste relatório concentra-se estritamente na mecânica da solução, fornecendo especificações precisas para replicação em código e comparação experimental.

---

## 1. Fundamentação da Escalarização por Soma Ponderada

O problema abordado trata a formulação bi-objetivo do problema (BITTP), cujas metas simultâneas e conflitantes consistem em:
1. Maximizar o lucro bruto dos itens coletados:
   $$ \max g(z) = \sum_{j=1}^{m} p_j \cdot z_j $$
2. Minimizar o tempo total de viagem do ladrão:
   $$ \min h(\pi, z) = \sum_{i=1}^{n-1} \frac{d(\pi_i, \pi_{i+1})}{v_{\max} - \nu \cdot \omega(i, \pi, z)} + \frac{d(\pi_n, \pi_1)}{v_{\max} - \nu \cdot \omega(n, \pi, z)} $$
   onde $\nu = \frac{v_{\max} - v_{\min}}{W}$ e $\omega(i, \pi, z)$ representa o peso acumulado na mochila após a visita à cidade $\pi_i$.

### A Função de Escalarização Parametrizada

Para contornar a complexidade de otimizadores populacionais multiobjetivo convencionais (como NSGA-II) que frequentemente sofrem em instâncias combinatórias com interdependência não linear, os autores decompõem o espaço bi-objetivo em múltiplas instâncias mono-objetivo. Essa conversão é regida por um peso escalar $\alpha \in [0, 1]$ combinado diretamente à taxa de aluguel $R$ específica da instância:

$$ \max f(\pi, z, \alpha) = \alpha \cdot g(z) - (1 - \alpha) \cdot R \cdot h(\pi, z) $$

### Propriedades da Formulação
* **Interpretação Física dos Extremos de $\alpha$**:
  * Quando $\alpha \to 1$: o termo temporal perde relevância e o problema converge para um Problema da Mochila (KP) clássico de capacidade $W$.
  * Quando $\alpha \to 0$: a função busca estritamente a minimização do tempo de viagem ponderado pelo aluguel. O ladrão prioriza carregar mochila vazia ou quase vazia na rota mais curta possível.
  * Quando $\alpha = 0.5$: ambos os objetivos recebem pesos equilibrados, equivalendo proporcionalmente à formulação mono-objetivo clássica do TTP1 proposta por Polyakovskiy et al.
* **Superação da Não-Convexidade**: A soma ponderada exata é teoricamente incapaz de capturar soluções situadas em trechos não-convexos da fronteira de Pareto ótima. Contudo, como cada subproblema escalarizado é resolvido por heurísticas aproximadas (e não por resolvedores exatos), o caráter estocástico e a subotimalidade local das buscas permitem alcançar e mapear pontos também em regiões não-convexas do espaço de objetivos.

---

## 2. Arquitetura Global do Algoritmo WSM

O método é concebido em torno de um ciclo contínuo que equilibra **Exploração** (geração de novas soluções viáveis distribuídas globalmente) e **Explotação** (intensificação local em torno das soluções mais promissoras). O algoritmo mantém um conjunto arquivador $S$ que armazena unicamente soluções não-dominadas (aproximação da Fronteira de Pareto).

### Ciclo Operacional

1. **Fase de Exploração (Geração Global)**:
   * Uma nova rota $\pi$ é gerada pelo resolvedor Chained Lin-Kernighan (CLK).
   * Executa-se um loop de $\eta$ repetições onde:
     * Amostra-se $\alpha \sim \mathcal{D}$ (distribuição de probabilidade sobre $[0, 1]$).
     * Constrói-se um plano de empacotamento $z$ via `RandomizedPackingAlgorithm(\pi, \rho, \alpha, \gamma)`.
     * Atualiza-se o arquivo de soluções não-dominadas $S$ com o par $\langle \pi, z \rangle$.
2. **Fase de Explotação (Intensificação Local)**:
   * Amostra-se um novo valor de $\alpha \sim \mathcal{D}$.
   * Recupera-se do arquivo $S$ a melhor solução $\langle \pi', z' \rangle$ segundo a função escalarizada $f(\pi', z', \alpha)$ para atuar como pivô.
   * **Busca Local na Rota**: Aplica-se o operador 2-opt sobre $\pi'$ filtrando os movimentos com tolerância $\beta$. A melhor rota melhorada $\pi''$ é integrada ao arquivo $S$.
   * **Busca Local na Mochila**: Aplica-se o operador Bit-Flip amostrado com probabilidade $\lambda$ sobre os itens do plano $z'$, atualizando $S$ com os vizinhos não-dominados gerados.

---

## 3. Heurística de Empacotamento Randomizada (`RandomizedPackingAlgorithm`)

Dada uma rota fixa $\pi$, determinar a seleção ótima de itens permanece um problema NP-difícil. Os autores propõem uma extensão randomizada e adaptativa da heurística determinística `PackIterative` de Faulkner et al.

### Mecanismo de Pontuação com Três Expoentes Normalizados

Em cada tentativa de construção, o algoritmo sorteia três números reais aleatórios uniformes:
$$ a, b, c \sim U(0, 1) $$
Em seguida, normaliza-os de modo que estabeleçam frações percentuais estritas de importância:
$$ a \leftarrow \frac{a}{a+b+c}, \quad b \leftarrow \frac{b}{a+b+c}, \quad c \leftarrow \frac{c}{a+b+c} \quad \implies \quad a + b + c = 1 $$

O score $s_j$ de cada item $j \in \{1, \dots, m\}$ é calculado por:
$$ s_j = \frac{p_j^a}{w_j^b \cdot d_j^c} $$
onde:
* $p_j$: valor (lucro) do item.
* $w_j$: peso do item.
* $d_j$: distância residual da rota que o ladrão ainda terá que percorrer transportando o item $j$ desde a cidade onde ele é coletado até o retorno à cidade 1.

Ao contrário do `PackIterative` original (que fixa o expoente da distância em $1.0$ e varia apenas um parâmetro $\alpha$ sobre valor e peso), o algoritmo de Chagas & Wagner randomiza o impacto da distância restante ($c$). Isso permite descobrir planos de empacotamento com perfis de coleta radicalmente distintos em cada execução.

### Frequência Dinâmica de Checagem ($\phi$)

Avaliar a função objetivo completa $f(\pi, z, \alpha)$ consome tempo $O(n + m)$. Para mitigar esse custo:
1. Os itens são ordenados decrescentemente pelo score $s_j$.
2. O algoritmo define uma janela de checagem inicial $\phi$:
   $$ \phi = \left\lceil \frac{m}{\gamma} \cdot \alpha + \epsilon \right\rceil $$
   onde $\gamma$ é um parâmetro de ajuste e $\epsilon = 10^{-5}$ impede divisão por zero quando $\alpha = 0$.
3. Se $\alpha \to 0$ (foco em velocidade), $\phi$ assume valores muito pequenos: o algoritmo confere quase item a item se a inclusão não degrada o tempo de viagem.
4. Se $\alpha \to 1$ (foco em lucro), $\phi$ assume valores elevados: muitos itens são incluídos em bloco antes que uma reavaliação cara seja disparada.
5. **Mecanismo de Recuo e Poda**: Se após a adição de um lote de $\phi$ itens a função objetivo $f(\pi, z', \alpha)$ decair em relação a $f(\pi, z, \alpha)$, o bloco é descartado, o estado anterior é restaurado e a janela de busca é dividida pela metade:
   $$ \phi \leftarrow \lfloor \phi / 2 \rfloor $$
   Isso permite que o algoritmo teste inclusões mais cirúrgicas antes de encerrar o empacotamento.

Para maximizar a diversidade, todo esse processo é repetido $\rho$ vezes de forma independente para a mesma rota $\pi$, retornando o melhor plano $z_{\text{best}}$.

---

## 4. Fase de Explotação e Operadores de Busca Local

Após selecionar o pivô $\langle \pi', z' \rangle \in S$ com base em um novo $\alpha$ sorteado, duas buscas locais especializadas são aplicadas:

### 1. Refinamento de Rota: 2-OPT com Poda Geométrica por Tolerância ($\beta$)
O espaço de todas as permutações 2-opt em uma rota com $n$ cidades tem tamanho $\frac{n(n-1)}{2}$. Avaliar o impacto temporal da mochila em cada vizinho 2-opt é computacionalmente inviável para instâncias médias e grandes.

Para contornar esse gargalo, os autores introduzem um **filtro de estiramento de rota**:
* Seja $\bar{\ell}$ a distância geográfica média entre todos os pares de cidades da instância.
* Um candidato 2-opt $\pi'''$ só é avaliado na função objetivo se o aumento do seu comprimento geográfico total em relação ao tour original $\pi'$ não ultrapassar um limiar rígido:
  $$ d(\pi''') - d(\pi') \le \bar{\ell} \cdot \beta $$
* Se $\beta = -\infty$, o operador 2-opt é desativado.
* Dentre todas as rotas $\pi'''$ aprovadas pelo filtro que resultem em melhora estrita ($f(\pi''', z', \alpha) > f(\pi'', z', \alpha)$), seleciona-se a melhor rota $\pi''$, atualizando o arquivo $S$.

### 2. Refinamento de Empacotamento: Bit-Flip Amostrado ($\lambda$)
Testar a inversão do estado de coleta de todos os $m$ itens tornaria a busca local proibitivamente lenta em instâncias com dezenas de milhares de itens.

Por esse motivo, o operador aplica uma taxa de amostragem estocástica $\lambda \in [0, 0.5]$:
* Para cada item $j \in \{1, \dots, m\}$, sorteia-se um valor uniforme $r \sim U(0, 1)$.
* A inversão do bit só é processada se $r \le \lambda$:
  * Se $j \in z'$: remove o item da mochila e avalia a nova solução $\langle \pi', z' \setminus \{j\} \rangle$.
  * Se $j \notin z'$: adiciona o item apenas se o peso acumulado não violar a capacidade máxima ($W$). Se viável, avalia $\langle \pi', z' \cup \{j\} \rangle$.
* Qualquer solução resultante que não seja dominada pelas soluções já presentes no arquivo $S$ é imediatamente inserida no arquivo.

---

## 5. Pseudocódigos Formais

### Algoritmo 1: Weighted-Sum Method (WSM)

```text
Entrada: 
  D       : Distribuição de probabilidade para amostragem de alpha
  eta     : Número de planos de empacotamento gerados por rota (exploração)
  rho     : Número de repetições na heurística de empacotamento
  gamma   : Divisor de controle da frequência de checagem phi
  beta    : Coeficiente de tolerância de distância para o filtro 2-opt
  lambda  : Probabilidade de teste no operador bit-flip

Saída:
  S       : Conjunto de soluções não-dominadas (fronteira de Pareto aproximada)

1: S <- vazio
2: repita
3:    // --- FASE DE EXPLORAÇÃO ---
4:    pi <- Resolver_TSP_Chained_Lin_Kernighan()
5:    para k <- 1 até eta faça
6:        alpha <- Amostrar_Valor(D)
7:        z <- RandomizedPackingAlgorithm(pi, rho, alpha, gamma)
8:        Atualizar_Arquivo_Nao_Dominadas(S, <pi, z>)
9:    fim para
10:
11:   // --- FASE DE EXPLOTAÇÃO ---
12:   alpha <- Amostrar_Valor(D)
13:   <pi', z'> <- Obter_Melhor_Solucao(S, alpha)
14:   
15:   // Explotação da Rota via 2-opt com filtro beta
16:   Pi_vizinhos <- Gerar_Todos_Movimentos_2opt(pi')
17:   l_medio <- Calcular_Distancia_Media_Cidades()
18:   pi'' <- pi'
19:   para cada pi''' em Pi_vizinhos faça
20:       se (Distancia_Total(pi''') - Distancia_Total(pi')) <= (l_medio * beta) então
21:           se f(pi''', z', alpha) > f(pi'', z', alpha) então
22:               pi'' <- pi'''
23:           fim se
24:       fim se
25:   fim para
26:   se pi'' != pi' então
27:       Atualizar_Arquivo_Nao_Dominadas(S, <pi'', z'>)
28:   fim se
29:   
30:   // Explotação da Mochila via Bit-Flip amostrado
31:   para cada item j <- 1 até m faça
32:       se Random(0, 1) <= lambda então
33:           se j pertence a z' então
34:               Atualizar_Arquivo_Nao_Dominadas(S, <pi', z' \ {j}>)
35:           senão se Peso(z' união {j}) <= W então
36:               Atualizar_Arquivo_Nao_Dominadas(S, <pi', z' união {j}>)
37:           fim se
38:       fim se
39:   fim para
40: até que o critério de parada (ex: tempo limite) seja atingido
41: retorne S
```

---

### Algoritmo 2: Heurística de Empacotamento Randomizada (`RandomizedPackingAlgorithm`)

```text
Entrada:
  pi    : Rota fixa das cidades
  rho   : Número de tentativas independentes
  alpha : Escalar da função objetivo atual
  gamma : Parâmetro de escala da frequência phi

Saída:
  z_best: Melhor plano de empacotamento encontrado para a rota pi

1: z_best <- vazio
2: para attempt <- 1 até rho faça
3:    a_rand <- Random(0, 1), b_rand <- Random(0, 1), c_rand <- Random(0, 1)
4:    soma <- a_rand + b_rand + c_rand
5:    a <- a_rand / soma, b <- b_rand / soma, c <- c_rand / soma
6:    
7:    para cada item j <- 1 até m faça
8:        d_j <- Distancia_Restante_Ate_Fim_Do_Tour(pi, cidade_de(j))
9:        score[j] <- (p_j^a) / ((w_j^b) * (d_j^c))
10:   fim para
11:   
12:   phi <- teto((m / gamma) * alpha + 1e-5)
13:   z <- vazio
14:   z_cand <- vazio
15:   houve_inclusao <- falso
16:   k <- 1
17:   k_cand <- 1
18:   
19:   enquanto k_cand <= m e phi >= 1 faça
20:       j <- Obter_Item_Com_k_Esimo_Maior_Score(score, k_cand)
21:       se Peso(z_cand união {j}) <= W então
22:           z_cand <- z_cand união {j}
23:           houve_inclusao <- verdadeiro
24:       fim se
25:       
26:       se (k_cand mod phi == 0) e (houve_inclusao == verdadeiro) então
27:           se f(pi, z_cand, alpha) > f(pi, z, alpha) então
28:               z <- z_cand
29:               k <- k_cand
30:           senão
31:               z_cand <- z
32:               k_cand <- k
33:               phi <- piso(phi / 2)
34:           fim se
35:           houve_inclusao <- falso
36:       fim se
37:       k_cand <- k_cand + 1
38:   fim enquanto
39:   
40:   se f(pi, z, alpha) > f(pi, z_best, alpha) então
41:       z_best <- z
42:   fim se
43: fim para
44: retorne z_best
```

---

## 6. Configurações Calibradas e Recomendações de Parâmetros

No artigo, os autores submeteram o algoritmo ao calibrador automático **irace** sobre 96 grupos de instâncias (abrangendo desde 51 até 85.900 cidades). 

Para uso direto em comparações experimentais sem calibração prévia por instância, os autores derivaram uma **configuração geral recomendada** baseada na média/moda dos experimentos:

| Parâmetro | Descrição | Valor Recomendado | Comportamento Observado |
| :--- | :--- | :---: | :--- |
| $\mathcal{D}$ | Distribuição de amostragem de $\alpha$ | **$U(0, 1)$** | A distribuição uniforme cobriu o espectro de compromisso com robustez na quase totalidade das instâncias. |
| $\eta$ | Planos construídos por rota | **117** | Em instâncias grandes, resolver o CLK é caro; portanto, o irace aumentou $\eta$ para explorar muitos empacotamentos na mesma rota. |
| $\rho$ | Tentativas de empacotamento randomizado | **12** | Poucas tentativas já são suficientes para capturar combinações de expoentes ($a, b, c$) favoráveis. |
| $\gamma$ | Escala da frequência $\phi$ | **41** | Permite checagens frequentes sem comprometer o tempo de execução. |
| $\beta$ | Tolerância de estiramento do 2-opt | **0.001** | Filtro estrito: avalia apenas movimentos 2-opt que praticamente não aumentem a extensão geográfica do tour. |
| $\lambda$ | Probabilidade de Bit-Flip | **0.22** | Amostra cerca de 22% dos itens por iteração de explotação, mantendo o custo computacional baixo. |

### Diferencial Competitivo em Cenário Experimental
Em comparação direta contra 21 algoritmos consolidados da literatura de TTP mono-objetivo, esta abordagem baseada em soma ponderada estabeleceu **novos recordes de melhor solução conhecida em 379 instâncias**, demonstrando que decompor e escalarizar o problema com variações estocásticas de importância pode superar metaheurísticas que operam apenas sob formulações fixas.
