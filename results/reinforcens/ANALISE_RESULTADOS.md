# Análise Experimental — ReinforceNS vs ReinforceNS-PM (18 Instâncias)

Este diretório reúne os resultados completos do protocolo experimental das **Hipóteses 1 e 2** do artigo *"Heurísticas para o Travelling Thief Problem"*, executados sobre as **18 instâncias** propostas na Seção 3 (`eil76`, `kroA100`, `ch130`, `u159`, `a280` e `u574` nas categorias `CatA`, `CatB` e `CatC`), com **5 repetições pareadas por instância** (**90 execuções pareadas** ao todo), **2 execuções-piloto** por instância para definição da escala $S_i$ e orçamento de **$T_{\max} = 5{,}0\text{s}$** por chamada (`--clk-kicks 0`).

- **Tabela pareada (H1 e H2):** [`experimento_reinforcens_18_instancias.csv`](experimento_reinforcens_18_instancias.csv)
- **Dados consolidados para o dashboard:** [`experimento_reinforcens_18_instancias_visualizador.csv`](experimento_reinforcens_18_instancias_visualizador.csv)

---

## 1. Resumo Executivo

### Hipótese 1 (`ReinforceNS` com *Q-Learning* vs `ReinforceNS-PM` com Primeira Melhoria)
- **Placar pareado (90 execuções):** `ReinforceNS-PM` venceu em **40** execuções, empatou em **7** e `ReinforceNS` venceu em **43**.
- **Diferença média normalizada ($\Delta = 100 \times \frac{f_{PM} - f_Q}{S_i}$):** **`-0,079%`**
- **IC 95% Bootstrap (10.000 reamostragens):** **`[-0,867%, +0,667%]`**
- **Critério de não-inferioridade (limite inferior $> -1{,}000\%$):** **ATENDIDO**.

### Hipótese 2 (Ablação Fatorial $2 \times 2$ do Pós-Processamento sobre `P0`)
Partindo exatamente da mesma solução saída da busca local `RLNS` (`P0`):

| Configuração | Etapas | Ganho médio sobre `P0` | Mediana | Máximo | Melhorou em |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`P1`** | Apenas `IFP` (*Item Flipping*) | **`+1,040%`** | `+0,408%` | `+7,159%` | **84 / 90** |
| **`P2`** | Apenas `RRP` (*Route Refining*) | **`+0,102%`** | `+0,000%` | `+1,982%` | **26 / 90** |
| **`P3`** | `IFP + RRP` (completo) | **`+1,155%`** | `+0,596%` | `+7,159%` | **89 / 90** |

---

## 2. Resultados Médios por Instância (5 Repetições Pareadas)

| Instância | Inicial ($f_0$) | `ReinforceNS` ($f_Q$) | `ReinforceNS-PM` ($f_{PM}$) | $\Delta_{H1}$ (%) | Starts ($Q$ / $PM$) | Ganho `P1` | Ganho `P2` | Ganho `P3` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `eil76_n75_bounded-strongly-corr_01` (CatA) | `3.257,97` | `3.685,10` | **`3.886,09`** | `+5,516%` | `1,8 / 19,6` | `+0,640%` | `+1,011%` | `+1,752%` |
| `eil76_n375_uncorr-similar-weights_05` (CatB) | `17.504,71` | `18.264,54` | **`18.913,50`** | `+3,525%` | `1,0 / 11,6` | `+0,650%` | `+0,000%` | `+0,650%` |
| `eil76_n750_uncorr_10` (CatC) | `74.999,88` | `76.025,06` | `76.025,06` | `0,000%` | `1,0 / 12,0` | `+0,529%` | `+0,000%` | `+0,529%` |
| `kroA100_n99_bounded-strongly-corr_01` (CatA) | `4.232,89` | `4.425,96` | **`4.449,18`** | `+0,522%` | `1,0 / 23,4` | `+1,656%` | `+0,000%` | `+1,656%` |
| `kroA100_n495_uncorr-similar-weights_05` (CatB) | `39.600,42` | **`42.433,72`** | `41.532,30` | `-2,110%` | `1,0 / 11,6` | `+1,038%` | `+0,000%` | `+1,038%` |
| `kroA100_n990_uncorr_10` (CatC) | `148.564,65` | `153.646,90` | **`154.152,78`** | `+0,330%` | `1,0 / 7,6` | `+0,257%` | `+0,000%` | `+0,257%` |
| `ch130_n129_bounded-strongly-corr_01` (CatA) | `8.489,83` | `8.869,65` | **`8.952,23`** | `+0,943%` | `1,0 / 2,6` | `+0,978%` | `+0,305%` | `+1,286%` |
| `ch130_n645_uncorr-similar-weights_05` (CatB) | `43.316,99` | **`51.406,63`** | `49.182,53` | `-4,644%` | `1,0 / 2,8` | `+2,212%` | `+0,053%` | `+2,266%` |
| `ch130_n1290_uncorr_10` (CatC) | `179.487,59` | **`188.859,33`** | `184.794,26` | `-2,224%` | `1,0 / 1,0` | `+0,447%` | `+0,000%` | `+0,447%` |
| `u159_n158_bounded-strongly-corr_01` (CatA) | `7.678,69` | `8.478,69` | **`8.668,66`** | `+2,317%` | `1,0 / 1,8` | `+3,324%` | `+0,112%` | `+3,519%` |
| `u159_n790_uncorr-similar-weights_05` (CatB) | `54.431,82` | **`59.730,16`** | `58.813,89` | `-1,536%` | `1,0 / 1,4` | `+0,153%` | `+0,000%` | `+0,153%` |
| `u159_n1580_uncorr_10` (CatC) | `231.837,67` | **`248.133,31`** | `244.030,41` | `-1,662%` | `1,0 / 3,0` | `+0,364%` | `+0,000%` | `+0,364%` |
| `a280_n279_bounded-strongly-corr_01` (CatA) | `14.612,79` | `16.545,09` | **`16.670,83`** | `+0,773%` | `1,0 / 2,4` | `+1,658%` | `+0,146%` | `+1,819%` |
| `a280_n1395_uncorr-similar-weights_05` (CatB) | `81.848,35` | **`96.134,68`** | `95.135,76` | `-1,047%` | `1,0 / 2,0` | `+1,696%` | `+0,024%` | `+1,721%` |
| `a280_n2790_uncorr_10` (CatC) | `370.786,27` | **`380.758,17`** | `380.374,37` | `-0,101%` | `1,0 / 1,0` | `+0,199%` | `+0,000%` | `+0,199%` |
| `u574_n573_bounded-strongly-corr_01` (CatA) | `23.735,65` | **`25.276,82`** | `24.930,12` | `-1,365%` | `1,0 / 1,0` | `+2,332%` | `+0,150%` | `+2,522%` |
| `u574_n2865_uncorr-similar-weights_05` (CatB) | `211.471,08` | **`233.594,05`** | `233.108,34` | `-0,209%` | `1,0 / 1,0` | `+0,301%` | `+0,031%` | `+0,318%` |
| `u574_n5730_uncorr_10` (CatC) | `887.246,59` | **`914.913,76`** | `910.818,53` | `-0,448%` | `1,0 / 1,0` | `+0,291%` | `+0,000%` | `+0,291%` |

---

## 3. Interpretação dos Achados para o Artigo

1. **Equivalência global e transição de regime por tamanho da instância (Hipótese 1):**
   - Na média global das 18 instâncias, o `ReinforceNS-PM` é **não-inferior** ao `ReinforceNS` original ($\bar{\Delta} = -0{,}079\%$, IC 95% $[-0{,}867\%, +0{,}667\%]$).
   - **Por que o comportamento muda com $n$ e $m$?**
     - Em instâncias menores (`eil76`, `kroA100`), a Primeira Melhoria converge para um ótimo local 2-opt em poucas passagens e consegue executar **12 a 23 ciclos *multi-start*** dentro de $5\text{s}$ (contra `1,0–1,8` do *Q-Learning*), explorando mais configurações $(\kappa, \beta, \gamma, \delta)$ do `IGP` e vencendo com folga (`+3,5%` a `+5,5%` na `eil76`).
     - Já nas instâncias maiores ou mais densas (`ch130`, `u159`, `a280` CatB/CatC e **todas as 3 `u574`**), onde ambos os métodos completam apenas **1 a 3 *starts*** dentro de $5\text{s}$, o **`ReinforceNS` com *Q-Learning* supera o `PM` na maioria dos casos**, porque consegue escapar do primeiro ótimo local 2-opt ao longo das passagens guiadas por $Q(i,j)$.

2. **Papel complementar de `IFP` e `RRP` no Pós-Processamento (Hipótese 2):**
   - O **`IFP` (`P1`) é o principal responsável pelo ganho do pós-processamento** (`+1,040%` em média, melhorando em 84/90 execuções), pois remove a trava artificial de capacidade $\delta \cdot W$ usada durante o `IGP` e o `MOV2OPTREEMP`, permitindo preencher a mochila até $100\%$ de $W$ conforme a função objetivo exata.
   - O **`RRP` (`P2`) atua como ajuste fino dependente da categoria da instância**: em `CatC` (10 itens por cidade), quase todas as cidades têm algum item coletado, então raramente existem cidades vazias para realocar (`+0,000%`). Já em `CatA` (`bounded-strongly-corr_01`, apenas 1 item por cidade), sobram mais cidades sem coleta e o `RRP` gera ganhos consistentes (ex.: `+1,011%` na `eil76`, `+0,305%` na `ch130`, `+0,150%` na `u574`).
   - Combinados em **`P3` (`IFP + RRP`)**, os dois estágios produzem ganho positivo em **89 das 90 execuções** (**`+1,155%`** em média).
