import math
from typing import Dict, List
from src.core.models import TTPInstance, SolucaoTTP, Item, Grafo
from src.solvers.base import BaseSolver, SolverRegistry


def resolver_tsp_farthest_insertion(grafo: Grafo) -> List[int]:
    """
    Resolve o TSP utilizando a heurística Farthest Insertion.
    Retorna a rota em ciclo fechado iniciando na cidade 1.
    """
    cidades = list(grafo.cidades.keys())
    n = len(cidades)
    if n <= 2:
        idx_1 = cidades.index(1) if 1 in cidades else 0
        return cidades[idx_1:] + cidades[:idx_1]

    # Encontra o par de cidades mais distante para inicializar o subtour
    max_d = -1.0
    u_init, v_init = cidades[0], cidades[1]
    for i in range(n):
        for j in range(i + 1, n):
            d = grafo.distancia(cidades[i], cidades[j])
            if d > max_d:
                max_d = d
                u_init, v_init = cidades[i], cidades[j]

    subtour = [u_init, v_init]
    nao_visitadas = set(cidades)
    nao_visitadas.discard(u_init)
    nao_visitadas.discard(v_init)

    while nao_visitadas:
        # Encontra a cidade k não visitada cuja distância mínima ao subtour é MÁXIMA
        best_k = None
        max_min_d = -1.0
        for k in nao_visitadas:
            min_d = min(grafo.distancia(k, node) for node in subtour)
            if min_d > max_min_d:
                max_min_d = min_d
                best_k = k

        # Encontra a melhor posição de inserção (menor acréscimo de custo)
        best_pos = 0
        min_cost_inc = float("inf")
        m = len(subtour)
        for i in range(m):
            u = subtour[i]
            v = subtour[(i + 1) % m]
            cost_inc = grafo.distancia(u, best_k) + grafo.distancia(best_k, v) - grafo.distancia(u, v)
            if cost_inc < min_cost_inc:
                min_cost_inc = cost_inc
                best_pos = i + 1

        subtour.insert(best_pos, best_k)
        nao_visitadas.remove(best_k)

    # Rotaciona para iniciar na cidade 1
    idx_1 = subtour.index(1)
    return subtour[idx_1:] + subtour[:idx_1]


def resolver_tsp_multistart_nn(grafo: Grafo) -> List[int]:
    """
    Resolve o TSP executando o vizinho mais próximo a partir de cada cidade
    e seleciona a rota de menor custo total, rotacionando para iniciar na cidade 1.
    """
    cidades = list(grafo.cidades.keys())
    n = len(cidades)
    melhor_rota = list(cidades)
    melhor_dist = float("inf")

    for start in cidades:
        nao_vis = set(cidades)
        nao_vis.remove(start)
        rota = [start]
        atual = start
        while nao_vis:
            prox = min(nao_vis, key=lambda c: grafo.distancia(atual, c))
            rota.append(prox)
            nao_vis.remove(prox)
            atual = prox

        dist_total = sum(grafo.distancia(rota[i], rota[(i + 1) % n]) for i in range(n))
        if dist_total < melhor_dist:
            melhor_dist = dist_total
            idx_1 = rota.index(1)
            melhor_rota = rota[idx_1:] + rota[:idx_1]

    return melhor_rota


def obter_melhor_rota_tsp(grafo: Grafo) -> List[int]:
    """
    Compara Farthest Insertion e Multi-start Nearest Neighbor, retornando
    o ciclo hamiltoniano de menor distância total com início na cidade 1.
    """
    rota_fi = resolver_tsp_farthest_insertion(grafo)
    rota_nn = resolver_tsp_multistart_nn(grafo)

    n = len(grafo.cidades)
    dist_fi = sum(grafo.distancia(rota_fi[i], rota_fi[(i + 1) % n]) for i in range(n))
    dist_nn = sum(grafo.distancia(rota_nn[i], rota_nn[(i + 1) % n]) for i in range(n))

    return rota_fi if dist_fi <= dist_nn else rota_nn


@SolverRegistry.register
class Heuristic37Solver(BaseSolver):
    """
    Solucionador baseado na teoria do Problema da Secretária (regra dos 37%):
    - Constrói uma rota TSP de alta qualidade (Farthest Insertion / Multi-start) iniciando em 1.
    - Avalia os itens na ordem sequencial da rota (ordem de chegada/descoberta).
    - Utiliza o custo marginal real de transporte (distância restante até a cidade 1).
    - Observa os primeiros 37% dos itens para calibrar o limiar de qualidade (benchmark).
    - A partir dos 37%, seleciona itens que superem esse benchmark enquanto couber na mochila.
    """

    @property
    def nome(self) -> str:
        return "heuristic_37"

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        # 1. Rota TSP de menor distância iniciando na cidade 1
        rota = obter_melhor_rota_tsp(instance.grafo)
        n = len(rota)

        # 2. Distâncias restantes de cada cidade até o fechamento do ciclo na cidade 1
        # O peso de um item pego na cidade rota[i] é transportado pelas pernas
        # rota[i] -> rota[i+1] -> ... -> rota[n-1] -> rota[0]
        distancias_restantes = {}
        dist_acc = instance.grafo.distancia(rota[-1], rota[0])
        distancias_restantes[rota[-1]] = dist_acc

        for i in range(n - 2, -1, -1):
            dist_acc += instance.grafo.distancia(rota[i], rota[i + 1])
            distancias_restantes[rota[i]] = dist_acc

        # 3. Cálculo do score marginal sem alpha arbitrário
        # Impacto físico marginal: tempo adicional gasto devido à redução de velocidade
        # provocada pelo peso do item transportado pela distância restante.
        v_avg = (instance.v_max + instance.v_min) / 2.0
        cap_mochila = max(instance.capacidade_mochila, 1e-6)

        def calcular_score(item: Item) -> float:
            if item.peso <= 0:
                return float("inf")
            d_restante = distancias_restantes.get(item.cidade_origem, 0.0)
            # Custo marginal estimado = aluguel * delta_tempo
            custo_transporte = (
                instance.taxa_aluguel
                * d_restante
                * ((instance.v_max - instance.v_min) / (cap_mochila * (v_avg ** 2)))
                * item.peso
            )
            lucro_liquido = item.valor - custo_transporte
            return lucro_liquido / item.peso

        # 4. Sequência da secretária:
        # Itens dispostos na ordem natural em que são visitados ao longo da rota
        # (NUNCA pré-ordenados por score, pois isso colocaria os melhores na calibração)
        itens_sequenciais: List[Item] = []
        for cidade_id in rota:
            cidade = instance.grafo.cidades[cidade_id]
            itens_sequenciais.extend(cidade.itens)

        m = len(itens_sequenciais)
        limite_calibracao = int(0.37 * m)
        itens_calibracao = itens_sequenciais[:limite_calibracao]
        itens_coleta = itens_sequenciais[limite_calibracao:]

        # Fase de calibração: observa os scores dos primeiros 37%
        scores_calibracao = [calcular_score(item) for item in itens_calibracao]
        melhor_threshold = max(scores_calibracao) if scores_calibracao else 0.0
        # O limiar nunca deve ser negativo, para não coletar itens que geram prejuízo
        limiar = max(melhor_threshold, 0.0)

        # Fase de decisão/coleta: coleta itens que superam o benchmark
        peso_atual = 0.0
        plano_coleta: Dict[int, List[int]] = {}

        for item in itens_coleta:
            score = calcular_score(item)
            if score >= limiar and score > 0:
                if peso_atual + item.peso <= instance.capacidade_mochila:
                    peso_atual += item.peso
                    if item.cidade_origem not in plano_coleta:
                        plano_coleta[item.cidade_origem] = []
                    plano_coleta[item.cidade_origem].append(item.id)

        # Fallback de segurança: se o limiar dos 37% foi muito alto e nenhum item foi pego,
        # aceita qualquer item dos 63% restantes com score positivo (> 0)
        if not plano_coleta:
            for item in itens_coleta:
                score = calcular_score(item)
                if score > 0 and peso_atual + item.peso <= instance.capacidade_mochila:
                    peso_atual += item.peso
                    if item.cidade_origem not in plano_coleta:
                        plano_coleta[item.cidade_origem] = []
                    plano_coleta[item.cidade_origem].append(item.id)

        return SolucaoTTP(rota=rota, plano_coleta=plano_coleta)
