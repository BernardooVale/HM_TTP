from typing import List
from src.core.models import Grafo


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
