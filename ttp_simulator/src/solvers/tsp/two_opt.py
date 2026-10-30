from typing import List
import numpy as np
from src.core.models import Grafo
from src.solvers.numba_kernels import (
    NUMBA_AVAILABLE,
    numba_otimizar_2opt,
    numba_caminho_fixo_bb,
    numba_caminho_fixo_2opt,
)


def _pode_usar_matriz_densa(grafo: Grafo) -> bool:
    if not NUMBA_AVAILABLE or not grafo.cidades:
        return False
    max_id = max(grafo.cidades.keys())
    return max_id == len(grafo.cidades) and 1 in grafo.cidades


def otimizar_2opt(rota: List[int], grafo: Grafo) -> List[int]:
    """
    Refinamento local 2-opt para um ciclo hamiltoniano iniciando na cidade 1.
    Desfaz cruzamentos invertendo subtrechos.
    """
    n = len(rota)
    if n <= 3:
        return list(rota)

    if _pode_usar_matriz_densa(grafo):
        D = grafo.distance_matrix
        rota_arr = np.asarray(rota, dtype=np.int32)
        res_arr = numba_otimizar_2opt(rota_arr, D)
        return res_arr.tolist()

    melhor_rota = list(rota)
    melhorou = True
    while melhorou:
        melhorou = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                u1, v1 = melhor_rota[i - 1], melhor_rota[i]
                u2, v2 = melhor_rota[j], melhor_rota[(j + 1) % n]
                d_atual = grafo.distancia(u1, v1) + grafo.distancia(u2, v2)
                d_nova = grafo.distancia(u1, u2) + grafo.distancia(v1, v2)
                if d_nova < d_atual - 1e-6:
                    melhor_rota[i : j + 1] = reversed(melhor_rota[i : j + 1])
                    melhorou = True
                    break
            if melhorou:
                break
    return melhor_rota


def resolver_caminho_fixo_bb(A: int, B: int, cidades: List[int], grafo: Grafo) -> List[int]:
    """
    Resolve o Problema do Caminho Hamiltoniano com extremidades fixas A e B
    para um conjunto pequeno de cidades usando Branch and Bound exato com poda de limite inferior.
    """
    n = len(cidades)
    if n <= 1:
        return list(cidades)

    if _pode_usar_matriz_densa(grafo):
        D = grafo.distance_matrix
        cid_arr = np.asarray(cidades, dtype=np.int32)
        res_arr = numba_caminho_fixo_bb(int(A), int(B), cid_arr, D)
        return res_arr.tolist()

    d_orig = (
        grafo.distancia(A, cidades[0])
        + sum(grafo.distancia(cidades[i], cidades[i + 1]) for i in range(n - 1))
        + grafo.distancia(cidades[-1], B)
    )
    melhor_dist = [d_orig]
    melhor_ordem = list(cidades)

    nodes = [A] + list(cidades) + [B]
    dist_matrix = [[grafo.distancia(u, v) for v in nodes] for u in nodes]

    # Menor aresta de saída válida para cada nó intermediário (índices 1..n)
    min_out = [0.0] * (n + 1)
    for i in range(1, n + 1):
        min_out[i] = min(dist_matrix[i][j] for j in range(1, n + 2) if j != i)

    def bb(curr_idx: int, unvisited_mask: int, dist_acc: float, path: List[int]):
        if unvisited_mask == 0:
            total = dist_acc + dist_matrix[curr_idx][n + 1]
            if total < melhor_dist[0] - 1e-6:
                melhor_dist[0] = total
                melhor_ordem[:] = path[:]
            return

        # Limite inferior (Lower Bound) para poda
        lb = dist_acc + min(
            dist_matrix[curr_idx][j] for j in range(1, n + 1) if (unvisited_mask & (1 << (j - 1)))
        )
        for j in range(1, n + 1):
            if unvisited_mask & (1 << (j - 1)):
                lb += min_out[j]
        if lb >= melhor_dist[0] - 1e-6:
            return

        # Exploração dos ramos
        for j in range(1, n + 1):
            bit = 1 << (j - 1)
            if unvisited_mask & bit:
                d = dist_acc + dist_matrix[curr_idx][j]
                if d >= melhor_dist[0] - 1e-6:
                    continue
                path.append(nodes[j])
                bb(j, unvisited_mask ^ bit, d, path)
                path.pop()

    bb(0, (1 << n) - 1, 0.0, [])
    return melhor_ordem


def resolver_caminho_fixo_2opt(A: int, B: int, cidades: List[int], grafo: Grafo) -> List[int]:
    """
    Otimiza a ordem de cidades intermediárias com extremidades fixas A e B
    usando busca local 2-opt e relocação 1-opt.
    """
    m = len(cidades)
    if m <= 1:
        return list(cidades)

    if _pode_usar_matriz_densa(grafo):
        D = grafo.distance_matrix
        cid_arr = np.asarray(cidades, dtype=np.int32)
        res_arr = numba_caminho_fixo_2opt(int(A), int(B), cid_arr, D)
        return res_arr.tolist()

    melhor = list(cidades)
    melhorou = True
    while melhorou:
        melhorou = False
        # 1. Movimentos 2-opt (inversão de subtrechos)
        for i in range(m):
            for j in range(i + 1, m):
                u1 = A if i == 0 else melhor[i - 1]
                v1 = melhor[i]
                u2 = melhor[j]
                v2 = B if j == m - 1 else melhor[j + 1]

                d_atual = grafo.distancia(u1, v1) + grafo.distancia(u2, v2)
                d_nova = grafo.distancia(u1, u2) + grafo.distancia(v1, v2)

                if d_nova < d_atual - 1e-6:
                    melhor[i : j + 1] = reversed(melhor[i : j + 1])
                    melhorou = True
                    break
            if melhorou:
                break

        if melhorou:
            continue

        # 2. Movimentos de Relocação 1-opt (Or-Opt de nó único)
        for i in range(m):
            cid = melhor[i]
            pred = A if i == 0 else melhor[i - 1]
            succ = B if i == m - 1 else melhor[i + 1]
            custo_remover = grafo.distancia(pred, succ) - (grafo.distancia(pred, cid) + grafo.distancia(cid, succ))

            sub = melhor[:i] + melhor[i + 1:]
            melhor_insercao = None
            maior_economia = custo_remover

            for pos in range(len(sub) + 1):
                p_pred = A if pos == 0 else sub[pos - 1]
                p_succ = B if pos == len(sub) else sub[pos]
                custo_inserir = grafo.distancia(p_pred, cid) + grafo.distancia(cid, p_succ) - grafo.distancia(p_pred, p_succ)
                delta_total = custo_remover + custo_inserir
                if delta_total < maior_economia - 1e-6:
                    maior_economia = delta_total
                    melhor_insercao = pos

            if melhor_insercao is not None:
                sub.insert(melhor_insercao, cid)
                melhor = sub
                melhorou = True
                break

    return melhor
