"""
Módulo de Pré-processamento 1-Tree e Alpha-Nearness para o ReinforceNS.

Referências:
- Wu et al. (2024), "ReinforceNS: Reinforcement Learning-based Multi-start
  Neighborhood Search for Solving the Traveling Thief Problem", IJCAI-24,
  Seção 4.1 eSeção 4.3 (Equação 6).
- Helsgaun (2000), "An effective implementation of the Lin-Kernighan
  traveling salesman heuristic", EJOR 126(1):106-130.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
from src.core.models import Grafo


@dataclass
class OneTreeResult:
    """Resultado do pré-processamento da 1-tree mínima e inicialização de Q."""

    l_t: float
    alpha: np.ndarray
    q_table: np.ndarray
    candidate_neighbors: Dict[int, List[int]]
    tree_edges: Set[Tuple[int, int]]


def _prim_mst_and_beta(
    cost_mtx: np.ndarray,
    nodes: List[int],
) -> Tuple[float, Set[Tuple[int, int]], np.ndarray, Dict[int, int]]:
    """
    Executa o algoritmo de Prim em O(n^2) sobre o subconjunto de vértices `nodes`
    e calcula simultaneamente a matriz beta(u, v), onde beta(u, v) é o peso da
    maior aresta no único caminho entre u e v dentro da MST.
    """
    dim = cost_mtx.shape[0]
    beta = np.zeros((dim, dim), dtype=float)
    degrees: Dict[int, int] = {u: 0 for u in nodes}
    mst_edges: Set[Tuple[int, int]] = set()

    if len(nodes) <= 1:
        return 0.0, mst_edges, beta, degrees

    root = nodes[0]
    in_tree = {root}
    tree_nodes = [root]

    min_cost = {}
    parent = {}
    for v in nodes[1:]:
        min_cost[v] = float(cost_mtx[root, v])
        parent[v] = root

    mst_weight = 0.0

    while len(in_tree) < len(nodes):
        # Escolhe o vértice fora da árvore mais próximo da árvore atual
        best_u = None
        best_w = np.inf
        for v in nodes:
            if v not in in_tree and min_cost[v] < best_w:
                best_w = min_cost[v]
                best_u = v

        if best_u is None:
            break

        p_u = parent[best_u]
        edge = (min(p_u, best_u), max(p_u, best_u))
        mst_edges.add(edge)
        mst_weight += best_w
        degrees[p_u] += 1
        degrees[best_u] += 1

        # Recorrência de Helsgaun (2000) para beta(w, best_u) em O(|tree_nodes|)
        for w in tree_nodes:
            if w == p_u:
                b_val = best_w
            else:
                b_val = max(beta[w, p_u], best_w)
            beta[w, best_u] = b_val
            beta[best_u, w] = b_val

        in_tree.add(best_u)
        tree_nodes.append(best_u)

        # Atualiza distâncias mínimas para os vértices restantes
        for v in nodes:
            if v not in in_tree:
                c_uv = float(cost_mtx[best_u, v])
                if c_uv < min_cost[v]:
                    min_cost[v] = c_uv
                    parent[v] = best_u

    return mst_weight, mst_edges, beta, degrees


def compute_minimum_one_tree(
    dist_mtx: np.ndarray,
    special_node: int = 1,
    pi: np.ndarray | None = None,
) -> Tuple[float, np.ndarray, Set[Tuple[int, int]], Dict[int, int]]:
    """
    Calcula a 1-tree mínima T com vértice especial `special_node` e a matriz
    de alpha-nearness alpha(i, j) = L(T^+(i, j)) - L(T) >= 0.

    Se o vetor de penalidades de vértices `pi` for fornecido, opera sobre
    a matriz de custos transformada c_pi(i, j) = d(i, j) + pi[i] + pi[j].
    """
    dim = dist_mtx.shape[0]
    all_nodes = [i for i in range(1, dim)]
    if special_node not in all_nodes:
        special_node = all_nodes[0]

    if pi is not None:
        cost_mtx = dist_mtx.copy()
        for i in all_nodes:
            for j in all_nodes:
                if i != j:
                    cost_mtx[i, j] = dist_mtx[i, j] + pi[i] + pi[j]
    else:
        cost_mtx = dist_mtx

    other_nodes = [u for u in all_nodes if u != special_node]
    mst_weight, mst_edges, beta, degrees = _prim_mst_and_beta(
        cost_mtx, other_nodes
    )
    degrees[special_node] = 0

    alpha = np.zeros((dim, dim), dtype=float)
    tree_edges = set(mst_edges)

    if len(other_nodes) == 0:
        return 0.0, alpha, tree_edges, degrees

    if len(other_nodes) == 1:
        u1 = other_nodes[0]
        e1 = (min(special_node, u1), max(special_node, u1))
        tree_edges.add(e1)
        degrees[special_node] += 1
        degrees[u1] += 1
        l_t = float(2.0 * cost_mtx[special_node, u1])
        if pi is not None:
            l_t -= float(2.0 * np.sum(pi[1:]))
        return l_t, alpha, tree_edges, degrees

    # Encontra as duas menores arestas incidentes ao vértice especial
    sorted_incident = sorted(
        other_nodes, key=lambda u: float(cost_mtx[special_node, u])
    )
    u1, u2 = sorted_incident[0], sorted_incident[1]
    c_u1 = float(cost_mtx[special_node, u1])
    c_u2 = float(cost_mtx[special_node, u2])

    tree_edges.add((min(special_node, u1), max(special_node, u1)))
    tree_edges.add((min(special_node, u2), max(special_node, u2)))
    degrees[special_node] += 2
    degrees[u1] += 1
    degrees[u2] += 1

    l_t_penalized = mst_weight + c_u1 + c_u2
    if pi is not None:
        l_t = float(l_t_penalized - 2.0 * np.sum(pi[1:]))
    else:
        l_t = float(l_t_penalized)

    # Calcula alpha(i, j) = L(T^+(i, j)) - L(T) >= 0
    # 1. Arestas incidentes a special_node: substituem a segunda menor aresta (special_node, u2)
    for u in other_nodes:
        if u in (u1, u2):
            val = 0.0
        else:
            val = max(0.0, float(cost_mtx[special_node, u]) - c_u2)
        alpha[special_node, u] = val
        alpha[u, special_node] = val

    # 2. Arestas entre vértices em V \ {special_node}: substituem a maior aresta beta(u, v) no caminho da MST
    for idx_a in range(len(other_nodes)):
        u = other_nodes[idx_a]
        for idx_b in range(idx_a + 1, len(other_nodes)):
            v = other_nodes[idx_b]
            val = max(0.0, float(cost_mtx[u, v] - beta[u, v]))
            alpha[u, v] = val
            alpha[v, u] = val

    return l_t, alpha, tree_edges, degrees


def compute_q_matrix(
    l_t: float,
    alpha: np.ndarray,
    dist_mtx: np.ndarray,
    tau: float = 0.5,
    b: float = 1e-6,
) -> np.ndarray:
    """
    Inicializa a matriz Q(i, j) conforme a Equação (6) de Wu et al. (2024)
    (Equação 2 do artigo de referência):

        Q(i, j) = L(T) / (tau * alpha(i, j) + (1 - tau) * d(i, j) + b)
    """
    dim = dist_mtx.shape[0]
    q_table = np.zeros((dim, dim), dtype=float)

    for i in range(1, dim):
        for j in range(1, dim):
            if i != j and np.isfinite(dist_mtx[i, j]):
                denom = tau * alpha[i, j] + (1.0 - tau) * dist_mtx[i, j] + b
                q_table[i, j] = l_t / max(denom, b)

    return q_table


def build_one_tree_preprocessing(
    grafo: Grafo,
    num_candidates: int = 15,
    tau: float = 0.5,
    b: float = 1e-6,
    subgradient_steps: int = 0,
    neighbor_ranking: str = "distance",
) -> OneTreeResult:
    """
    Executa o pré-processamento completo do ReinforceNS:
    1. Calcula a 1-tree mínima (com otimização por subgradiente opcional).
    2. Calcula a matriz alpha(i, j) e o comprimento L(T).
    3. Inicializa a matriz Q(i, j) via Equação (6).
    4. Constrói os conjuntos de vizinhos candidatos N(x_i) de tamanho até `num_candidates`.

    Parâmetros:
        grafo: Grafo da instância TTP.
        num_candidates: Tamanho máximo de N(x_i) (default: 15).
        tau: Peso de alpha(i, j) na Equação (6) (default: 0.5).
        b: Constante de estabilidade numérica na Equação (6) (default: 1e-6).
        subgradient_steps: Número de passos de subgradiente para ajustar graus rumo a 2.
        neighbor_ranking: Critério para ordenar N(x_i):
            - "distance": os `num_candidates` vizinhos mais próximos por d(i, j)
              (conforme Seção 2.1 do artigo: "15 vizinhos mais próximos").
            - "alpha": ordena por menor alpha(i, j) e desempata por menor d(i, j).
    """
    dist_mtx = grafo.distance_matrix
    dim = dist_mtx.shape[0]
    n_cities = dim - 1
    k = min(num_candidates, max(0, n_cities - 1))

    best_l_t, best_alpha, best_edges, degrees = compute_minimum_one_tree(
        dist_mtx, special_node=1, pi=None
    )

    if subgradient_steps > 0 and n_cities >= 3:
        pi = np.zeros(dim, dtype=float)
        prev_v = np.zeros(dim, dtype=float)
        step_size = max(best_l_t / (2.0 * n_cities), 1.0)

        for step in range(subgradient_steps):
            v = np.zeros(dim, dtype=float)
            for u in range(1, dim):
                v[u] = float(degrees.get(u, 2) - 2)

            norm_sq = float(np.sum(v[1:] ** 2))
            if norm_sq == 0.0:
                break

            direction = v if step == 0 else (0.7 * v + 0.3 * prev_v)
            pi[1:] += step_size * direction[1:]
            pi[1:] -= np.mean(pi[1:])  # Mantém soma de pi próxima de zero
            prev_v = v

            cand_l_t, cand_alpha, cand_edges, degrees = (
                compute_minimum_one_tree(dist_mtx, special_node=1, pi=pi)
            )
            if cand_l_t > best_l_t:
                best_l_t = cand_l_t
                best_alpha = cand_alpha
                best_edges = cand_edges
                step_size *= 1.05
            else:
                step_size *= 0.7

    q_table = compute_q_matrix(
        l_t=best_l_t,
        alpha=best_alpha,
        dist_mtx=dist_mtx,
        tau=tau,
        b=b,
    )

    candidate_neighbors: Dict[int, List[int]] = {}
    for i in range(1, dim):
        others = [j for j in range(1, dim) if j != i]
        if neighbor_ranking == "alpha":
            others.sort(key=lambda j: (best_alpha[i, j], dist_mtx[i, j]))
        else:
            others.sort(key=lambda j: (dist_mtx[i, j], best_alpha[i, j]))
        candidate_neighbors[i] = others[:k]

    return OneTreeResult(
        l_t=best_l_t,
        alpha=best_alpha,
        q_table=q_table,
        candidate_neighbors=candidate_neighbors,
        tree_edges=best_edges,
    )
