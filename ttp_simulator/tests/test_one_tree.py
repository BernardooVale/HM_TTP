import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.core.models import Cidade, Grafo
from src.scenarios.parser import InstanceParser
from src.solvers.tsp.CLK import ChainedLK
from src.solvers.tsp.one_tree import (
    build_one_tree_preprocessing,
    compute_minimum_one_tree,
)


def _is_spanning_tree(nodes: list[int], edges: list[tuple[int, int]]) -> bool:
    if len(edges) != len(nodes) - 1:
        return False
    adj = {u: [] for u in nodes}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    visited = {nodes[0]}
    stack = [nodes[0]]
    while stack:
        curr = stack.pop()
        for nxt in adj[curr]:
            if nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)
    return len(visited) == len(nodes)


def _brute_force_min_one_tree_with_edge(
    dist_mtx: np.ndarray,
    n: int,
    forced_edge: tuple[int, int] | None = None,
) -> float:
    """
    Oráculo de força bruta que enumera todas as 1-trees válidas (com vértice
    especial 1) e retorna o menor comprimento que inclui `forced_edge`.
    """
    other_nodes = list(range(2, n + 1))
    all_other_edges = list(itertools.combinations(other_nodes, 2))
    incident_1 = [(1, u) for u in other_nodes]

    if forced_edge is not None:
        fe = (min(forced_edge), max(forced_edge))
    else:
        fe = None

    best_len = math.inf
    for tree_cand in itertools.combinations(
        all_other_edges, len(other_nodes) - 1
    ):
        if not _is_spanning_tree(other_nodes, list(tree_cand)):
            continue
        tree_cost = sum(dist_mtx[u, v] for u, v in tree_cand)
        for inc_cand in itertools.combinations(incident_1, 2):
            one_tree_edges = set(tree_cand) | set(inc_cand)
            if fe is not None and fe not in one_tree_edges:
                continue
            inc_cost = sum(dist_mtx[u, v] for u, v in inc_cand)
            best_len = min(best_len, float(tree_cost + inc_cost))

    return best_len


def test_one_tree_analytical_rectangle():
    """
    Testa um retângulo 10 x 20 onde L(T) e alpha(i, j) são conhecidos analiticamente.
    """
    cidades = {
        1: Cidade(id=1, x=0.0, y=0.0),
        2: Cidade(id=2, x=10.0, y=0.0),
        3: Cidade(id=3, x=10.0, y=20.0),
        4: Cidade(id=4, x=0.0, y=20.0),
    }
    grafo = Grafo(cidades=cidades, edge_weight_type="EUC_2D")
    res = build_one_tree_preprocessing(grafo, num_candidates=15, tau=0.5, b=1e-6)

    # MST em {2, 3, 4}: (3,4)=10 + (2,3)=20 = 30
    # Duas menores incidentes a 1: (1,2)=10 + (1,4)=20 = 30
    # L(T) = 60.0
    assert res.l_t == pytest.approx(60.0, rel=1e-9)
    assert len(res.tree_edges) == 4

    # Arestas da 1-tree devem ter alpha == 0
    for u, v in [(1, 2), (2, 3), (3, 4), (1, 4)]:
        assert res.alpha[u, v] == pytest.approx(0.0, abs=1e-9)
        assert res.alpha[v, u] == pytest.approx(0.0, abs=1e-9)

    # Diagonais (1,3) e (2,4) têm comprimento sqrt(500) e substituem aresta de peso 20
    diag = math.sqrt(500.0)
    expected_alpha_diag = diag - 20.0
    assert res.alpha[1, 3] == pytest.approx(expected_alpha_diag, rel=1e-9)
    assert res.alpha[2, 4] == pytest.approx(expected_alpha_diag, rel=1e-9)

    # Verifica Equação (6) para Q(1, 2) e Q(1, 3)
    expected_q_12 = 60.0 / (0.5 * 0.0 + 0.5 * 10.0 + 1e-6)
    expected_q_13 = 60.0 / (0.5 * expected_alpha_diag + 0.5 * diag + 1e-6)
    assert res.q_table[1, 2] == pytest.approx(expected_q_12, rel=1e-9)
    assert res.q_table[1, 3] == pytest.approx(expected_q_13, rel=1e-9)


def test_alpha_nearness_matches_brute_force_oracle():
    """
    Compara L(T) e alpha(i, j) calculados em O(n^2) contra o oráculo exato de
    força bruta para todos os pares (i, j) em um grafo aleatório de 6 vértices.
    """
    rng = np.random.default_rng(2024)
    n = 6
    cidades = {
        i: Cidade(
            id=i,
            x=float(rng.uniform(0, 100)),
            y=float(rng.uniform(0, 100)),
        )
        for i in range(1, n + 1)
    }
    grafo = Grafo(cidades=cidades, edge_weight_type="EUC_2D")
    l_t, alpha, tree_edges, _ = compute_minimum_one_tree(
        grafo.distance_matrix, special_node=1
    )

    oracle_l_t = _brute_force_min_one_tree_with_edge(
        grafo.distance_matrix, n, forced_edge=None
    )
    assert l_t == pytest.approx(oracle_l_t, rel=1e-9)
    assert len(tree_edges) == n

    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            oracle_forced = _brute_force_min_one_tree_with_edge(
                grafo.distance_matrix, n, forced_edge=(i, j)
            )
            expected_alpha = oracle_forced - oracle_l_t
            assert alpha[i, j] == pytest.approx(expected_alpha, abs=1e-7)
            assert alpha[j, i] == pytest.approx(expected_alpha, abs=1e-7)


def test_one_tree_lower_bound_and_candidates_on_real_instance():
    inst_path = Path(__file__).parent.parent / "bateria_testes" / "11.ttp"
    if not inst_path.exists():
        pytest.skip("Instância 11.ttp não encontrada.")

    np.random.seed(42)
    instance = InstanceParser.carregar(str(inst_path))
    n = instance.num_cities

    res_base = build_one_tree_preprocessing(
        instance.grafo, num_candidates=15, subgradient_steps=0
    )
    res_subgrad = build_one_tree_preprocessing(
        instance.grafo,
        num_candidates=15,
        subgradient_steps=15,
        neighbor_ranking="alpha",
    )

    clk_tour = ChainedLK(
        grafo=instance.grafo, num_tries=5, num_kicks=2
    ).solve()

    # Invariante Held-Karp: L(T_0) <= L(T_subgrad) <= Custo de qualquer tour TSP válido
    assert res_base.l_t <= res_subgrad.l_t + 1e-6
    assert res_subgrad.l_t <= clk_tour.cost + 1e-6

    # Verifica vizinhança candidata N(x_i) de tamanho 15 e tabela Q
    for i in range(1, n + 1):
        cand = res_base.candidate_neighbors[i]
        assert len(cand) == 15
        assert 0 not in cand
        assert i not in cand
        assert len(set(cand)) == 15
        for j in cand:
            assert res_base.q_table[i, j] > 0.0
            assert np.isfinite(res_base.q_table[i, j])
