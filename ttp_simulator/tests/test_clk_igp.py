import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.core.evaluator_ttp1 import evaluate as evaluate_ttp1
from src.core.models import Cidade, Grafo, SolucaoTTP, Tour
from src.scenarios.parser import InstanceParser
from src.solvers.packing.greedy import IGP
from src.solvers.tsp.cheapest_insertion import CheapInsertionRegret
from src.solvers.tsp.CLK import ChainedLK, DoubleBridge, LKLocalSearch


def _cycle_cost(order: list[int], grafo: Grafo) -> float:
    n = len(order)
    return float(
        sum(
            grafo.distance_matrix[order[i], order[(i + 1) % n]]
            for i in range(n)
        )
    )


@pytest.fixture
def grafo_12_cidades():
    rng = np.random.default_rng(42)
    cidades = {
        i: Cidade(
            id=i,
            x=float(rng.uniform(0, 100)),
            y=float(rng.uniform(0, 100)),
        )
        for i in range(1, 13)
    }
    return Grafo(cidades=cidades, edge_weight_type="EUC_2D")


def test_closest_nodes_distance_zero_last(grafo_12_cidades):
    closest = grafo_12_cidades.pre_calculate_closest_nodes(num_closest=11)
    for city_id in range(1, 13):
        neighbors = closest[city_id]
        # Como num_closest=11 e existem 11 cidades reais vizinhas (dist < inf),
        # o índice 0 (dist = inf) fica no fim e não entra nos 11 mais próximos
        assert 0 not in neighbors
        assert city_id not in neighbors
        assert set(neighbors) == set(range(1, 13)) - {city_id}


def test_cheapest_insertion_regret_valid_tour(grafo_12_cidades):
    ci = CheapInsertionRegret(grafo=grafo_12_cidades)
    tour = ci.find_regret_insertion()

    assert len(tour.order) == 12
    assert set(tour.order) == set(range(1, 13))
    assert 0 not in tour.order
    expected_cost = _cycle_cost(tour.order, grafo_12_cidades)
    assert tour.cost == pytest.approx(expected_cost, rel=1e-6)


def test_end_distance_cache_matches_manual(grafo_12_cidades):
    order = list(range(1, 13))
    cost = _cycle_cost(order, grafo_12_cidades)
    tour = Tour(order=order, cost=cost)

    for idx, node in enumerate(tour.order):
        manual_dist = grafo_12_cidades.distance_matrix[
            tour.order[-1], tour.order[0]
        ]
        for j in range(idx, len(tour.order) - 1):
            manual_dist += grafo_12_cidades.distance_matrix[
                tour.order[j], tour.order[j + 1]
            ]
        assert tour.end_distance(node, grafo_12_cidades) == pytest.approx(
            manual_dist, rel=1e-9
        )

    # Verifica invalidação do cache após flip e normalize_to_city_1
    tour.flip(3, 8)
    for idx, node in enumerate(tour.order):
        manual_dist = grafo_12_cidades.distance_matrix[
            tour.order[-1], tour.order[0]
        ]
        for j in range(idx, len(tour.order) - 1):
            manual_dist += grafo_12_cidades.distance_matrix[
                tour.order[j], tour.order[j + 1]
            ]
        assert tour.end_distance(node, grafo_12_cidades) == pytest.approx(
            manual_dist, rel=1e-9
        )


def test_clk_monotonicity_and_invariants(grafo_12_cidades):
    np.random.seed(123)
    ci = CheapInsertionRegret(grafo=grafo_12_cidades)
    init_tour = ci.find_regret_insertion()

    lk = LKLocalSearch(grafo=grafo_12_cidades, num_tries=5)
    lk_tour, lk_cost = lk.solve(init_tour)

    clk = ChainedLK(grafo=grafo_12_cidades, num_tries=5, num_kicks=3)
    clk_tour = clk.solve(initial_tour=init_tour)

    # Monotonicidade de melhoria
    assert lk_cost <= init_tour.cost + 1e-9
    assert clk_tour.cost <= lk_cost + 1e-9

    # Invariantes de permutação e normalização
    assert clk_tour.order[0] == 1
    assert len(clk_tour.order) == 12
    assert set(clk_tour.order) == set(range(1, 13))
    assert clk_tour.cost == pytest.approx(
        _cycle_cost(clk_tour.order, grafo_12_cidades), rel=1e-6
    )
    for idx, city in enumerate(clk_tour.order):
        assert clk_tour.pos[city] == idx


def test_clk_and_igp_on_real_instance():
    inst_path = (
        Path(__file__).parent.parent / "bateria_testes" / "11.ttp"
    )
    if not inst_path.exists():
        pytest.skip("Instância 11.ttp não encontrada.")

    np.random.seed(42)
    instance = InstanceParser.carregar(str(inst_path))

    ci_tour = CheapInsertionRegret(grafo=instance.grafo).find_regret_insertion()
    clk = ChainedLK(grafo=instance.grafo, num_tries=5, num_kicks=2)
    tour = clk.solve(initial_tour=ci_tour)

    assert tour.order[0] == 1
    assert len(tour.order) == instance.num_cities
    assert set(tour.order) == set(range(1, instance.num_cities + 1))
    assert tour.cost <= ci_tour.cost + 1e-9

    igp = IGP(scenario=instance, tour=tour, modelo="TTP1")
    pack_dict = igp.solve(theta=5.0, num_iterations=20)

    # Verifica que hiperparâmetros foram armazenados para o RLNS
    assert hasattr(igp, "best_k") and igp.best_k is not None
    assert hasattr(igp, "best_beta") and igp.best_beta is not None
    assert hasattr(igp, "best_gamma") and igp.best_gamma is not None
    assert hasattr(igp, "best_delta") and 0.5 <= igp.best_delta <= 1.0

    # Verifica respeito à capacidade restrita (delta * W) e equivalência com evaluate_ttp1
    total_weight = sum(item.peso for item in igp.best_packed_items)
    assert total_weight <= igp.best_delta * instance.capacidade_mochila + 1e-6

    solucao = SolucaoTTP(rota=tour.order, plano_coleta=pack_dict)
    res_oficial = evaluate_ttp1(instance, solucao)
    assert igp.best_score == pytest.approx(res_oficial["lucro_liquido_G"], abs=1e-5)
