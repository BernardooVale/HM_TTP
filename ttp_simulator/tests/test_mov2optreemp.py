import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.core.evaluator_ttp1 import evaluate as evaluate_ttp1
from src.core.evaluator_ttp2 import evaluate as evaluate_ttp2
from src.core.models import Tour
from src.scenarios.parser import InstanceParser
from src.solvers.iterative.mov2optreemp import Mov2OptReempOperator
from src.solvers.packing.greedy import IGP
from src.solvers.tsp.CLK import ChainedLK


@pytest.fixture
def instance_11():
    inst_path = Path(__file__).parent.parent / "bateria_testes" / "11.ttp"
    if not inst_path.exists():
        pytest.skip("Instância 11.ttp não encontrada.")
    return InstanceParser.carregar(str(inst_path))


def test_mov2optreemp_isolates_outside_subroute_and_matches_oracle(instance_11):
    np.random.seed(42)
    clk_tour = ChainedLK(
        grafo=instance_11.grafo, num_tries=5, num_kicks=1
    ).solve()
    igp = IGP(scenario=instance_11, tour=clk_tour, modelo="TTP1")
    init_plan = igp.solve(theta=5.0, num_iterations=10)

    op = Mov2OptReempOperator(
        instance=instance_11,
        k=igp.best_k,
        beta=igp.best_beta,
        gamma=igp.best_gamma,
        delta=igp.best_delta,
        modelo="TTP1",
    )

    state0 = op.build_initial_state(clk_tour.order, init_plan)
    res0 = evaluate_ttp1(instance_11, state0.to_solucao_ttp())
    assert state0.objective == pytest.approx(res0["lucro_liquido_G"], abs=1e-6)

    # Aplica um movimento entre as posições 15 e 45 da rota
    city_a = state0.route[15]
    city_b = state0.route[45]
    sub_set = set(state0.route[15:46])
    outside_cities = [c for c in state0.route if c not in sub_set]

    state1 = op.apply_move(state0, city_a, city_b, in_place=False)

    # 1. Cidades fora da sub-rota devem manter exatamente os mesmos itens na mochila
    for c in outside_cities:
        assert state1.plan[c] == state0.plan[c]

    # 2. A sub-rota entre 15 e 45 deve ter sido invertida exatamente
    assert state1.route[15:46] == state0.route[15:46][::-1]
    assert state1.route[:15] == state0.route[:15]
    assert state1.route[46:] == state0.route[46:]
    assert state1.route[0] == 1

    # 3. Todas as distâncias até o depósito (end_dist / ξ) devem bater com Tour.end_distance
    ref_tour = Tour(order=state1.route, cost=0.0)
    for c in state1.route:
        assert state1.end_dist[c] == pytest.approx(
            ref_tour.end_distance(c, instance_11.grafo), rel=1e-9
        )

    # 4. Peso total deve respeitar delta * W
    assert (
        state1.total_weight
        <= igp.best_delta * instance_11.capacidade_mochila + 1e-6
    )

    # 5. Os itens escolhidos na sub-rota devem coincidir com a ordenação manual da Equação (5)
    sub_items = [
        it for it in instance_11.itens if it.cidade_origem in sub_set
    ]
    sub_items_sorted = sorted(
        sub_items,
        key=lambda it: (it.valor ** igp.best_k)
        / (
            (it.peso ** igp.best_beta)
            * (ref_tour.end_distance(it.cidade_origem, instance_11.grafo) ** igp.best_gamma)
        ),
        reverse=True,
    )
    weight_outside = sum(
        state0.city_picked_weight[c] for c in outside_cities
    )
    expected_sub_ids = set()
    w_curr = weight_outside
    for it in sub_items_sorted:
        if w_curr + it.peso <= igp.best_delta * instance_11.capacidade_mochila + 1e-9:
            expected_sub_ids.add(it.id)
            w_curr += it.peso

    actual_sub_ids = {
        it_id for c in sub_set for it_id in state1.plan[c]
    }
    assert actual_sub_ids == expected_sub_ids


def test_mov2optreemp_50_random_moves_match_official_evaluators(instance_11):
    rng = np.random.default_rng(123)
    np.random.seed(123)
    clk_tour = ChainedLK(
        grafo=instance_11.grafo, num_tries=5, num_kicks=1
    ).solve()

    for modelo, official_eval in [("TTP1", evaluate_ttp1), ("TTP2", evaluate_ttp2)]:
        igp = IGP(scenario=instance_11, tour=clk_tour, modelo=modelo)
        init_plan = igp.solve(theta=5.0, num_iterations=5)

        op = Mov2OptReempOperator(
            instance=instance_11,
            k=igp.best_k,
            beta=igp.best_beta,
            gamma=igp.best_gamma,
            delta=igp.best_delta,
            modelo=modelo,
        )
        state = op.build_initial_state(clk_tour.order, init_plan)

        non_depot_cities = [c for c in state.route if c != 1]
        for _ in range(25):
            c_a, c_b = rng.choice(non_depot_cities, size=2, replace=False)
            state = op.apply_move(state, int(c_a), int(c_b), in_place=True)

            res = official_eval(instance_11, state.to_solucao_ttp())
            assert state.objective == pytest.approx(
                res["lucro_liquido_G"], abs=1e-5
            )
            assert state.travel_time == pytest.approx(
                res["tempo_viagem_f"], abs=1e-6
            )
            assert state.total_weight == pytest.approx(
                res["peso_acumulado"], abs=1e-6
            )
