import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.core.evaluator_ttp1 import evaluate as evaluate_ttp1
from src.core.models import Cidade, Grafo, Item, TTPInstance
from src.scenarios.parser import InstanceParser
from src.solvers.iterative.mov2optreemp import Mov2OptReempOperator
from src.solvers.iterative.rlns_local_search import (
    run_first_improvement_search,
    run_rlns_qlearning_search,
)
from src.solvers.packing.greedy import IGP
from src.solvers.tsp.CLK import ChainedLK
from src.solvers.tsp.one_tree import build_one_tree_preprocessing


@pytest.fixture
def synthetic_ttp_instance():
    """
    Instância sintética de 8 cidades com uma rota inicial propositalmente
    cruzada/subótima para garantir que tanto o RLNS quanto o First Improvement
    encontrem melhorias reais.
    """
    # Cidades dispostas em um círculo de raio 50
    cidades = {}
    for i in range(1, 9):
        angle = 2.0 * np.pi * (i - 1) / 8.0
        cidades[i] = Cidade(
            id=i,
            x=float(50.0 * np.cos(angle)),
            y=float(50.0 * np.sin(angle)),
        )
    itens = [
        Item(id=idx, peso=10.0 + idx, valor=100.0 * idx, cidade_origem=idx + 1)
        for idx in range(1, 8)
    ]
    for it in itens:
        cidades[it.cidade_origem].itens.append(it)

    grafo = Grafo(cidades=cidades, edge_weight_type="EUC_2D")
    return TTPInstance(
        nome="synthetic_8",
        grafo=grafo,
        itens=itens,
        capacidade_mochila=60.0,
        v_min=0.1,
        v_max=1.0,
        taxa_aluguel=1.5,
        constante_degradacao=1.0,
    )


def test_q_learning_update_equation_exact(synthetic_ttp_instance):
    """
    Audita um passo determinístico (epsilon=0.0) do RLNS para comprovar que a
    Equação (3) / Equação (7) atualiza Q(s_t, a_t) exatamente como especificado.
    """
    prep = build_one_tree_preprocessing(
        synthetic_ttp_instance.grafo, num_candidates=5
    )
    op = Mov2OptReempOperator(
        instance=synthetic_ttp_instance,
        k=1.0,
        beta=1.0,
        gamma=1.0,
        delta=0.8,
        modelo="TTP1",
    )
    # Rota propositalmente embaralhada (com cidade 1 fixa no início)
    init_route = [1, 5, 2, 7, 3, 8, 4, 6]
    init_state = op.build_initial_state(init_route, {2: [1], 4: [3], 6: [5]})

    # Restringe a vizinhança a 1 candidato por cidade para rastrear cada passo exato
    single_neighbor = {
        c: [nb for nb in prep.candidate_neighbors[c] if nb != 1][:1]
        for c in range(2, 9)
    }

    q0 = prep.q_table.copy()
    alpha_lr = 0.1
    gamma_disc = 0.9

    # Simula manualmente 1 passagem do Algoritmo 2
    expected_q = q0.copy()
    sim_state = init_state.copy()
    n = len(sim_state.route)
    for idx in range(1, n):
        x_i = sim_state.route[idx]
        x_j = single_neighbor[x_i][0]
        f_before = sim_state.objective
        cand_state = op.apply_move(sim_state, x_i, x_j, in_place=False)
        reward = cand_state.objective - f_before
        next_idx = idx + 1 if (idx + 1) < n else 1
        s_next = sim_state.route[next_idx]
        max_q_next = max(expected_q[s_next, a] for a in single_neighbor[s_next])
        expected_q[x_i, x_j] = (1.0 - alpha_lr) * expected_q[
            x_i, x_j
        ] + alpha_lr * (reward + gamma_disc * max_q_next)

    res = run_rlns_qlearning_search(
        initial_state=init_state,
        operator=op,
        candidate_neighbors=single_neighbor,
        initial_q_table=q0,
        iter_max=1,
        epsilon=0.0,
        alpha_lr=alpha_lr,
        gamma_discount=gamma_disc,
        rng=np.random.default_rng(0),
    )

    assert res.passes_completed == 1
    assert np.allclose(res.q_table, expected_q, atol=1e-9)
    assert res.best_state.objective >= init_state.objective - 1e-9


def test_first_improvement_and_rlns_improve_suboptimal_solution(
    synthetic_ttp_instance,
):
    prep = build_one_tree_preprocessing(
        synthetic_ttp_instance.grafo, num_candidates=6
    )
    op = Mov2OptReempOperator(
        instance=synthetic_ttp_instance,
        k=1.0,
        beta=1.0,
        gamma=1.0,
        delta=0.85,
        modelo="TTP1",
    )
    # Rota cruzada com alto tempo de viagem
    init_route = [1, 5, 2, 6, 3, 7, 4, 8]
    init_state = op.build_initial_state(init_route, {2: [1], 5: [4], 8: [7]})

    res_pm = run_first_improvement_search(
        initial_state=init_state,
        operator=op,
        candidate_neighbors=prep.candidate_neighbors,
        iter_max=100,
        rng=np.random.default_rng(42),
    )
    res_rl = run_rlns_qlearning_search(
        initial_state=init_state,
        operator=op,
        candidate_neighbors=prep.candidate_neighbors,
        initial_q_table=prep.q_table,
        iter_max=20,
        epsilon=0.2,
        rng=np.random.default_rng(42),
    )

    # Ambos devem melhorar estritamente a solução inicial cruzada
    assert res_pm.best_state.objective > init_state.objective + 1e-6
    assert res_rl.best_state.objective > init_state.objective + 1e-6

    # First Improvement deve convergir para ótimo local antes de iter_max=100
    assert res_pm.passes_completed < 100
    assert not res_pm.stopped_by_timeout

    # Equivalência com o avaliador oficial
    eval_pm = evaluate_ttp1(
        synthetic_ttp_instance, res_pm.best_state.to_solucao_ttp()
    )
    eval_rl = evaluate_ttp1(
        synthetic_ttp_instance, res_rl.best_state.to_solucao_ttp()
    )
    assert res_pm.best_state.objective == pytest.approx(
        eval_pm["lucro_liquido_G"], abs=1e-6
    )
    assert res_rl.best_state.objective == pytest.approx(
        eval_rl["lucro_liquido_G"], abs=1e-6
    )


def test_timeout_respected_on_real_instance():
    inst_path = Path(__file__).parent.parent / "bateria_testes" / "11.ttp"
    if not inst_path.exists():
        pytest.skip("Instância 11.ttp não encontrada.")

    np.random.seed(42)
    instance = InstanceParser.carregar(str(inst_path))
    prep = build_one_tree_preprocessing(instance.grafo, num_candidates=15)

    # Usa rota inicial simples para testar rapidamente a busca local com deadline curto
    init_route = list(range(1, instance.num_cities + 1))
    igp = IGP(
        scenario=instance,
        tour=ChainedLK(grafo=instance.grafo, num_tries=2, num_kicks=0).solve(),
        modelo="TTP1",
    )
    init_plan = igp.solve(theta=5.0, num_iterations=3)

    op = Mov2OptReempOperator(
        instance=instance,
        k=igp.best_k,
        beta=igp.best_beta,
        gamma=igp.best_gamma,
        delta=igp.best_delta,
        modelo="TTP1",
    )
    init_state = op.build_initial_state(init_route, init_plan)

    deadline_rl = time.perf_counter() + 0.05
    res_rl = run_rlns_qlearning_search(
        initial_state=init_state,
        operator=op,
        candidate_neighbors=prep.candidate_neighbors,
        initial_q_table=prep.q_table,
        iter_max=1000,
        deadline=deadline_rl,
        rng=np.random.default_rng(7),
    )
    assert res_rl.stopped_by_timeout
    assert res_rl.best_state.objective >= init_state.objective - 1e-9
