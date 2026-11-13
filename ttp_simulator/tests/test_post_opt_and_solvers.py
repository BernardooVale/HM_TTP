import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.core.evaluator_ttp1 import evaluate as evaluate_ttp1
from src.core.models import Cidade, Grafo, Item, SolucaoTTP, TTPInstance
from src.scenarios.parser import InstanceParser
from src.solvers.base import SolverRegistry
from src.solvers.iterative.reinforcens_post_opt import (
    apply_post_optimization,
    compute_rrp_delta_psi_eq8,
)
from src.solvers.iterative.reinforcens_solver import (
    ReinforceNSPMSolver,
    ReinforceNSSolver,
)


@pytest.fixture
def synthetic_post_opt_instance():
    """
    Instância de 10 cidades onde há itens pesados coletados no início da rota
    e cidades vazias no final da rota, permitindo que tanto o IFP (P1) quanto
    o RRP (P2) e a combinação (P3) melhorem estritamente a solução.
    """
    rng = np.random.default_rng(99)
    cidades = {
        i: Cidade(
            id=i,
            x=float(rng.uniform(0, 200)),
            y=float(rng.uniform(0, 200)),
        )
        for i in range(1, 11)
    }
    # Itens apenas nas cidades 2, 3, 4 e 5; cidades 6..10 ficam sem itens coletados
    itens = [
        Item(id=1, peso=35.0, valor=500.0, cidade_origem=2),
        Item(id=2, peso=25.0, valor=450.0, cidade_origem=3),
        Item(id=3, peso=15.0, valor=800.0, cidade_origem=4),
        Item(id=4, peso=10.0, valor=10.0, cidade_origem=5),  # Item pesado/pouco lucrativo
    ]
    for it in itens:
        cidades[it.cidade_origem].itens.append(it)

    grafo = Grafo(cidades=cidades, edge_weight_type="EUC_2D")
    return TTPInstance(
        nome="post_opt_10",
        grafo=grafo,
        itens=itens,
        capacidade_mochila=80.0,
        v_min=0.1,
        v_max=1.0,
        taxa_aluguel=2.0,
        constante_degradacao=1.0,
    )


def test_rrp_equation_8_matches_paper_figure_1():
    """
    Valida a Equação (8) exatamente no exemplo numérico da Figura 1 (página 5)
    do artigo original ReinforceNS [Wu et al., 2024]:
      d(x_i, x_{i+1}) = 7, d(x_{j-1}, x_j) = 4, d(x_j, x_{j+1}) = 3
      d(x_i, x_j) = 5, d(x_j, x_{i+1}) = 6, d(x_{j-1}, x_{j+1}) = 5
      v_{x_i} = 5, v_{x_{j-1}} = 2
      Resultado esperado no artigo: Δψ = (5+6)/5 + 5/2 - (7/5 + (3+4)/2) = -0.2
    """
    dist_mtx = np.zeros((7, 7), dtype=float)
    x_i, x_i_next = 1, 2
    x_j_prev, x_j, x_j_next = 3, 4, 5

    dist_mtx[x_i, x_i_next] = 7.0
    dist_mtx[x_j_prev, x_j] = 4.0
    dist_mtx[x_j, x_j_next] = 3.0
    dist_mtx[x_i, x_j] = 5.0
    dist_mtx[x_j, x_i_next] = 6.0
    dist_mtx[x_j_prev, x_j_next] = 5.0

    delta_psi = compute_rrp_delta_psi_eq8(
        dist_mtx=dist_mtx,
        x_i=x_i,
        x_i_next=x_i_next,
        x_j_prev=x_j_prev,
        x_j=x_j,
        x_j_next=x_j_next,
        v_xi=5.0,
        v_xj_prev=2.0,
    )
    assert delta_psi == pytest.approx(-0.2, abs=1e-12)


def test_rrp_equation_8_matches_full_evaluator_for_all_edges(
    synthetic_post_opt_instance,
):
    """
    Compara Δψ calculado em O(1) pela Equação (8) contra a diferença real de
    tempo de viagem (ψ(Π', P) - ψ(Π, P)) do avaliador oficial para todas as
    cidades sem itens coletados e todas as arestas anteriores.
    """
    inst = synthetic_post_opt_instance
    route = list(range(1, 11))
    plan = {2: [1], 3: [2]}  # Cidades 4..10 não têm itens coletados

    res_base = evaluate_ttp1(inst, SolucaoTTP(rota=route, plano_coleta=plan))
    psi_base = res_base["tempo_viagem_f"]

    # Calcula velocidades reais ao sair de cada cidade na rota base
    item_by_id = {it.id: it for it in inst.itens}
    city_w = {
        c: sum(item_by_id[i].peso for i in plan.get(c, []))
        for c in route
    }
    speeds = []
    w_acc = 0.0
    factor = (inst.v_max - inst.v_min) / inst.capacidade_mochila
    for c in route:
        w_acc += city_w[c]
        speeds.append(max(inst.v_min, inst.v_max - w_acc * factor))

    n = len(route)
    for pos_j in range(3, n):
        x_j = route[pos_j]
        assert city_w[x_j] == 0.0
        x_j_prev = route[pos_j - 1]
        x_j_next = route[(pos_j + 1) % n]

        for i in range(0, pos_j - 1):
            x_i = route[i]
            x_i_next = route[i + 1]

            delta_psi_eq8 = compute_rrp_delta_psi_eq8(
                dist_mtx=inst.grafo.distance_matrix,
                x_i=x_i,
                x_i_next=x_i_next,
                x_j_prev=x_j_prev,
                x_j=x_j,
                x_j_next=x_j_next,
                v_xi=speeds[i],
                v_xj_prev=speeds[pos_j - 1],
            )

            # Constrói a rota modificada Π' movendo x_j para entre x_i e x_{i+1}
            new_route = route.copy()
            new_route.pop(pos_j)
            new_route.insert(i + 1, x_j)
            res_new = evaluate_ttp1(
                inst, SolucaoTTP(rota=new_route, plano_coleta=plan)
            )
            psi_new = res_new["tempo_viagem_f"]

            assert delta_psi_eq8 == pytest.approx(
                psi_new - psi_base, abs=1e-9
            )


def test_post_optimization_modes_p0_p1_p2_p3_monotonicity(
    synthetic_post_opt_instance,
):
    inst = synthetic_post_opt_instance
    route = list(range(1, 11))
    # Plano inicial subótimo: inclui o item 1 e 4, mas deixa de fora o item 3 (muito valioso)
    init_plan = {2: [1], 5: [4]}

    res_p0 = apply_post_optimization(inst, route, init_plan, mode="P0")
    res_p1 = apply_post_optimization(inst, route, init_plan, mode="P1")
    res_p2 = apply_post_optimization(inst, route, init_plan, mode="P2")
    res_p3 = apply_post_optimization(inst, route, init_plan, mode="P3")

    # P0 não altera nada
    assert res_p0.objective_final == pytest.approx(
        res_p0.objective_before, abs=1e-9
    )
    assert res_p0.ifp_flips_accepted == 0
    assert res_p0.rrp_relocations_accepted == 0

    # P1 (Item Flipping) e P2 (Route Refining) melhoram estritamente sobre P0
    assert res_p1.objective_final > res_p0.objective_final + 1e-6
    assert res_p1.ifp_flips_accepted > 0
    assert res_p1.rrp_relocations_accepted == 0

    assert res_p2.objective_final > res_p0.objective_final + 1e-6
    assert res_p2.ifp_flips_accepted == 0
    assert res_p2.rrp_relocations_accepted > 0

    # P3 (IFP + RRP) nunca piora P1 e bate com o avaliador oficial
    assert res_p3.objective_final >= res_p1.objective_final - 1e-9
    eval_p3 = evaluate_ttp1(
        inst, SolucaoTTP(rota=res_p3.route, plano_coleta=res_p3.plan)
    )
    assert res_p3.objective_final == pytest.approx(
        eval_p3["lucro_liquido_G"], abs=1e-6
    )


def test_reinforcens_and_reinforcens_pm_paired_seed_and_registry(
    synthetic_post_opt_instance,
):
    # Verifica registro no SolverRegistry
    registered = SolverRegistry.listar()
    assert "reinforcens" in registered
    assert "reinforcens_pm" in registered
    assert "reinforcens_p0" in registered
    assert "reinforcens_pm_p0" in registered

    solver_q = ReinforceNSSolver(
        max_starts=1,
        iter_max=15,
        clk_num_tries=3,
        clk_num_kicks=1,
        igp_iterations=5,
        seed=2026,
    )
    solver_pm = ReinforceNSPMSolver(
        max_starts=1,
        iter_max=15,
        clk_num_tries=3,
        clk_num_kicks=1,
        igp_iterations=5,
        seed=2026,
    )

    sol_q = solver_q.solve(synthetic_post_opt_instance)
    sol_pm = solver_pm.solve(synthetic_post_opt_instance)

    # Pareamento estrito (Seção 4.1): mesma semente => solução inicial (Π_0, P_0) idêntica!
    assert solver_q.last_telemetry is not None
    assert solver_pm.last_telemetry is not None
    assert solver_q.last_telemetry.first_start_init_objective == pytest.approx(
        solver_pm.last_telemetry.first_start_init_objective, abs=1e-9
    )

    # Não-degradação em ambos os solvers e validade perante o avaliador oficial
    eval_q = evaluate_ttp1(synthetic_post_opt_instance, sol_q)
    eval_pm = evaluate_ttp1(synthetic_post_opt_instance, sol_pm)
    assert solver_q.last_telemetry.best_objective == pytest.approx(
        eval_q["lucro_liquido_G"], abs=1e-6
    )
    assert solver_pm.last_telemetry.best_objective == pytest.approx(
        eval_pm["lucro_liquido_G"], abs=1e-6
    )
    assert (
        solver_q.last_telemetry.best_objective
        >= solver_q.last_telemetry.first_start_init_objective - 1e-9
    )
    assert (
        solver_pm.last_telemetry.best_objective
        >= solver_pm.last_telemetry.first_start_init_objective - 1e-9
    )
