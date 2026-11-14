import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

import src.solvers  # noqa: F401 - garante registro de todos os solvers
from src.core.evaluator_ttp1 import avaliar_ttp1
from src.core.models import Cidade, Grafo, Item, SolucaoTTP, TTPInstance
from src.solvers.base import SolverRegistry
from src.solvers.operators.faulkner_operators import (
    bit_flip,
    insertion_local_search,
    one_plus_one_ea,
    pack,
    pack_iterative,
)
from src.solvers.iterative.faulkner_solvers import (
    FaulknerC1Solver,
    FaulknerC2Solver,
    FaulknerC3Solver,
    FaulknerC4Solver,
    FaulknerC5Solver,
    FaulknerC6Solver,
    FaulknerS3Solver,
    FaulknerS5Solver,
)
from src.solvers.tsp import ChainedLK


@pytest.fixture
def instance_10_cidades():
    rng = np.random.default_rng(7)
    cidades = {
        i: Cidade(
            id=i,
            x=float(rng.uniform(0, 100)),
            y=float(rng.uniform(0, 100)),
        )
        for i in range(1, 11)
    }
    itens = []
    item_id = 1
    for c_id in range(2, 11):
        for _ in range(2):
            itens.append(
                Item(
                    id=item_id,
                    peso=float(rng.integers(5, 30)),
                    valor=float(rng.integers(20, 120)),
                    cidade_origem=c_id,
                )
            )
            cidades[c_id].itens.append(itens[-1])
            item_id += 1

    grafo = Grafo(cidades=cidades, edge_weight_type="EUC_2D")
    return TTPInstance(
        nome="Test10",
        grafo=grafo,
        itens=itens,
        capacidade_mochila=90.0,
        v_min=0.1,
        v_max=1.0,
        taxa_aluguel=0.5,
        constante_degradacao=10.0,
        dimension=10,
        num_items=len(itens),
    )


def _validar_solucao(instance: TTPInstance, solucao: SolucaoTTP):
    n = len(instance.grafo.cidades)
    assert len(solucao.rota) == n
    assert set(solucao.rota) == set(instance.grafo.cidades.keys())
    assert solucao.rota[0] == 1

    itens_dict = {it.id: it for it in instance.itens}
    peso_total = 0.0
    for c_id, ids in solucao.plano_coleta.items():
        assert c_id != 1
        for it_id in ids:
            assert itens_dict[it_id].cidade_origem == c_id
            peso_total += itens_dict[it_id].peso
    assert peso_total <= instance.capacidade_mochila + 1e-6


def test_clk_seed_e_random_start(instance_10_cidades):
    clk_a = ChainedLK(grafo=instance_10_cidades.grafo, num_tries=5, num_kicks=2, seed=123)
    clk_b = ChainedLK(grafo=instance_10_cidades.grafo, num_tries=5, num_kicks=2, seed=123)

    t_a = clk_a.solve(random_start=True)
    t_b = clk_b.solve(random_start=True)

    assert t_a.order == t_b.order
    assert t_a.order[0] == 1
    assert set(t_a.order) == set(range(1, 11))


def test_pack_e_pack_iterative_equivalentes_avaliador(instance_10_cidades):
    rota = ChainedLK(grafo=instance_10_cidades.grafo, seed=1).solve().order

    plano_p, z_p = pack(instance_10_cidades, rota, alpha=5.0)
    sol_p = SolucaoTTP(rota=rota, plano_coleta=plano_p)
    _validar_solucao(instance_10_cidades, sol_p)
    assert z_p == pytest.approx(avaliar_ttp1(instance_10_cidades, sol_p)["lucro_liquido_G"], abs=1e-6)

    plano_pi, z_pi = pack_iterative(instance_10_cidades, rota)
    sol_pi = SolucaoTTP(rota=rota, plano_coleta=plano_pi)
    _validar_solucao(instance_10_cidades, sol_pi)
    assert z_pi == pytest.approx(avaliar_ttp1(instance_10_cidades, sol_pi)["lucro_liquido_G"], abs=1e-6)
    assert z_pi >= z_p - 1e-9


def test_bit_flip_diferencial_bate_com_avaliador(instance_10_cidades):
    rota = ChainedLK(grafo=instance_10_cidades.grafo, seed=2).solve().order
    plano_init, z_init = pack_iterative(instance_10_cidades, rota)

    plano_1, z_1, _ = bit_flip(instance_10_cidades, rota, plano_init, max_passes=1)
    sol_1 = SolucaoTTP(rota=rota, plano_coleta=plano_1)
    _validar_solucao(instance_10_cidades, sol_1)
    assert z_1 == pytest.approx(avaliar_ttp1(instance_10_cidades, sol_1)["lucro_liquido_G"], abs=1e-6)
    assert z_1 >= z_init - 1e-9

    plano_conv, z_conv, _ = bit_flip(instance_10_cidades, rota, plano_init, max_passes=None)
    sol_conv = SolucaoTTP(rota=rota, plano_coleta=plano_conv)
    _validar_solucao(instance_10_cidades, sol_conv)
    assert z_conv == pytest.approx(avaliar_ttp1(instance_10_cidades, sol_conv)["lucro_liquido_G"], abs=1e-6)
    assert z_conv >= z_1 - 1e-9


def test_one_plus_one_ea(instance_10_cidades):
    rota = ChainedLK(grafo=instance_10_cidades.grafo, seed=3).solve().order
    plano_init, z_init = pack_iterative(instance_10_cidades, rota)

    rng = np.random.default_rng(99)
    plano_ea, z_ea, _ = one_plus_one_ea(
        instance_10_cidades, rota, plano_init, max_iters=200, rng=rng
    )
    sol_ea = SolucaoTTP(rota=rota, plano_coleta=plano_ea)
    _validar_solucao(instance_10_cidades, sol_ea)
    assert z_ea == pytest.approx(avaliar_ttp1(instance_10_cidades, sol_ea)["lucro_liquido_G"], abs=1e-6)
    assert z_ea >= z_init - 1e-9


def test_insertion_segmento_parcial_bate_com_avaliador(instance_10_cidades):
    # Rota deliberadamente subótima (ordem sequencial) com itens pesados no início
    rota_init = list(range(1, 11))
    plano_init, z_init = pack_iterative(instance_10_cidades, rota_init)

    rota_opt, z_opt, _ = insertion_local_search(
        instance_10_cidades, rota_init, plano_init, max_passes=None
    )
    sol_opt = SolucaoTTP(rota=rota_opt, plano_coleta=plano_init)
    _validar_solucao(instance_10_cidades, sol_opt)
    assert z_opt == pytest.approx(avaliar_ttp1(instance_10_cidades, sol_opt)["lucro_liquido_G"], abs=1e-6)
    assert z_opt >= z_init - 1e-9


@pytest.mark.parametrize(
    "solver_name",
    [
        "faulkner_s1",
        "faulkner_s2",
        "faulkner_s3",
        "faulkner_s4",
        "faulkner_s5",
        "faulkner_c1",
        "faulkner_c2",
        "faulkner_c3",
        "faulkner_c4",
        "faulkner_c5",
        "faulkner_c6",
    ],
)
def test_todos_solvers_faulkner_registrados(solver_name, instance_10_cidades):
    assert solver_name in SolverRegistry.listar()
    solver = SolverRegistry.get(solver_name)
    # Reduz orçamento temporal para testes rápidos mantendo validação do fluxo real
    if hasattr(solver, "time_limit"):
        solver.time_limit = 0.05
    if hasattr(solver, "ea_iters_c2"):
        solver.ea_iters_c2 = 100
    solucao = solver.solve(instance_10_cidades)
    _validar_solucao(instance_10_cidades, solucao)
    res = avaliar_ttp1(instance_10_cidades, solucao)
    assert np.isfinite(res["lucro_liquido_G"])


def test_solvers_faulkner_orcamento_tempo_explicito(instance_10_cidades):
    for cls in (
        FaulknerS3Solver,
        FaulknerS5Solver,
        FaulknerC3Solver,
        FaulknerC4Solver,
        FaulknerC5Solver,
        FaulknerC6Solver,
    ):
        solver = cls(time_limit=0.05, seed=42, ea_iters_c2=100)
        sol = solver.solve(instance_10_cidades)
        _validar_solucao(instance_10_cidades, sol)

    for cls in (FaulknerC1Solver, FaulknerC2Solver):
        solver = cls(seed=42, ea_iters_c2=100)
        sol = solver.solve(instance_10_cidades)
        _validar_solucao(instance_10_cidades, sol)
