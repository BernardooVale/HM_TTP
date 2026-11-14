import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

import src.solvers  # noqa: F401 - garante registro dos solvers
from src.core.evaluator_ttp1 import avaliar_ttp1
from src.core.models import Cidade, Grafo, Item, SolucaoTTP, TTPInstance
from src.solvers.base import SolverRegistry
from src.solvers.iterative.pgch_solvers import CTTP2OptSolver, CTTPPGCHSolver
from src.solvers.operators.pgch_operators import (
    calcular_prefixo_minimo,
    pgch_move,
    preparar_itens_ordenados_por_cidade,
    rand_flip_one_item,
    two_opt_convencional_move,
)


@pytest.fixture
def instance_10_cidades():
    rng = np.random.default_rng(17)
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
        nome="TestPGCH10",
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


def test_prefixo_minimo_monotonico(instance_10_cidades):
    por_cid, razoes, sentinela = preparar_itens_ordenados_por_cidade(instance_10_cidades)
    rota = list(range(1, 11))
    # Coleta o item menos rentável de algumas cidades
    plano = {c: [por_cid[c][-1].id] for c in (2, 5, 8)}
    pi_cp = calcular_prefixo_minimo(rota, plano, razoes, sentinela)

    assert pi_cp[0] == pytest.approx(sentinela)
    for k in range(1, len(pi_cp)):
        assert pi_cp[k] <= pi_cp[k - 1] + 1e-12


def test_pgch_move_respeita_peso_e_reverte_segmento(instance_10_cidades):
    por_cid, razoes, sentinela = preparar_itens_ordenados_por_cidade(instance_10_cidades)
    rota = list(range(1, 11))
    # Plano inicial viável: 1 item por cidade par (respeitando capacidade 90)
    plano = {}
    peso_acc = 0.0
    for c in (2, 4, 6, 8, 10):
        it = por_cid[c][-1]
        if peso_acc + it.peso <= instance_10_cidades.capacidade_mochila:
            plano[c] = [it.id]
            peso_acc += it.peso

    pi_cp = calcular_prefixo_minimo(rota, plano, razoes, sentinela)
    r_2opt, p_2opt = two_opt_convencional_move(rota, plano, 2, 7)
    r_pgch, p_pgch = pgch_move(rota, plano, 2, 7, pi_cp, por_cid, razoes)

    assert r_2opt == r_pgch
    assert r_pgch[2:8] == rota[2:8][::-1]
    assert p_2opt == {c: ids for c, ids in plano.items() if ids}
    _validar_solucao(instance_10_cidades, SolucaoTTP(rota=r_pgch, plano_coleta=p_pgch))

    itens_dict = {it.id: it for it in instance_10_cidades.itens}
    peso_pgch = sum(itens_dict[i].peso for ids in p_pgch.values() for i in ids)
    assert peso_pgch <= peso_acc + 1e-6


def test_rand_flip_one_item_melhora_ou_mantem(instance_10_cidades):
    rota = list(range(1, 11))
    plano_init = {}
    z_init = avaliar_ttp1(instance_10_cidades, SolucaoTTP(rota=rota, plano_coleta=plano_init))["lucro_liquido_G"]

    rng = np.random.default_rng(42)
    plano_opt, z_opt, _ = rand_flip_one_item(
        instance_10_cidades, rota, plano_init, duration_seconds=0.02, rng=rng, min_iters=20
    )
    sol = SolucaoTTP(rota=rota, plano_coleta=plano_opt)
    _validar_solucao(instance_10_cidades, sol)
    assert z_opt == pytest.approx(avaliar_ttp1(instance_10_cidades, sol)["lucro_liquido_G"], abs=1e-6)
    assert z_opt >= z_init - 1e-9


@pytest.mark.parametrize("solver_name", ["cttp_2opt", "cttp_pgch"])
def test_solvers_cttp_registrados_e_validos(solver_name, instance_10_cidades):
    assert solver_name in SolverRegistry.listar()
    solver = SolverRegistry.get(solver_name)
    solver.time_limit = 0.08
    sol = solver.solve(instance_10_cidades)
    _validar_solucao(instance_10_cidades, sol)
    res = avaliar_ttp1(instance_10_cidades, sol)
    assert np.isfinite(res["lucro_liquido_G"])


def test_cttp_instancia_direta_com_seed(instance_10_cidades):
    s_2opt = CTTP2OptSolver(time_limit=0.05, seed=10)
    s_pgch = CTTPPGCHSolver(time_limit=0.05, seed=10)

    sol1 = s_2opt.solve(instance_10_cidades)
    sol2 = s_pgch.solve(instance_10_cidades)
    _validar_solucao(instance_10_cidades, sol1)
    _validar_solucao(instance_10_cidades, sol2)
