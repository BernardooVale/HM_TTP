import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

import src.solvers  # noqa: F401 - garante registro dos solvers
from src.core.evaluator_ttp1 import avaliar_ttp1
from src.core.models import Cidade, Grafo, Item, SolucaoTTP, TTPInstance
from src.solvers.base import SolverRegistry
from src.solvers.iterative.coco_solvers import CoCoSolver
from src.solvers.operators.coco_operators import (
    boundary_bit_flip,
    calcular_perfis_e_fronteira_coco,
    pgch_estendido_move,
)
from src.solvers.operators.pgch_operators import preparar_itens_ordenados_por_cidade


@pytest.fixture
def instance_12_cidades():
    rng = np.random.default_rng(31)
    cidades = {
        i: Cidade(
            id=i,
            x=float(rng.uniform(0, 100)),
            y=float(rng.uniform(0, 100)),
        )
        for i in range(1, 13)
    }
    itens = []
    item_id = 1
    for c_id in range(2, 13):
        for _ in range(3):
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
        nome="TestCoCo12",
        grafo=grafo,
        itens=itens,
        capacidade_mochila=120.0,
        v_min=0.1,
        v_max=1.0,
        taxa_aluguel=0.5,
        constante_degradacao=10.0,
        dimension=12,
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


def test_perfis_e_fronteira_coco(instance_12_cidades):
    por_cid, razoes, sentinela = preparar_itens_ordenados_por_cidade(instance_12_cidades)
    rota = list(range(1, 13))
    plano = {c: [por_cid[c][-1].id] for c in (2, 6, 10)}

    pi_p, omega_q, bolsa = calcular_perfis_e_fronteira_coco(
        rota, plano, por_cid, razoes, sentinela
    )

    assert pi_p[0] == pytest.approx(sentinela)
    for k in range(1, len(pi_p)):
        assert pi_p[k] <= pi_p[k - 1] + 1e-12
    for k in range(1, len(rota) - 1):
        assert omega_q[k] >= omega_q[k + 1] - 1e-12
    assert len(bolsa) > 0


def test_pgch_estendido_respeita_peso_e_inverte(instance_12_cidades):
    por_cid, razoes, sentinela = preparar_itens_ordenados_por_cidade(instance_12_cidades)
    rota = list(range(1, 13))
    plano = {}
    peso_acc = 0.0
    for c in (2, 4, 6, 8, 10, 12):
        it = por_cid[c][-1]
        if peso_acc + it.peso <= instance_12_cidades.capacidade_mochila:
            plano[c] = [it.id]
            peso_acc += it.peso

    pi_p, omega_q, _ = calcular_perfis_e_fronteira_coco(
        rota, plano, por_cid, razoes, sentinela
    )
    r_cand, p_cand = pgch_estendido_move(
        rota, plano, 2, 9, pi_p, omega_q, por_cid, razoes
    )

    assert r_cand[2:10] == rota[2:10][::-1]
    _validar_solucao(instance_12_cidades, SolucaoTTP(rota=r_cand, plano_coleta=p_cand))

    itens_dict = {it.id: it for it in instance_12_cidades.itens}
    peso_novo = sum(itens_dict[i].peso for ids in p_cand.values() for i in ids)
    assert peso_novo <= peso_acc + 1e-6


def test_boundary_bit_flip_melhora_ou_mantem(instance_12_cidades):
    por_cid, razoes, sentinela = preparar_itens_ordenados_por_cidade(instance_12_cidades)
    rota = list(range(1, 13))
    plano_init = {}
    z_init = avaliar_ttp1(instance_12_cidades, SolucaoTTP(rota=rota, plano_coleta=plano_init))["lucro_liquido_G"]

    rng = np.random.default_rng(7)
    plano_opt, z_opt, _ = boundary_bit_flip(
        instance_12_cidades, rota, plano_init, por_cid, razoes, sentinela, rng=rng
    )
    sol = SolucaoTTP(rota=rota, plano_coleta=plano_opt)
    _validar_solucao(instance_12_cidades, sol)
    assert z_opt == pytest.approx(avaliar_ttp1(instance_12_cidades, sol)["lucro_liquido_G"], abs=1e-6)
    assert z_opt >= z_init - 1e-9


def test_coco_solver_registrado_e_valido(instance_12_cidades):
    assert "coco_solver" in SolverRegistry.listar()
    solver = SolverRegistry.get("coco_solver")
    assert isinstance(solver, CoCoSolver)
    solver.time_limit = 0.08
    solver.seed = 42

    sol = solver.solve(instance_12_cidades)
    _validar_solucao(instance_12_cidades, sol)
    assert np.isfinite(avaliar_ttp1(instance_12_cidades, sol)["lucro_liquido_G"])
