import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

import src.solvers  # noqa: F401 - garante registro dos solvers
from src.core.evaluator_ttp1 import avaliar_ttp1
from src.core.models import Cidade, Grafo, Item, SolucaoTTP, TTPInstance
from src.solvers.base import SolverRegistry
from src.solvers.operators.cosolver2b_operators import (
    bit_flip_cosolver2b,
    gerar_grafo_delaunay,
    indexar_itens_por_cidade,
    inicializar_acumuladores,
    insercao_e_eliminacao,
    two_opt_delaunay,
)


@pytest.fixture
def instance_12_cidades():
    rng = np.random.default_rng(23)
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
        nome="TestCosolver2B12",
        grafo=grafo,
        itens=itens,
        capacidade_mochila=110.0,
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


def test_delaunay_e_acumuladores_batem_com_avaliador(instance_12_cidades):
    adj = gerar_grafo_delaunay(instance_12_cidades.grafo)
    assert set(adj.keys()) == set(range(1, 13))
    for cid, viz in adj.items():
        assert cid not in viz
        assert len(viz) >= 2

    rota = list(range(1, 13))
    plano, z_ie = insercao_e_eliminacao(instance_12_cidades, rota)
    sol = SolucaoTTP(rota=rota, plano_coleta=plano)
    _validar_solucao(instance_12_cidades, sol)

    res_oficial = avaliar_ttp1(instance_12_cidades, sol)["lucro_liquido_G"]
    assert z_ie == pytest.approx(res_oficial, abs=1e-6)

    _, _, _, _, _, z_acc = inicializar_acumuladores(instance_12_cidades, rota, plano)
    assert z_acc == pytest.approx(res_oficial, abs=1e-6)


@pytest.mark.parametrize("estrategia", ["first_fit", "best_fit"])
def test_two_opt_delaunay_e_bit_flip_equivalentes_avaliador(instance_12_cidades, estrategia):
    adj = gerar_grafo_delaunay(instance_12_cidades.grafo)
    itens_por_cid = indexar_itens_por_cidade(instance_12_cidades)
    rota = list(range(1, 13))
    plano, z_init = insercao_e_eliminacao(instance_12_cidades, rota)

    rota_opt, z_rota, _ = two_opt_delaunay(
        instance_12_cidades, rota, plano, adj, estrategia=estrategia
    )
    sol_r = SolucaoTTP(rota=rota_opt, plano_coleta=plano)
    _validar_solucao(instance_12_cidades, sol_r)
    assert z_rota == pytest.approx(avaliar_ttp1(instance_12_cidades, sol_r)["lucro_liquido_G"], abs=1e-6)
    assert z_rota >= z_init - 1e-9

    plano_opt, z_item, _ = bit_flip_cosolver2b(
        instance_12_cidades, rota_opt, plano, itens_por_cid, estrategia=estrategia
    )
    sol_i = SolucaoTTP(rota=rota_opt, plano_coleta=plano_opt)
    _validar_solucao(instance_12_cidades, sol_i)
    assert z_item == pytest.approx(avaliar_ttp1(instance_12_cidades, sol_i)["lucro_liquido_G"], abs=1e-6)
    assert z_item >= z_rota - 1e-9


@pytest.mark.parametrize("solver_name", ["cosolver2b_first_fit", "cosolver2b_best_fit"])
def test_cosolver2b_registrados_e_deterministicos(solver_name, instance_12_cidades):
    assert solver_name in SolverRegistry.listar()
    s1 = SolverRegistry.get(solver_name)
    s2 = SolverRegistry.get(solver_name)
    s1.seed = 42
    s2.seed = 42

    sol1 = s1.solve(instance_12_cidades)
    sol2 = s2.solve(instance_12_cidades)

    _validar_solucao(instance_12_cidades, sol1)
    assert sol1.rota == sol2.rota
    assert sol1.plano_coleta == sol2.plano_coleta
    assert np.isfinite(avaliar_ttp1(instance_12_cidades, sol1)["lucro_liquido_G"])
