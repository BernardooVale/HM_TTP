import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

import src.solvers  # noqa: F401 - garante registro dos solvers
from src.core.evaluator_ttp1 import avaliar_ttp1
from src.core.models import Cidade, Grafo, Item, SolucaoTTP, TTPInstance
from src.solvers.base import SolverRegistry
from src.solvers.iterative.wsm_solvers import WSMSolver
from src.solvers.operators.wsm_operators import (
    ParetoEntry,
    atualizar_arquivo_pareto,
    avaliar_biobjetivo,
    randomized_packing_algorithm,
)


@pytest.fixture
def instance_10_cidades():
    rng = np.random.default_rng(55)
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
        nome="TestWSM10",
        grafo=grafo,
        itens=itens,
        capacidade_mochila=100.0,
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


def test_arquivo_pareto_nao_dominancia():
    arquivo: list[ParetoEntry] = []
    assert atualizar_arquivo_pareto(arquivo, [1, 2], {1}, g=100.0, h=10.0)
    # Solução dominada (menor lucro e maior tempo) deve ser rejeitada
    assert not atualizar_arquivo_pareto(arquivo, [1, 2], {2}, g=90.0, h=12.0)
    # Solução de trade-off (maior lucro, maior tempo) deve entrar
    assert atualizar_arquivo_pareto(arquivo, [1, 2], {1, 2}, g=150.0, h=14.0)
    assert len(arquivo) == 2
    # Solução que domina a primeira (maior lucro e menor tempo) deve substituí-la
    assert atualizar_arquivo_pareto(arquivo, [1, 2], {3}, g=110.0, h=9.0)
    assert len(arquivo) == 2
    assert all(e.lucro_bruto != 100.0 for e in arquivo)


def test_randomized_packing_bate_com_avaliador(instance_10_cidades):
    rota = list(range(1, 11))
    rng = np.random.default_rng(12)
    itens_ids, g, h = randomized_packing_algorithm(
        instance_10_cidades, rota, rho=6, alpha=0.5, gamma=41.0, rng=rng
    )
    itens_by_id = {it.id: it for it in instance_10_cidades.itens}
    g_ref, h_ref = avaliar_biobjetivo(instance_10_cidades, rota, itens_ids, itens_by_id)
    assert g == pytest.approx(g_ref, abs=1e-6)
    assert h == pytest.approx(h_ref, abs=1e-6)

    plano = {}
    for it_id in itens_ids:
        plano.setdefault(itens_by_id[it_id].cidade_origem, []).append(it_id)
    sol = SolucaoTTP(rota=rota, plano_coleta=plano)
    _validar_solucao(instance_10_cidades, sol)
    res = avaliar_ttp1(instance_10_cidades, sol)
    assert g - instance_10_cidades.taxa_aluguel * h == pytest.approx(res["lucro_liquido_G"], abs=1e-6)


def test_wsm_solver_registrado_e_fronteira_pareto(instance_10_cidades):
    assert "wsm_solver" in SolverRegistry.listar()
    solver = SolverRegistry.get("wsm_solver")
    assert isinstance(solver, WSMSolver)
    solver.time_limit = 0.08
    solver.seed = 42
    solver.eta = 15
    solver.rho = 4

    sol = solver.solve(instance_10_cidades)
    _validar_solucao(instance_10_cidades, sol)
    assert len(solver.last_pareto_archive) >= 1

    # Nenhuma solução no arquivo pode dominar outra
    arq = solver.last_pareto_archive
    for i, a in enumerate(arq):
        for j, b in enumerate(arq):
            if i != j:
                assert not (
                    a.lucro_bruto >= b.lucro_bruto
                    and a.tempo_viagem <= b.tempo_viagem
                    and (a.lucro_bruto > b.lucro_bruto or a.tempo_viagem < b.tempo_viagem)
                )
