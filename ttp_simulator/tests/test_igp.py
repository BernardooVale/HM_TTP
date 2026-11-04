import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.core.models import Cidade, Grafo, Item, Tour, TTPInstance
from src.solvers.packing.greedy import IGP


@pytest.fixture
def grafo_basico():
    cidades = {
        1: Cidade(id=1, x=0.0, y=0.0),
        2: Cidade(id=2, x=10.0, y=0.0),
        3: Cidade(id=3, x=20.0, y=0.0),
    }
    return Grafo(cidades=cidades, edge_weight_type="EUC_2D")


@pytest.fixture
def tour_basico():
    return Tour(order=[1, 2, 3], cost=40.0)


def test_igp_sem_itens(grafo_basico, tour_basico):
    instancia = TTPInstance(
        nome="inst_vazia",
        grafo=grafo_basico,
        itens=[],
        capacidade_mochila=50.0,
        v_min=0.1,
        v_max=1.0,
        taxa_aluguel=1.0,
        constante_degradacao=1.0,
    )
    igp = IGP(scenario=instancia, tour=tour_basico)
    resultado = igp.solve()
    assert resultado == {}


def test_igp_dimension_zero_poucos_itens(grafo_basico, tour_basico):
    itens = [Item(id=1, peso=10.0, valor=50.0, cidade_origem=2)]
    instancia = TTPInstance(
        nome="inst_um_item",
        grafo=grafo_basico,
        itens=itens,
        capacidade_mochila=50.0,
        v_min=0.1,
        v_max=1.0,
        taxa_aluguel=1.0,
        constante_degradacao=1.0,
        dimension=0,
    )
    igp = IGP(scenario=instancia, tour=tour_basico)
    resultado = igp.solve()
    assert isinstance(resultado, dict)
    assert resultado == {2: [1]}


def test_igp_maximiza_lucro(grafo_basico, tour_basico):
    itens = [
        Item(id=1, peso=0.5, valor=1.0, cidade_origem=2),
        Item(id=2, peso=20.0, valor=1000.0, cidade_origem=2),
        Item(id=3, peso=20.0, valor=1000.0, cidade_origem=3),
    ]
    instancia = TTPInstance(
        nome="inst_lucro",
        grafo=grafo_basico,
        itens=itens,
        capacidade_mochila=50.0,
        v_min=0.1,
        v_max=1.0,
        taxa_aluguel=1.0,
        constante_degradacao=1.0,
    )
    igp = IGP(scenario=instancia, tour=tour_basico)
    resultado = igp.solve()

    coletados = set()
    for ids in resultado.values():
        coletados.update(ids)

    # Garante que os itens mais lucrativos foram selecionados
    assert 2 in coletados and 3 in coletados


def test_igp_parse_solution():
    itens = [
        Item(id=10, peso=5.0, valor=100.0, cidade_origem=2),
        Item(id=20, peso=5.0, valor=200.0, cidade_origem=2),
        Item(id=30, peso=5.0, valor=300.0, cidade_origem=3),
    ]
    plano = IGP.parse_solution(itens)
    assert plano == {2: [10, 20], 3: [30]}
