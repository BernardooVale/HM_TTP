import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import math
from src.core.models import Item, Cidade, Grafo, TTPInstance, SolucaoTTP
from src.core import evaluator_ttp1
from src.core import evaluator_ttp2

@pytest.fixture
def ttp_instance():
    # Cities: triangle (0,0), (3,0), (0,4)
    # Distances: 1->2 (3), 2->3 (5), 3->1 (4)
    cidades = {
        1: Cidade(id=1, x=0, y=0),
        2: Cidade(id=2, x=3, y=0),
        3: Cidade(id=3, x=0, y=4)
    }
    
    itens = [
        Item(id=1, peso=10, valor=100, cidade_origem=2),
        Item(id=2, peso=20, valor=200, cidade_origem=3),
        Item(id=3, peso=30, valor=300, cidade_origem=3)
    ]
    
    cidades[2].itens.append(itens[0])
    cidades[3].itens.extend([itens[1], itens[2]])
    
    grafo = Grafo(cidades=cidades, edge_weight_type='EUC_2D')
    
    return TTPInstance(
        nome='TestInstance',
        grafo=grafo,
        itens=itens,
        capacidade_mochila=100.0,
        v_min=0.1,
        v_max=1.0,
        taxa_aluguel=1.0,
        constante_degradacao=1.0,
        dimension=3,
        num_items=3
    )

def test_ttp1_basic(ttp_instance):
    # Route: 1 -> 2 -> 3 -> 1
    # Items to collect: item 1 at city 2, item 2 at city 3
    solucao = SolucaoTTP(
        rota=[1, 2, 3],
        plano_coleta={2: [1], 3: [2]}
    )
    
    # Expected calculations:
    # 1 -> 2: weight = 0, speed = 1.0, dist = 3.0, time = 3.0 / 1.0 = 3.0
    # 2 -> 3: collect item 1 (weight 10), accumulated weight = 10
    #         speed = 1.0 - 10 * ((1.0 - 0.1) / 100) = 1.0 - 0.09 = 0.91
    #         dist = 5.0, time = 5.0 / 0.91 = 5.494505494505495
    # 3 -> 1: collect item 2 (weight 20), accumulated weight = 30
    #         speed = 1.0 - 30 * 0.009 = 1.0 - 0.27 = 0.73
    #         dist = 4.0, time = 4.0 / 0.73 = 5.47945205479452
    
    expected_travel_time = 3.0 + (5.0 / 0.91) + (4.0 / 0.73)
    expected_profit = 300.0 - (1.0 * expected_travel_time)
    
    result = evaluator_ttp1.evaluate(ttp_instance, solucao)
    
    assert math.isclose(result['tempo_viagem_f'], expected_travel_time)
    assert result['valor_bruto_itens_g'] == 300.0
    assert result['peso_acumulado'] == 30.0
    assert result['qtd_itens_coletados'] == 2
    assert math.isclose(result['lucro_liquido_G'], expected_profit)

def test_ttp1_empty_knapsack(ttp_instance):
    solucao = SolucaoTTP(rota=[1, 2, 3], plano_coleta={})
    
    # Total distance: 3 + 5 + 4 = 12
    # Constant speed = 1.0, travel time = 12.0
    # Profit = 0 - 1.0 * 12.0 = -12.0
    
    result = evaluator_ttp1.evaluate(ttp_instance, solucao)
    
    assert result['tempo_viagem_f'] == 12.0
    assert result['valor_bruto_itens_g'] == 0.0
    assert result['peso_acumulado'] == 0.0
    assert result['lucro_liquido_G'] == -12.0

def test_ttp1_full_knapsack(ttp_instance):
    solucao = SolucaoTTP(
        rota=[1, 2, 3],
        plano_coleta={2: [1], 3: [2, 3]}
    )
    
    result = evaluator_ttp1.evaluate(ttp_instance, solucao)
    assert result['peso_acumulado'] == 60.0
    assert result['qtd_itens_coletados'] == 3
    assert result['valor_bruto_itens_g'] == 600.0

def test_ttp2_degradation(ttp_instance):
    solucao = SolucaoTTP(
        rota=[1, 2, 3],
        plano_coleta={2: [1], 3: [2]}
    )
    
    result = evaluator_ttp2.evaluate(ttp_instance, solucao)
    
    # 1 -> 2: weight = 0, speed = 1.0, time = 3.0
    # 2 -> 3: collect item 1, weight = 10, speed = 0.91, time = 5.4945...
    #         Item 1 collected at time = 3.0
    # 3 -> 1: collect item 2, weight = 30, speed = 0.73, time = 5.4794...
    #         Item 2 collected at time = 3.0 + 5.4945 = 8.4945...
    # Total time = 3.0 + 5.4945 + 5.4794 = 13.9739...
    
    # Degradation:
    # Item 1: time in knapsack = 13.9739 - 3.0 = 10.9739
    #         C = 1.0, ceil(10.9739 / 1.0) = 11
    #         Degraded value = 100 - 11 = 89
    # Item 2: time in knapsack = 13.9739 - 8.4945 = 5.4794
    #         C = 1.0, ceil(5.4794 / 1.0) = 6
    #         Degraded value = 200 - 6 = 194
    # Total gross value = 89 + 194 = 283
    
    expected_travel_time = 3.0 + (5.0 / 0.91) + (4.0 / 0.73)
    item1_degrad_time = expected_travel_time - 3.0
    item2_degrad_time = expected_travel_time - (3.0 + (5.0 / 0.91))
    
    expected_val = (100 - math.ceil(item1_degrad_time)) + (200 - math.ceil(item2_degrad_time))
    
    assert math.isclose(result['tempo_viagem_f'], expected_travel_time)
    assert result['valor_bruto_itens_g'] == expected_val
    assert result['lucro_liquido_G'] == expected_val  # TTP2 no rent cost in G
    
def test_ttp2_vs_ttp1_comparison(ttp_instance):
    solucao = SolucaoTTP(
        rota=[1, 2, 3],
        plano_coleta={2: [1], 3: [2]}
    )
    
    result1 = evaluator_ttp1.evaluate(ttp_instance, solucao)
    result2 = evaluator_ttp2.evaluate(ttp_instance, solucao)
    
    assert result1['tempo_viagem_f'] == result2['tempo_viagem_f']
    assert result1['peso_acumulado'] == result2['peso_acumulado']
    
    # TTP2 items should have degraded value
    assert result2['valor_bruto_itens_g'] < result1['valor_bruto_itens_g']
