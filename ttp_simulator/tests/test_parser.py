import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import tempfile
import os
from src.scenarios.parser import InstanceParser

@pytest.fixture
def temp_instance_file():
    content = """PROBLEM NAME: test_instance
KNAPSACK DATA TYPE: UNCORRELATED
DIMENSION: 3
NUMBER OF ITEMS: 2
CAPACITY OF KNAPSACK: 100
MIN SPEED: 0.1
MAX SPEED: 1.0
RENTING RATIO: 1.5
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0 0
2 3 0
3 0 4
ITEMS_SECTION
1 50 10 2
2 80 20 3
"""
    fd, path = tempfile.mkstemp()
    with open(fd, 'w', encoding='utf-8') as f:
        f.write(content)
    yield path
    os.remove(path)

@pytest.fixture
def temp_explicit_instance_file():
    content = """PROBLEM NAME: explicit_test
DIMENSION: 3
NUMBER OF ITEMS: 1
CAPACITY OF KNAPSACK: 50
MIN SPEED: 0.1
MAX SPEED: 1.0
RENTING RATIO: 1.0
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_SECTION
0 10 20
10 0 15
20 15 0
ITEMS_SECTION
1 100 5 2
"""
    fd, path = tempfile.mkstemp()
    with open(fd, 'w', encoding='utf-8') as f:
        f.write(content)
    yield path
    os.remove(path)

def test_parse_euc_2d_file(temp_instance_file):
    parser = InstanceParser()
    instance = parser.parse(temp_instance_file)
    
    assert instance.nome == 'test_instance'
    assert instance.dimension == 3
    assert instance.num_items == 2
    assert instance.capacidade_mochila == 100.0
    assert instance.v_min == 0.1
    assert instance.v_max == 1.0
    assert instance.taxa_aluguel == 1.5
    
    assert len(instance.grafo.cidades) == 3
    assert instance.grafo.edge_weight_type == 'EUC_2D'
    assert instance.grafo.cidades[1].x == 0.0
    assert instance.grafo.cidades[2].x == 3.0
    
    assert len(instance.itens) == 2
    assert instance.itens[0].id == 1
    assert instance.itens[0].valor == 50.0
    assert instance.itens[0].peso == 10.0
    assert instance.itens[0].cidade_origem == 2
    
    assert len(instance.grafo.cidades[2].itens) == 1
    assert instance.grafo.cidades[2].itens[0].id == 1

def test_parse_explicit_file(temp_explicit_instance_file):
    parser = InstanceParser()
    instance = parser.parse(temp_explicit_instance_file)
    
    assert instance.grafo.edge_weight_type == 'EXPLICIT'
    assert len(instance.grafo.cidades) == 3
    assert instance.grafo.distancia(1, 2) == 10.0
    assert instance.grafo.distancia(2, 3) == 15.0
    assert instance.grafo.distancia(1, 3) == 20.0
    
def test_assign_items_to_cities(temp_instance_file):
    parser = InstanceParser()
    instance = parser.parse(temp_instance_file)
    
    assert len(instance.grafo.cidades[1].itens) == 0
    assert len(instance.grafo.cidades[2].itens) == 1
    assert len(instance.grafo.cidades[3].itens) == 1
