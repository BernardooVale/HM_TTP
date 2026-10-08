import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import os
import tempfile
from src.scenarios.seed_manager import set_global_seed, get_random_seed
from src.scenarios.graph_generator import (
    gerar_grafo_normal,
    gerar_grafo_esparso,
    gerar_grafo_equidistante,
    gerar_grafo_concentrado
)
from src.scenarios.item_generator import gerar_itens, calcular_capacidade_mochila
from src.scenarios.exporter import exportar_cenario
from src.scenarios.parser import InstanceParser
from src.core.models import TTPInstance

def test_seed_manager():
    set_global_seed(1234)
    seed1 = get_random_seed()
    set_global_seed(1234)
    seed2 = get_random_seed()
    assert seed1 == seed2

def test_graph_generators():
    g_normal = gerar_grafo_normal(n_cidades=5, seed=42)
    assert len(g_normal.cidades) == 5
    assert g_normal.edge_weight_type == 'EUC_2D'
    
    g_esparso = gerar_grafo_esparso(n_cidades=5, seed=42)
    assert len(g_esparso.cidades) == 5
    assert g_esparso.edge_weight_type == 'EXPLICIT'
    
    g_equi = gerar_grafo_equidistante(n_cidades=6, seed=42)
    assert len(g_equi.cidades) == 6
    
    g_conc = gerar_grafo_concentrado(n_cidades=8, seed=42, n_clusters=2)
    assert len(g_conc.cidades) == 8

def test_item_generator_and_capacity():
    g = gerar_grafo_normal(n_cidades=4, seed=42)
    itens = gerar_itens(n_itens=6, cidades=g.cidades, seed=42, distribuicao='uniforme')
    assert len(itens) == 6
    
    cap = calcular_capacidade_mochila(itens, tightness_ratio=0.5)
    peso_total = sum(i.peso for i in itens)
    assert cap == pytest.approx(0.5 * peso_total)

def test_exportar_cenario_single_and_parse():
    g = gerar_grafo_normal(n_cidades=4, seed=42)
    itens = gerar_itens(n_itens=4, cidades=g.cidades, seed=42)
    cap = calcular_capacidade_mochila(itens, tightness_ratio=0.6)
    
    inst = TTPInstance(
        nome='TestExport',
        grafo=g,
        itens=itens,
        capacidade_mochila=cap,
        v_min=0.1,
        v_max=1.0,
        taxa_aluguel=1.5,
        constante_degradacao=10.0,
        dimension=4,
        num_items=4
    )
    
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tf:
        temp_path = tf.name
        
    try:
        exportar_cenario(inst, temp_path, formato='unico')
        assert os.path.exists(temp_path)
        
        parsed = InstanceParser.carregar(temp_path)
        assert parsed.nome == 'TestExport'
        assert len(parsed.grafo.cidades) == 4
        assert len(parsed.itens) == 4
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
