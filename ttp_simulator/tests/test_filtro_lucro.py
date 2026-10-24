import math
import sys
from pathlib import Path
import pytest

_raiz = Path(__file__).resolve().parents[2]
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))

from src.core.models import Cidade, Grafo, Item, SolucaoTTP, TTPInstance
from src.core.evaluator_ttp1 import avaliar_ttp1
from scripts.instancias.filtrar_instancias_lucro import (
    calcular_mochila_divisivel,
    calcular_mst_prim,
    analisar_instancia_lucro,
)


def criar_instancia_teste_pequena():
    """Cria uma instância sintética conhecida para teste."""
    cidades = {
        1: Cidade(id=1, x=0.0, y=0.0),
        2: Cidade(id=2, x=0.0, y=10.0),
        3: Cidade(id=3, x=10.0, y=10.0),
        4: Cidade(id=4, x=10.0, y=0.0),
    }
    grafo = Grafo(cidades=cidades, edge_weight_type="EUC_2D")
    
    itens = [
        Item(id=1, peso=10.0, valor=100.0, cidade_origem=2),
        Item(id=2, peso=20.0, valor=150.0, cidade_origem=2),
        Item(id=3, peso=15.0, valor=90.0, cidade_origem=3),
        Item(id=4, peso=5.0, valor=80.0, cidade_origem=4),
    ]
    for it in itens:
        cidades[it.cidade_origem].itens.append(it)
        
    return TTPInstance(
        nome="TestePequeno",
        grafo=grafo,
        itens=itens,
        capacidade_mochila=30.0,
        v_min=0.1,
        v_max=1.0,
        taxa_aluguel=2.0,
        constante_degradacao=1.0,
    )


def test_mochila_divisivel_otima():
    """Verifica se a mochila fracionária obtém o limitante superior exato."""
    itens = [
        Item(id=1, peso=10.0, valor=100.0, cidade_origem=1),  # ratio 10.0
        Item(id=2, peso=20.0, valor=150.0, cidade_origem=1),  # ratio 7.5
        Item(id=3, peso=15.0, valor=90.0, cidade_origem=1),   # ratio 6.0
    ]
    # Capacidade 25: item 1 inteiro (peso 10, val 100) + 15/20 do item 2 (peso 15, val 112.5) -> Total: 212.5
    valor, peso = calcular_mochila_divisivel(itens, 25.0)
    assert peso == pytest.approx(25.0)
    assert valor == pytest.approx(212.5)


def test_mst_prim_quadrado():
    """Verifica o custo da MST em um quadrado 10x10 (arestas mínimas = 10 + 10 + 10 = 30)."""
    cidades = {
        1: Cidade(id=1, x=0.0, y=0.0),
        2: Cidade(id=2, x=0.0, y=10.0),
        3: Cidade(id=3, x=10.0, y=10.0),
        4: Cidade(id=4, x=10.0, y=0.0),
    }
    grafo = Grafo(cidades=cidades, edge_weight_type="EUC_2D")
    mst_cost = calcular_mst_prim(grafo)
    assert mst_cost == pytest.approx(30.0)


def test_relaxacao_upper_bound_garantida():
    """
    Teorema Fundamental do Filtro:
    Para QUALQUER solução admissível (rota e picking plan),
    o lucro líquido G obtido deve ser estritamente menor ou igual ao Upper Bound UB.
    """
    inst = criar_instancia_teste_pequena()
    res_analise = analisar_instancia_lucro(inst)
    ub_calculado = res_analise["ub_lucro_liquido"]

    # Testa com várias soluções válidas
    solucoes = [
        # Sem itens coletados
        SolucaoTTP(rota=[1, 2, 3, 4], plano_coleta={}),
        # Coletando item 4 (peso 5)
        SolucaoTTP(rota=[1, 2, 3, 4], plano_coleta={4: [4]}),
        # Coletando item 1 (peso 10) e item 4 (peso 5)
        SolucaoTTP(rota=[1, 2, 3, 4], plano_coleta={2: [1], 4: [4]}),
        # Rota invertida
        SolucaoTTP(rota=[1, 4, 3, 2], plano_coleta={2: [1], 4: [4]}),
    ]

    for sol in solucoes:
        aval = avaliar_ttp1(inst, sol)
        lucro_real = aval["lucro_liquido_G"]
        # UB deve ser cota superior estrita (ou igual)
        assert ub_calculado >= lucro_real, f"Violou limitante: UB={ub_calculado} < Real={lucro_real}"
