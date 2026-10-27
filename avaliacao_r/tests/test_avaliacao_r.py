"""
Testes unitários e de integração para o pacote avaliacao_r.
"""

import pytest
import math
import tempfile
import shutil
from pathlib import Path

import sys
_raiz = Path(__file__).resolve().parent.parent.parent
_sim_path = _raiz / "ttp_simulator"
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))
if str(_sim_path) not in sys.path:
    sys.path.insert(0, str(_sim_path))

from src.core.models import Item, Cidade, Grafo, TTPInstance, SolucaoTTP
from avaliacao_r.config import AvaliacaoRConfig
from avaliacao_r.exact_solvers import (
    resolver_kp_exato,
    resolver_tsp_exato,
    resolver_rota_exata_itens_fixos,
    resolver_packing_exato_rota_fixa,
    resolver_ttp_exato_bb
)
from avaliacao_r.heuristic_solvers import (
    resolver_kp_guloso,
    resolver_tsp_guloso
)
from avaliacao_r.evaluator import avaliar_instancia_ttp
from avaliacao_r.relatorios import (
    obter_instancias_ja_calculadas,
    salvar_todos_resultados
)


@pytest.fixture
def instancia_teste():
    """Cria uma instância TTP sintética pequena e controlada (4 cidades, 4 itens)."""
    # 4 cidades em quadrado: (0,0), (0,10), (10,10), (10,0)
    cidades = {
        1: Cidade(id=1, x=0.0, y=0.0),
        2: Cidade(id=2, x=0.0, y=10.0),
        3: Cidade(id=3, x=10.0, y=10.0),
        4: Cidade(id=4, x=10.0, y=0.0),
    }
    grafo = Grafo(cidades=cidades, edge_weight_type='EUC_2D')
    
    # 4 itens distribuídos
    itens = [
        Item(id=1, valor=100.0, peso=20.0, cidade_origem=2),
        Item(id=2, valor=60.0, peso=15.0, cidade_origem=3),
        Item(id=3, valor=80.0, peso=10.0, cidade_origem=4),
        Item(id=4, valor=120.0, peso=25.0, cidade_origem=4),
    ]
    for it in itens:
        cidades[it.cidade_origem].itens.append(it)
        
    return TTPInstance(
        nome="instancia_mock_4c",
        grafo=grafo,
        itens=itens,
        capacidade_mochila=45.0,
        v_min=0.1,
        v_max=1.0,
        taxa_aluguel=1.0,
        constante_degradacao=1.0,
        dimension=4,
        num_items=4
    )


def test_kp_exato_encontra_otimo():
    """Verifica se o solver exato do Knapsack 0-1 supera ou iguala o guloso."""
    # Itens montados onde o guloso falha:
    # Item 1: peso 30, valor 31 (ratio 1.033)
    # Item 2: peso 20, valor 20 (ratio 1.000)
    # Item 3: peso 20, valor 20 (ratio 1.000)
    # Capacidade: 40.
    # Guloso pega item 1 (lucro 31). Ótimo pega itens 2 e 3 (lucro 40).
    itens = [
        Item(id=1, valor=31.0, peso=30.0, cidade_origem=2),
        Item(id=2, valor=20.0, peso=20.0, cidade_origem=3),
        Item(id=3, valor=20.0, peso=20.0, cidade_origem=4),
    ]
    lucro_gul, peso_gul, ids_gul = resolver_kp_guloso(itens, capacidade=40.0)
    assert lucro_gul == 31.0
    assert ids_gul == [1]

    lucro_opt, peso_opt, ids_opt = resolver_kp_exato(itens, capacidade=40.0)
    assert lucro_opt == 40.0
    assert sorted(ids_opt) == [2, 3]


def test_tsp_exato(instancia_teste):
    """Verifica se o TSP exato fecha o ciclo no quadrado com perímetro 40."""
    dist_opt, rota_opt = resolver_tsp_exato(instancia_teste.grafo)
    # No quadrado (0,0)->(0,10)->(10,10)->(10,0)->(0,0), a distância mínima é 40.0
    assert math.isclose(dist_opt, 40.0, rel_tol=1e-5)
    assert rota_opt[0] == 1
    assert len(rota_opt) == 4


def test_rota_exata_itens_fixos(instancia_teste):
    """Verifica que a rota ótima condicionada a itens calcula tempo coerente."""
    itens_sel = [instancia_teste.itens[2], instancia_teste.itens[3]] # itens na cidade 4
    tempo, rota = resolver_rota_exata_itens_fixos(
        instancia_teste.grafo, 
        itens_sel, 
        instancia_teste.capacidade_mochila,
        instancia_teste.v_min,
        instancia_teste.v_max
    )
    assert tempo > 0.0
    assert rota[0] == 1
    assert len(rota) == 4


def test_packing_exato_rota_fixa_sensibilidade_r(instancia_teste):
    """Verifica que o número de itens ou o lucro líquido diminui com o aumento de R."""
    rota = [1, 2, 3, 4]
    
    # Com R = 0 (sem penalidade de tempo), deve coletar o máximo lucro possível
    z_r0, plano_r0 = resolver_packing_exato_rota_fixa(instancia_teste, rota, taxa_r=0.0)
    
    # Com R = 100.0 (penalidade gigantesca), deve preferir coletar menos para não lentificar
    z_r100, plano_r100 = resolver_packing_exato_rota_fixa(instancia_teste, rota, taxa_r=100.0)
    
    itens_r0 = sum(len(ids) for ids in plano_r0.values())
    itens_r100 = sum(len(ids) for ids in plano_r100.values())
    
    assert z_r0 > z_r100
    assert itens_r0 >= itens_r100


def test_ttp_exato_bb(instancia_teste):
    """Verifica que o Branch & Bound do TTP encontra uma solução válida."""
    z_opt, sol_opt, provado = resolver_ttp_exato_bb(instancia_teste, taxa_r=1.0, timeout_s=10.0)
    assert provado is True
    assert sol_opt is not None
    assert len(sol_opt.rota) == 4


def test_avaliacao_instancia_e_persistencia(instancia_teste):
    """Testa o fluxo completo de avaliação de uma instância e persistência atômica."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AvaliacaoRConfig(
            diretorio_resultados=Path(tmpdir) / "resultados",
            fatores_r=[0.0, 0.5, 1.0, 2.0, 10.0],
            max_cidades_exato_ttp=5
        )
        
        resultado = avaliar_instancia_ttp(instancia_teste, config)
        
        assert resultado.instancia_nome == "instancia_mock_4c"
        assert len(resultado.pontos_r) == 5
        assert resultado.r_crossover >= 0.0
        
        # Teste de persistência segura
        salvar_todos_resultados(resultado, config)
        
        # Verifica se o arquivo JSON foi criado
        json_path = config.diretorio_resultados / "detalhes" / "instancia_mock_4c.json"
        assert json_path.exists()
        
        # Verifica se os CSVs foram gerados
        csv_geral = config.diretorio_resultados / "relatorio_geral_r.csv"
        csv_curvas = config.diretorio_resultados / "curvas_r_detalhadas.csv"
        assert csv_geral.exists()
        assert csv_curvas.exists()
        
        # Testa a detecção de retomada (resume)
        calculadas = obter_instancias_ja_calculadas(config)
        assert "instancia_mock_4c" in calculadas
