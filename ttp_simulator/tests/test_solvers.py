import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import src.solvers  # Trigger registration
from src.solvers.base import BaseSolver, SolverRegistry
from src.core.models import TTPInstance, SolucaoTTP, Grafo, Cidade, Item

class DummySolver(BaseSolver):
    @property
    def nome(self) -> str:
        return "Dummy"
        
    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        rota = list(instance.grafo.cidades.keys())
        return SolucaoTTP(rota=rota, plano_coleta={})

@pytest.fixture(autouse=True)
def setup_teardown():
    if "Dummy" not in SolverRegistry._registry:
        SolverRegistry.register(DummySolver)
    yield

def test_solver_registry_register_and_get():
    solver = SolverRegistry.get("Dummy")
    assert isinstance(solver, DummySolver)

def test_solver_registry_listar():
    solvers = SolverRegistry.listar()
    assert "Dummy" in solvers
    assert "random" in solvers
    assert "greedy" in solvers
    assert "local_search" in solvers
    assert "heuristic_37" in solvers

@pytest.fixture
def sample_instance():
    cidades = {
        1: Cidade(id=1, x=0.0, y=0.0),
        2: Cidade(id=2, x=10.0, y=0.0),
        3: Cidade(id=3, x=10.0, y=10.0),
        4: Cidade(id=4, x=0.0, y=10.0)
    }
    itens = [
        Item(id=1, peso=10.0, valor=50.0, cidade_origem=2),
        Item(id=2, peso=20.0, valor=100.0, cidade_origem=2),
        Item(id=3, peso=15.0, valor=80.0, cidade_origem=3),
        Item(id=4, peso=30.0, valor=120.0, cidade_origem=4),
    ]
    for it in itens:
        cidades[it.cidade_origem].itens.append(it)
        
    grafo = Grafo(cidades=cidades, edge_weight_type='EUC_2D')
    return TTPInstance(
        nome='SampleTest',
        grafo=grafo,
        itens=itens,
        capacidade_mochila=45.0,
        v_min=0.1,
        v_max=1.0,
        taxa_aluguel=1.5,
        constante_degradacao=10.0,
        dimension=4,
        num_items=4
    )

@pytest.mark.parametrize("solver_name", ["random", "greedy", "heuristic_37", "local_search"])
def test_registered_solvers_produce_valid_solution(solver_name, sample_instance):
    solver = SolverRegistry.get(solver_name)
    solucao = solver.solve(sample_instance)
    
    # 1. Rota visita todas as cidades sem duplicatas
    assert len(solucao.rota) == 4
    assert set(solucao.rota) == {1, 2, 3, 4}
    assert len(solucao.rota) == len(set(solucao.rota))
    assert solucao.rota[0] == 1
    
    # 2. Respeita capacidade da mochila
    itens_dict = {item.id: item for item in sample_instance.itens}
    peso_total = 0.0
    for city_id, item_ids in solucao.plano_coleta.items():
        assert city_id in sample_instance.grafo.cidades
        for i_id in item_ids:
            assert i_id in itens_dict
            assert itens_dict[i_id].cidade_origem == city_id
            peso_total += itens_dict[i_id].peso
            
    assert peso_total <= sample_instance.capacidade_mochila
