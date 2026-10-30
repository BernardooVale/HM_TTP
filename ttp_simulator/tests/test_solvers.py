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
    assert "queimada" in solvers

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

@pytest.mark.parametrize("solver_name", [
    "random", "greedy", "heuristic_37", "heuristic_37_refinada", "heuristic_37_simetrica",
    "heuristic_37_ttp2", "local_search", "queimada", "queimada_adj", "queimada_bfs",
    "queimada_inicio", "queimada_kp", "queimada_exact", "queimada_kruskal",
    "queimada_reversa", "queimada_refinada", "queimada_reversa_refinada", "queimada_legacy",
    "queimada_com_2opt", "queimada_sem_2opt", "queimada_com_afinidade", "queimada_sem_afinidade",
    "queimada_regime_r", "queimada_ttp2", "queimada_backward_arestas", "queimada_backward_sem_2opt",
    "queimada_backward_direcionada", "queimada_backward_direcionada_buracos", "queimada_backward_direcionada_sem_2opt_buracos",
    "queimada_forward_direcionada", "queimada_forward_direcionada_buracos", "queimada_forward_direcionada_sem_2opt_buracos",
    "pack_iterative", "cosolver",
    "s2_heuristic", "s2_refinada", "s5_heuristic", "s5_refinada"
])
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

@pytest.mark.parametrize("temp", ["total_value", "knapsack_exact", "knapsack_greedy"])
@pytest.mark.parametrize("diss", ["adjacent_only", "dijkstra", "bfs"])
@pytest.mark.parametrize("direcao", ["calor_final", "calor_inicio"])
@pytest.mark.parametrize("selecao", ["greedy_relative", "knapsack_relative"])
def test_queimada_all_variations(temp, diss, direcao, selecao, sample_instance):
    from src.solvers.queimada import QueimadaSolver
    solver = QueimadaSolver(
        temperatura=temp,
        dissipacao=diss,
        direcao=direcao,
        selecao=selecao
    )
    solucao = solver.solve(sample_instance)
    assert len(solucao.rota) == 4
    assert solucao.rota[0] == 1
    assert set(solucao.rota) == {1, 2, 3, 4}
    peso_total = sum(
        sample_instance.itens[it_id - 1].peso
        for it_ids in solucao.plano_coleta.values()
        for it_id in it_ids
    )
    assert peso_total <= sample_instance.capacidade_mochila


def test_queimada_otimizar_buracos(sample_instance):
    from src.solvers.queimada import QueimadaSolver, otimizar_trechos_vazios
    
    # Rota intencionalmente com cruzamento em cidades vazias
    rota = [1, 4, 3, 2] # Cidades 3 e 4 vazias
    plano = {2: [1]} # Apenas cidade 2 tem coleta
    
    rota_otimizada = otimizar_trechos_vazios(sample_instance, rota, plano)
    assert len(rota_otimizada) == 4
    assert rota_otimizada[0] == 1
    assert set(rota_otimizada) == {1, 2, 3, 4}
    
    solver = QueimadaSolver(otimizar_buracos=True)
    sol = solver.solve(sample_instance)
    assert len(sol.rota) == 4
    assert sol.rota[0] == 1
    assert "queimada_com_buracos" in SolverRegistry.listar()
    assert "queimada_backward_buracos" in SolverRegistry.listar()
    assert "queimada_backward_sem_2opt_buracos" in SolverRegistry.listar()


def test_queimada_directed_variations(sample_instance):
    for solver_name in [
        "queimada_backward_direcionada",
        "queimada_backward_direcionada_buracos",
        "queimada_backward_direcionada_sem_2opt_buracos",
        "queimada_forward_direcionada",
        "queimada_forward_direcionada_buracos",
        "queimada_forward_direcionada_sem_2opt_buracos",
    ]:
        solver = SolverRegistry.get(solver_name)
        sol = solver.solve(sample_instance)
        assert len(sol.rota) == 4
        assert sol.rota[0] == 1
        assert set(sol.rota) == {1, 2, 3, 4}
        peso_total = sum(
            sample_instance.itens[it_id - 1].peso
            for it_ids in sol.plano_coleta.values()
            for it_id in it_ids
        )
        assert peso_total <= sample_instance.capacidade_mochila


def test_adaptive_ram_cache_and_numba_equivalence(sample_instance):
    from src.solvers.numba_kernels import (
        AdaptiveRAMCache,
        detectar_memoria_sistema_bytes,
        obter_cache_ram_global,
    )
    from src.solvers.queimada import QueimadaSolver

    total_b, avail_b = detectar_memoria_sistema_bytes()
    assert total_b > 0
    assert avail_b > 0

    # Testa cache com limite estrito de memória e eviction LRU automático
    cache = AdaptiveRAMCache(num_workers=4, min_budget_mb=0.002, max_budget_mb=0.005)
    for i in range(100):
        cache.set(f"chave_{i}", [i] * 20)
    resumo = cache.status_resumo()
    assert resumo["evictions"] > 0
    assert cache._current_bytes <= cache.budget_bytes

    # Verifica hit no cache global durante execuções sucessivas da Queimada
    global_cache = obter_cache_ram_global()
    global_cache.clear()
    hits_antes = global_cache.hits

    solver = QueimadaSolver(
        temperatura="knapsack_exact",
        dissipacao="dijkstra",
        metodo_construcao="kruskal",
        refinamento=True,
        refinamento_rota=True,
        otimizar_buracos=True,
    )
    sol1 = solver.solve(sample_instance)
    sol2 = solver.solve(sample_instance)

    assert sol1.rota == sol2.rota
    assert sol1.plano_coleta == sol2.plano_coleta
    assert global_cache.hits > hits_antes
