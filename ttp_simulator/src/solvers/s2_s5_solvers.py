"""
Solucionadores Heurísticos S2 e S5 para o Traveling Thief Problem (TTP).

Referência bibliográfica:
Polyakovskiy, S., Bonyadi, M. R., Wagner, M., Michalewicz, Z., & Neumann, F. (2014).
"A Comprehensive Benchmark Set and Heuristics for the Traveling Thief Problem".
In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO 2014), pp. 977-984.
ACM.
"""

import math
from typing import Dict, List, Optional
from src.core.models import TTPInstance, SolucaoTTP, Item
from src.solvers.base import BaseSolver, SolverRegistry
from src.solvers.heuristic_37 import obter_melhor_rota_tsp
from src.solvers.pack_iterative import refinar_plano_coleta


def _calcular_distancias_restantes(instance: TTPInstance, rota: List[int]) -> tuple[Dict[int, float], float]:
    """Calcula a distância restante de cada cidade até o retorno à cidade 1 e o comprimento total da rota."""
    n = len(rota)
    dist_restante: Dict[int, float] = {}
    dist_acc = instance.grafo.distancia(rota[-1], rota[0])
    dist_restante[rota[-1]] = dist_acc

    for i in range(n - 2, -1, -1):
        dist_acc += instance.grafo.distancia(rota[i], rota[i + 1])
        dist_restante[rota[i]] = dist_acc

    total_dist = dist_acc
    return dist_restante, total_dist


@SolverRegistry.register
class S2HeuristicSolver(BaseSolver):
    """
    Heurística S2 (Score 2 / Distance-to-End Density) para o TTP (Polyakovskiy et al., 2014).
    
    Escore de cada item:
        S2(i) = v_i / (w_i * d_i_restante)
    onde d_i_restante é a distância percorrida da cidade do item até a cidade 1.
    """

    def __init__(self, refinamento: bool = False, nome: Optional[str] = None):
        self.refinamento = refinamento
        self._nome = nome or ("s2_refinada" if refinamento else "s2_heuristic")

    @property
    def nome(self) -> str:
        return self._nome

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        rota = obter_melhor_rota_tsp(instance.grafo)
        dist_restante, _ = _calcular_distancias_restantes(instance, rota)

        # Ordenação decrescente pelo escore S2
        def escore_s2(item: Item) -> float:
            if item.cidade_origem == 1:
                return -1.0
            if item.peso <= 0:
                return float("inf")
            d_rest = max(dist_restante.get(item.cidade_origem, 1.0), 0.001)
            return item.valor / (item.peso * d_rest)

        itens_ordenados = sorted(instance.itens, key=escore_s2, reverse=True)

        peso_atual = 0.0
        plano_coleta: Dict[int, List[int]] = {}

        for item in itens_ordenados:
            if item.cidade_origem == 1:
                continue
            if peso_atual + item.peso <= instance.capacidade_mochila:
                peso_atual += item.peso
                if item.cidade_origem not in plano_coleta:
                    plano_coleta[item.cidade_origem] = []
                plano_coleta[item.cidade_origem].append(item.id)

        if self.refinamento:
            plano_coleta = refinar_plano_coleta(instance, rota, plano_coleta)

        return SolucaoTTP(rota=rota, plano_coleta=plano_coleta)


@SolverRegistry.register
class S5HeuristicSolver(BaseSolver):
    """
    Heurística S5 (Score 5 / Exponential Distance Decay) para o TTP (Polyakovskiy et al., 2014).
    
    Escore de cada item:
        S5(i) = (v_i / w_i) * exp(-d_i_restante / D_total)
    """

    def __init__(self, refinamento: bool = False, nome: Optional[str] = None):
        self.refinamento = refinamento
        self._nome = nome or ("s5_refinada" if refinamento else "s5_heuristic")

    @property
    def nome(self) -> str:
        return self._nome

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        rota = obter_melhor_rota_tsp(instance.grafo)
        dist_restante, d_total = _calcular_distancias_restantes(instance, rota)
        d_total_norm = max(d_total, 1.0)

        # Ordenação decrescente pelo escore S5
        def escore_s5(item: Item) -> float:
            if item.cidade_origem == 1:
                return -1.0
            if item.peso <= 0:
                return float("inf")
            d_rest = max(dist_restante.get(item.cidade_origem, 0.0), 0.0)
            densidade = item.valor / item.peso
            decaimento = math.exp(-d_rest / d_total_norm)
            return densidade * decaimento

        itens_ordenados = sorted(instance.itens, key=escore_s5, reverse=True)

        peso_atual = 0.0
        plano_coleta: Dict[int, List[int]] = {}

        for item in itens_ordenados:
            if item.cidade_origem == 1:
                continue
            if peso_atual + item.peso <= instance.capacidade_mochila:
                peso_atual += item.peso
                if item.cidade_origem not in plano_coleta:
                    plano_coleta[item.cidade_origem] = []
                plano_coleta[item.cidade_origem].append(item.id)

        if self.refinamento:
            plano_coleta = refinar_plano_coleta(instance, rota, plano_coleta)

        return SolucaoTTP(rota=rota, plano_coleta=plano_coleta)


# Variações registradas para comparação com refinamento ativado
SolverRegistry.register(
    type(
        "S2RefinadaSolver",
        (S2HeuristicSolver,),
        {"nome": "s2_refinada", "__init__": lambda self: S2HeuristicSolver.__init__(self, refinamento=True, nome="s2_refinada")},
    )
)
SolverRegistry.register(
    type(
        "S5RefinadaSolver",
        (S5HeuristicSolver,),
        {"nome": "s5_refinada", "__init__": lambda self: S5HeuristicSolver.__init__(self, refinamento=True, nome="s5_refinada")},
    )
)
