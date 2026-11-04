"""
Módulo de Heurísticas Construtivas para o Traveling Thief Problem (TTP).

Reúne solucionadores que constroem soluções completas (rotas e planos de coleta)
de forma direta e estruturada:
- Baselines elementares: RandomSolver, GreedyIsolatedSolver
- Heurística da Secretária (37%): Heuristic37Solver
- Heurística da Queimada (propagações térmicas): QueimadaSolver
- Heurísticas da literatura: S2HeuristicSolver, S5HeuristicSolver
"""

from .baselines import GreedyIsolatedSolver, RandomSolver
from .heuristic_37 import Heuristic37Solver
from .queimada import (
    QueimadaSolver,
    otimizar_trechos_vazios,
    resolver_knapsack_01_exato,
)
from .s2_s5_solvers import S2HeuristicSolver, S5HeuristicSolver

__all__ = [
    "RandomSolver",
    "GreedyIsolatedSolver",
    "Heuristic37Solver",
    "QueimadaSolver",
    "S2HeuristicSolver",
    "S5HeuristicSolver",
    "otimizar_trechos_vazios",
    "resolver_knapsack_01_exato",
]
