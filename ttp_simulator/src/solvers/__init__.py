from .base import BaseSolver, SolverRegistry
from .baselines import RandomSolver, GreedyIsolatedSolver
from .local_search import AlternateLocalSearchSolver
from .heuristic_37 import Heuristic37Solver
from .queimada import QueimadaSolver

__all__ = [
    "BaseSolver",
    "SolverRegistry",
    "RandomSolver",
    "GreedyIsolatedSolver",
    "AlternateLocalSearchSolver",
    "Heuristic37Solver",
    "QueimadaSolver"
]

