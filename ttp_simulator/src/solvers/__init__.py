import importlib
import pkgutil
from pathlib import Path

from .base import BaseSolver, SolverRegistry
from .baselines import GreedyIsolatedSolver, RandomSolver
from .cosolver import CoSolver
from .heuristic_37 import Heuristic37Solver
from .local_search import AlternateLocalSearchSolver
from .pack_iterative import PackIterativeSolver, refinar_plano_coleta, resolver_pack_iterative
from .queimada import QueimadaSolver
from .s2_s5_solvers import S2HeuristicSolver, S5HeuristicSolver

# Auto-descoberta de quaisquer novos solucionadores adicionados na pasta src/solvers
_pasta_solvers = Path(__file__).parent
for _mod_info in pkgutil.iter_modules([str(_pasta_solvers)]):
    if not _mod_info.ispkg and _mod_info.name not in ("base", "__init__", "tsp"):
        try:
            importlib.import_module(f".{_mod_info.name}", package=__name__)
        except Exception:
            pass

__all__ = [
    "BaseSolver",
    "SolverRegistry",
    "RandomSolver",
    "GreedyIsolatedSolver",
    "AlternateLocalSearchSolver",
    "Heuristic37Solver",
    "QueimadaSolver",
    "PackIterativeSolver",
    "CoSolver",
    "S2HeuristicSolver",
    "S5HeuristicSolver",
    "refinar_plano_coleta",
    "resolver_pack_iterative",
]
