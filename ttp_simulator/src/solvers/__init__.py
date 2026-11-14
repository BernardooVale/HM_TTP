import importlib
import pkgutil
from pathlib import Path

from .base import BaseSolver, SolverRegistry
from .constructive import (
    FaulknerS1Solver,
    GreedyIsolatedSolver,
    Heuristic37Solver,
    QueimadaSolver,
    RandomSolver,
    S2HeuristicSolver,
    S5HeuristicSolver,
)
from .iterative import (
    AlternateLocalSearchSolver,
    CTTP2OptSolver,
    CTTPPGCHSolver,
    CoCoSolver,
    CoSolver,
    Cosolver2BBestFitSolver,
    Cosolver2BFirstFitSolver,
    FaulknerC1Solver,
    FaulknerC2Solver,
    FaulknerC3Solver,
    FaulknerC4Solver,
    FaulknerC5Solver,
    FaulknerC6Solver,
    FaulknerS2Solver,
    FaulknerS3Solver,
    FaulknerS4Solver,
    FaulknerS5Solver,
    PackIterativeSolver,
    refinar_plano_coleta,
    resolver_pack_iterative,
)

# Auto-descoberta de quaisquer novos solucionadores adicionados na pasta src/solvers e subpastas
_pasta_solvers = Path(__file__).parent
_ignorar = {
    "base",
    "__init__",
    "tsp",
    "packing",
    "operators",
    "templates",
    "constructive",
    "iterative",
}

# 1. Scaneia solucionadores adicionais na raiz de solvers/
for _mod_info in pkgutil.iter_modules([str(_pasta_solvers)]):
    if not _mod_info.ispkg and _mod_info.name not in _ignorar:
        try:
            importlib.import_module(f".{_mod_info.name}", package=__name__)
        except Exception:
            pass

# 2. Scaneia solucionadores em solvers/constructive/
_pasta_constructive = _pasta_solvers / "constructive"
if _pasta_constructive.exists():
    for _mod_info in pkgutil.iter_modules([str(_pasta_constructive)]):
        if not _mod_info.ispkg and _mod_info.name not in ("__init__",):
            try:
                importlib.import_module(f".constructive.{_mod_info.name}", package=__name__)
            except Exception:
                pass

# 3. Scaneia solucionadores em solvers/iterative/
_pasta_iterative = _pasta_solvers / "iterative"
if _pasta_iterative.exists():
    for _mod_info in pkgutil.iter_modules([str(_pasta_iterative)]):
        if not _mod_info.ispkg and _mod_info.name not in ("__init__",):
            try:
                importlib.import_module(f".iterative.{_mod_info.name}", package=__name__)
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
    "CoCoSolver",
    "CTTP2OptSolver",
    "CTTPPGCHSolver",
    "Cosolver2BFirstFitSolver",
    "Cosolver2BBestFitSolver",
    "S2HeuristicSolver",
    "S5HeuristicSolver",
    "refinar_plano_coleta",
    "resolver_pack_iterative",
]
