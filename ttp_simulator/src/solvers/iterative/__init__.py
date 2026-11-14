"""
Módulo de Heurísticas Iterativas e Busca Local para o Traveling Thief Problem (TTP).

Reúne solucionadores e operadores baseados em melhoria iterativa e busca local:
- CoSolver: Co-otimização alternada cooperativa entre tour e packing (Faulkner et al., 2015).
- AlternateLocalSearchSolver: Busca local alternada com operadores de perturbação.
- PackIterativeSolver: Otimização iterativa da mochila via 1-bit-flip / Drop & Add (Faulkner et al., 2015).
- refinar_plano_coleta, resolver_pack_iterative: Operadores centrais de refinamento.
"""

from .cosolver import CoSolver
from .faulkner_solvers import (
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
)
from .local_search import AlternateLocalSearchSolver
from .pack_iterative import (
    PackIterativeSolver,
    refinar_plano_coleta,
    resolver_pack_iterative,
)
from ..operators import (
    bit_flip,
    insertion_local_search,
    one_plus_one_ea,
    pack,
    pack_iterative,
)

__all__ = [
    "CoSolver",
    "AlternateLocalSearchSolver",
    "PackIterativeSolver",
    "refinar_plano_coleta",
    "resolver_pack_iterative",
    "pack",
    "pack_iterative",
    "bit_flip",
    "one_plus_one_ea",
    "insertion_local_search",
    "FaulknerS2Solver",
    "FaulknerS3Solver",
    "FaulknerS4Solver",
    "FaulknerS5Solver",
    "FaulknerC1Solver",
    "FaulknerC2Solver",
    "FaulknerC3Solver",
    "FaulknerC4Solver",
    "FaulknerC5Solver",
    "FaulknerC6Solver",
]
