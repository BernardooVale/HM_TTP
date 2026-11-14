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
from .pgch_solvers import CTTP2OptSolver, CTTPPGCHSolver
from ..operators import (
    bit_flip,
    calcular_prefixo_minimo,
    insertion_local_search,
    one_plus_one_ea,
    pack,
    pack_iterative,
    pgch_move,
    preparar_itens_ordenados_por_cidade,
    rand_flip_one_item,
    two_opt_convencional_move,
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
    "preparar_itens_ordenados_por_cidade",
    "calcular_prefixo_minimo",
    "two_opt_convencional_move",
    "pgch_move",
    "rand_flip_one_item",
    "CTTP2OptSolver",
    "CTTPPGCHSolver",
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
