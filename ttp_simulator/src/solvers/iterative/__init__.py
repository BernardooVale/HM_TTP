"""
Módulo de Heurísticas Iterativas e Busca Local para o Traveling Thief Problem (TTP).

Reúne solucionadores e operadores baseados em melhoria iterativa e busca local:
- CoSolver: Co-otimização alternada cooperativa entre tour e packing (Faulkner et al., 2015).
- AlternateLocalSearchSolver: Busca local alternada com operadores de perturbação.
- PackIterativeSolver: Otimização iterativa da mochila via 1-bit-flip / Drop & Add (Faulkner et al., 2015).
- refinar_plano_coleta, resolver_pack_iterative: Operadores centrais de refinamento.
"""

from .cosolver import CoSolver
from .local_search import AlternateLocalSearchSolver
from .pack_iterative import (
    PackIterativeSolver,
    refinar_plano_coleta,
    resolver_pack_iterative,
)

__all__ = [
    "CoSolver",
    "AlternateLocalSearchSolver",
    "PackIterativeSolver",
    "refinar_plano_coleta",
    "resolver_pack_iterative",
]
