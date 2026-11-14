"""
Heurística construtiva S1 de Faulkner et al. (2015) para o TTP1.
"""

from src.core.models import SolucaoTTP, TTPInstance
from src.solvers.base import SolverRegistry
from src.solvers.operators.faulkner_base import FaulknerBaseSolver
from src.solvers.operators.faulkner_operators import pack_iterative


@SolverRegistry.register
class FaulknerS1Solver(FaulknerBaseSolver):
    """S1: Construtiva rápida em dois estágios (CLK -> PackIterative)."""

    @property
    def nome(self) -> str:
        return "faulkner_s1"

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        rota = self._gerar_rota_clk(instance, seed=self.seed)
        plano, _ = pack_iterative(
            instance, rota, c=self.c, delta=self.delta, q=self.q
        )
        return SolucaoTTP(rota=rota, plano_coleta=plano)
