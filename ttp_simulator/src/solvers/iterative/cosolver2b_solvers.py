"""
Solucionadores Cosolver2B (El Yafrani & Ahiod, 2015) para o TTP1:
- Cosolver2BFirstFitSolver (`cosolver2b_first_fit`)
- Cosolver2BBestFitSolver (`cosolver2b_best_fit`)
"""

from typing import Optional

from src.core.models import SolucaoTTP, TTPInstance
from src.solvers.base import BaseSolver, SolverRegistry
from src.solvers.operators.cosolver2b_operators import (
    bit_flip_cosolver2b,
    gerar_grafo_delaunay,
    indexar_itens_por_cidade,
    insercao_e_eliminacao,
    two_opt_delaunay,
)
from src.solvers.tsp.CLK import ChainedLK


class _Cosolver2BBaseSolver(BaseSolver):
    """
    Loop determinístico do Cosolver2B alternando TSKP (2-OPT com poda por
    Triangulação de Delaunay) e KRP (Bit-Flip estruturado por cidade) até
    convergência local estrita.
    """

    estrategia: str = "best_fit"

    def __init__(
        self,
        seed: Optional[int] = None,
        clk_tries: int = 5,
        clk_kicks: int = 5,
    ):
        self.seed = seed
        self.clk_tries = clk_tries
        self.clk_kicks = clk_kicks

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        clk = ChainedLK(
            grafo=instance.grafo,
            num_tries=self.clk_tries,
            num_kicks=self.clk_kicks,
            seed=self.seed,
        )
        rota = list(clk.solve().order)
        plano, _ = insercao_e_eliminacao(instance, rota)

        grafo_delaunay = gerar_grafo_delaunay(instance.grafo)
        itens_por_cidade = indexar_itens_por_cidade(instance)

        while True:
            rota, _, mel_rota = two_opt_delaunay(
                instance,
                rota,
                plano,
                grafo_delaunay,
                estrategia=self.estrategia,
            )
            plano, _, mel_itens = bit_flip_cosolver2b(
                instance,
                rota,
                plano,
                itens_por_cidade,
                estrategia=self.estrategia,
            )
            if not mel_rota and not mel_itens:
                break

        return SolucaoTTP(rota=rota, plano_coleta=plano)


@SolverRegistry.register
class Cosolver2BFirstFitSolver(_Cosolver2BBaseSolver):
    """Cosolver2B com estratégia de vizinhança First-Fit (First Improvement)."""

    estrategia = "first_fit"

    @property
    def nome(self) -> str:
        return "cosolver2b_first_fit"


@SolverRegistry.register
class Cosolver2BBestFitSolver(_Cosolver2BBaseSolver):
    """Cosolver2B com estratégia de vizinhança Best-Fit (Best Improvement)."""

    estrategia = "best_fit"

    @property
    def nome(self) -> str:
        return "cosolver2b_best_fit"
