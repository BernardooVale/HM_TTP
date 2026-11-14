"""
Heurísticas iterativas S2–S5 e C1–C6 de Faulkner et al. (2015) para o TTP1.

Referência:
Faulkner, H., Polyakovskiy, S., Schultz, T., & Wagner, M. (2015).
"Solving the Traveling Thief Problem: PackIterative and Co-Solver Approaches".
"""

import time
from typing import Dict, List

import numpy as np

from src.core.models import SolucaoTTP, TTPInstance
from src.solvers.base import SolverRegistry
from src.solvers.operators.faulkner_base import FaulknerBaseSolver
from src.solvers.operators.faulkner_operators import (
    bit_flip,
    insertion_local_search,
    one_plus_one_ea,
    pack_iterative,
)


@SolverRegistry.register
class FaulknerS2Solver(FaulknerBaseSolver):
    """S2: Construtiva com refinamento iterativo de mochila (S1 -> BitFlip até convergir)."""

    @property
    def nome(self) -> str:
        return "faulkner_s2"

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        rota = self._gerar_rota_clk(instance, seed=self.seed)
        plano, _ = pack_iterative(
            instance, rota, c=self.c, delta=self.delta, q=self.q
        )
        plano, _, _ = bit_flip(instance, rota, plano, max_passes=None)
        return SolucaoTTP(rota=rota, plano_coleta=plano)


@SolverRegistry.register
class FaulknerS3Solver(FaulknerBaseSolver):
    """S3: Construtiva com refinamento estocástico (S1 -> (1+1)-EA até time_limit)."""

    @property
    def nome(self) -> str:
        return "faulkner_s3"

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        start = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        rota = self._gerar_rota_clk(instance, seed=self.seed)
        plano, _ = pack_iterative(
            instance, rota, c=self.c, delta=self.delta, q=self.q
        )
        plano, _, _ = one_plus_one_ea(
            instance,
            rota,
            plano,
            max_iters=None,
            time_limit=self.time_limit,
            start_time=start,
            rng=rng,
        )
        return SolucaoTTP(rota=rota, plano_coleta=plano)


@SolverRegistry.register
class FaulknerS4Solver(FaulknerBaseSolver):
    """S4: Construtiva com refinamento iterativo de rota (S1 -> Insertion até convergir)."""

    @property
    def nome(self) -> str:
        return "faulkner_s4"

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        rota = self._gerar_rota_clk(instance, seed=self.seed)
        plano, _ = pack_iterative(
            instance, rota, c=self.c, delta=self.delta, q=self.q
        )
        rota, _, _ = insertion_local_search(instance, rota, plano, max_passes=None)
        return SolucaoTTP(rota=rota, plano_coleta=plano)


@SolverRegistry.register
class FaulknerS5Solver(FaulknerBaseSolver):
    """S5: Reamostragem sistemática de rotas (CLK + PackIterative até time_limit)."""

    @property
    def nome(self) -> str:
        return "faulkner_s5"

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        start = time.perf_counter()
        rota, plano, _ = self._amostrar_clk_pack(
            instance, budget_seconds=self.time_limit, start_time=start
        )
        return SolucaoTTP(rota=rota, plano_coleta=plano)


@SolverRegistry.register
class FaulknerC1Solver(FaulknerBaseSolver):
    """C1: Alternância clássica entre mochila (BitFlip) e rota (Insertion)."""

    @property
    def nome(self) -> str:
        return "faulkner_c1"

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        rota = self._gerar_rota_clk(instance, seed=self.seed)
        plano, _ = pack_iterative(
            instance, rota, c=self.c, delta=self.delta, q=self.q
        )
        rota, plano, _ = self._executar_c1(instance, rota, plano)
        return SolucaoTTP(rota=rota, plano_coleta=plano)


@SolverRegistry.register
class FaulknerC2Solver(FaulknerBaseSolver):
    """C2: Alternância com perturbação estocástica (BitFlip + 10k (1+1)-EA + Insertion)."""

    @property
    def nome(self) -> str:
        return "faulkner_c2"

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        rng = np.random.default_rng(self.seed)
        rota = self._gerar_rota_clk(instance, seed=self.seed)
        plano, _ = pack_iterative(
            instance, rota, c=self.c, delta=self.delta, q=self.q
        )
        rota, plano, _ = self._executar_c2(instance, rota, plano, rng=rng)
        return SolucaoTTP(rota=rota, plano_coleta=plano)


@SolverRegistry.register
class FaulknerC3Solver(FaulknerBaseSolver):
    """C3: Amostragem inicial (10% do tempo) + refinamento C1 (90% restantes)."""

    @property
    def nome(self) -> str:
        return "faulkner_c3"

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        start = time.perf_counter()
        rota, plano, _ = self._amostrar_clk_pack(
            instance, budget_seconds=0.1 * self.time_limit, start_time=start
        )
        rota, plano, _ = self._executar_c1(
            instance, rota, plano, start_time=start, time_limit=self.time_limit
        )
        return SolucaoTTP(rota=rota, plano_coleta=plano)


@SolverRegistry.register
class FaulknerC4Solver(FaulknerBaseSolver):
    """C4: Amostragem inicial (10% do tempo) + refinamento C2 (90% restantes)."""

    @property
    def nome(self) -> str:
        return "faulkner_c4"

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        start = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        rota, plano, _ = self._amostrar_clk_pack(
            instance, budget_seconds=0.1 * self.time_limit, start_time=start
        )
        rota, plano, _ = self._executar_c2(
            instance,
            rota,
            plano,
            rng=rng,
            start_time=start,
            time_limit=self.time_limit,
        )
        return SolucaoTTP(rota=rota, plano_coleta=plano)


@SolverRegistry.register
class FaulknerC5Solver(FaulknerBaseSolver):
    """C5: Reinícios múltiplos de C1 até esgotar o orçamento de tempo."""

    @property
    def nome(self) -> str:
        return "faulkner_c5"

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        start = time.perf_counter()
        melhor_rota: List[int] = []
        melhor_plano: Dict[int, List[int]] = {}
        melhor_z = float("-inf")

        iter_idx = 0
        while True:
            iter_seed = None if self.seed is None else self.seed + iter_idx
            rota = self._gerar_rota_clk(
                instance, seed=iter_seed, random_start=(iter_idx > 0)
            )
            plano, _ = pack_iterative(
                instance, rota, c=self.c, delta=self.delta, q=self.q
            )
            rota, plano, z = self._executar_c1(
                instance, rota, plano, start_time=start, time_limit=self.time_limit
            )
            if z > melhor_z or not melhor_rota:
                melhor_rota = rota
                melhor_plano = plano
                melhor_z = z

            iter_idx += 1
            if time.perf_counter() - start >= self.time_limit:
                break

        return SolucaoTTP(rota=melhor_rota, plano_coleta=melhor_plano)


@SolverRegistry.register
class FaulknerC6Solver(FaulknerBaseSolver):
    """C6: Reinícios múltiplos de C2 até esgotar o orçamento de tempo."""

    @property
    def nome(self) -> str:
        return "faulkner_c6"

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        start = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        melhor_rota: List[int] = []
        melhor_plano: Dict[int, List[int]] = {}
        melhor_z = float("-inf")

        iter_idx = 0
        while True:
            iter_seed = None if self.seed is None else self.seed + iter_idx
            rota = self._gerar_rota_clk(
                instance, seed=iter_seed, random_start=(iter_idx > 0)
            )
            plano, _ = pack_iterative(
                instance, rota, c=self.c, delta=self.delta, q=self.q
            )
            rota, plano, z = self._executar_c2(
                instance,
                rota,
                plano,
                rng=rng,
                start_time=start,
                time_limit=self.time_limit,
            )
            if z > melhor_z or not melhor_rota:
                melhor_rota = rota
                melhor_plano = plano
                melhor_z = z

            iter_idx += 1
            if time.perf_counter() - start >= self.time_limit:
                break

        return SolucaoTTP(rota=melhor_rota, plano_coleta=melhor_plano)
