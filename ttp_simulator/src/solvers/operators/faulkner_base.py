"""Classe base compartilhada para as heurísticas S1–S5 e C1–C6 de Faulkner et al. (2015)."""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.models import TTPInstance
from src.solvers.base import BaseSolver
from src.solvers.operators.faulkner_operators import (
    bit_flip,
    insertion_local_search,
    one_plus_one_ea,
    pack_iterative,
)
from src.solvers.tsp.CLK import ChainedLK


class FaulknerBaseSolver(BaseSolver):
    """Base compartilhada para as heurísticas S1–S5 e C1–C6 de Faulkner et al. (2015)."""

    def __init__(
        self,
        time_limit: float = 5.0,
        seed: Optional[int] = None,
        clk_tries: int = 5,
        clk_kicks: int = 5,
        c: float = 5.0,
        delta: float = 2.5,
        q: int = 20,
        ea_iters_c2: int = 10_000,
    ):
        self.time_limit = time_limit
        self.seed = seed
        self.clk_tries = clk_tries
        self.clk_kicks = clk_kicks
        self.c = c
        self.delta = delta
        self.q = q
        self.ea_iters_c2 = ea_iters_c2

    def _gerar_rota_clk(
        self,
        instance: TTPInstance,
        seed: Optional[int],
        random_start: bool = False,
    ) -> List[int]:
        clk = ChainedLK(
            grafo=instance.grafo,
            num_tries=self.clk_tries,
            num_kicks=self.clk_kicks,
            seed=seed,
        )
        tour = clk.solve(random_start=random_start)
        return list(tour.order)

    def _amostrar_clk_pack(
        self,
        instance: TTPInstance,
        budget_seconds: float,
        start_time: float,
    ) -> Tuple[List[int], Dict[int, List[int]], float]:
        """Executa amostragem CLK + PackIterative dentro do orçamento de tempo."""
        melhor_rota: List[int] = []
        melhor_plano: Dict[int, List[int]] = {}
        melhor_z = float("-inf")

        iter_idx = 0
        while True:
            iter_seed = None if self.seed is None else self.seed + iter_idx
            rota = self._gerar_rota_clk(
                instance, seed=iter_seed, random_start=(iter_idx > 0)
            )
            plano, z = pack_iterative(
                instance, rota, c=self.c, delta=self.delta, q=self.q
            )
            if z > melhor_z or not melhor_rota:
                melhor_rota = rota
                melhor_plano = plano
                melhor_z = z

            iter_idx += 1
            if time.perf_counter() - start_time >= budget_seconds:
                break

        return melhor_rota, melhor_plano, melhor_z

    def _executar_c1(
        self,
        instance: TTPInstance,
        rota: List[int],
        plano: Dict[int, List[int]],
        start_time: Optional[float] = None,
        time_limit: Optional[float] = None,
    ) -> Tuple[List[int], Dict[int, List[int]], float]:
        """Alterna 1 passagem de BitFlip e 1 passagem de Insertion até convergir."""
        melhor_z = float("-inf")
        while True:
            if time_limit is not None and start_time is not None:
                if time.perf_counter() - start_time >= time_limit:
                    break

            plano, z_bf, mel_bf = bit_flip(
                instance,
                rota,
                plano,
                max_passes=1,
                time_limit=time_limit,
                start_time=start_time,
            )
            rota, z_ins, mel_ins = insertion_local_search(
                instance,
                rota,
                plano,
                max_passes=1,
                time_limit=time_limit,
                start_time=start_time,
            )
            melhor_z = z_ins if mel_ins else z_bf
            if not mel_bf and not mel_ins:
                break

        return rota, plano, melhor_z

    def _executar_c2(
        self,
        instance: TTPInstance,
        rota: List[int],
        plano: Dict[int, List[int]],
        rng: np.random.Generator,
        start_time: Optional[float] = None,
        time_limit: Optional[float] = None,
    ) -> Tuple[List[int], Dict[int, List[int]], float]:
        """Alterna BitFlip (1 passagem), (1+1)-EA (10k iters) e Insertion (1 passagem)."""
        melhor_z = float("-inf")
        while True:
            if time_limit is not None and start_time is not None:
                if time.perf_counter() - start_time >= time_limit:
                    break

            plano, _, mel_bf = bit_flip(
                instance,
                rota,
                plano,
                max_passes=1,
                time_limit=time_limit,
                start_time=start_time,
            )
            plano, _, mel_ea = one_plus_one_ea(
                instance,
                rota,
                plano,
                max_iters=self.ea_iters_c2,
                time_limit=time_limit,
                start_time=start_time,
                rng=rng,
            )
            rota, z_ins, mel_ins = insertion_local_search(
                instance,
                rota,
                plano,
                max_passes=1,
                time_limit=time_limit,
                start_time=start_time,
            )
            melhor_z = z_ins
            if not mel_bf and not mel_ea and not mel_ins:
                break

        return rota, plano, melhor_z
