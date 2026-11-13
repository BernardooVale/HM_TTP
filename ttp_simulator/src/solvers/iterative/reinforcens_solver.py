"""
Solvers Multi-Start ReinforceNS (Q-Learning / RLNS) e ReinforceNS-PM
(Primeira Melhoria / First Improvement), com suporte às configurações de
pós-processamento P0, P1, P2 e P3 e pareamento estrito de sementes.

Referências:
- Wu et al. (2024), "ReinforceNS: Reinforcement Learning-based Multi-start
  Neighborhood Search for Solving the Traveling Thief Problem", IJCAI-24,
  Algorithm 1 (p. 3).
- Campos et al., "Heurísticas para o Travelling Thief Problem", Seções 3 e 4
  (Hipóteses 1 e 2).
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from src.core.models import SolucaoTTP, Tour, TTPInstance
from src.solvers.base import BaseSolver, SolverRegistry
from src.solvers.iterative.mov2optreemp import Mov2OptReempOperator
from src.solvers.iterative.reinforcens_post_opt import (
    PostOptimizationResult,
    apply_post_optimization,
)
from src.solvers.iterative.rlns_local_search import (
    LocalSearchResult,
    run_first_improvement_search,
    run_rlns_qlearning_search,
)
from src.solvers.packing.greedy import IGP
from src.solvers.tsp.cheapest_insertion import CheapInsertionRegret
from src.solvers.tsp.CLK import ChainedLK, DoubleBridge, LKLocalSearch
from src.solvers.tsp.one_tree import build_one_tree_preprocessing


@dataclass
class StartTelemetry:
    """Telemetria de um único ciclo (start) do framework multi-start."""

    start_index: int
    init_objective: float
    local_search_objective: float
    post_opt_objective: float
    local_search_passes: int
    local_search_moves: int
    ifp_flips: int
    rrp_relocations: int
    k: float
    beta: float
    gamma: float
    delta: float


@dataclass
class ReinforceNSRunTelemetry:
    """Telemetria consolidada de uma execução completa do solver."""

    search_strategy: str
    post_opt_mode: str
    starts_completed: int
    best_objective: float
    first_start_init_objective: float = 0.0
    first_start_local_objective: float = 0.0
    first_start_post_opt_objective: float = 0.0
    elapsed_time: float = 0.0
    starts: List[StartTelemetry] = field(default_factory=list)


class _BaseReinforceNSSolver(BaseSolver):
    """
    Implementação base compartilhada entre ReinforceNS (Q-learning) e
    ReinforceNS-PM (First Improvement), garantindo que todos os componentes
    fora da busca local (1-tree, CLKH, IGP, MOV2OPTREEMP, IFP e RRP) e o
    pareamento de sementes sejam 100% idênticos.
    """

    def __init__(
        self,
        search_strategy: str = "rlns",
        post_opt_mode: str = "P3",
        time_limit: float = 10.0,
        max_starts: Optional[int] = None,
        iter_max: int = 1000,
        num_candidates: int = 15,
        clk_num_tries: int = 5,
        clk_num_kicks: int = 2,
        igp_iterations: int = 20,
        epsilon: float = 0.2,
        alpha_lr: float = 0.1,
        gamma_discount: float = 0.9,
        subgradient_steps: int = 0,
        neighbor_ranking: str = "distance",
        modelo: str = "TTP1",
        seed: Optional[int] = 42,
    ):
        if search_strategy not in ("rlns", "pm"):
            raise ValueError("search_strategy deve ser 'rlns' ou 'pm'.")
        self.search_strategy = search_strategy
        self.post_opt_mode = post_opt_mode.upper().strip()
        self.time_limit = float(time_limit)
        self.max_starts = max_starts
        self.iter_max = int(iter_max)
        self.num_candidates = int(num_candidates)
        self.clk_num_tries = int(clk_num_tries)
        self.clk_num_kicks = int(clk_num_kicks)
        self.igp_iterations = int(igp_iterations)
        self.epsilon = float(epsilon)
        self.alpha_lr = float(alpha_lr)
        self.gamma_discount = float(gamma_discount)
        self.subgradient_steps = int(subgradient_steps)
        self.neighbor_ranking = neighbor_ranking
        self.modelo = modelo
        self.seed = seed
        self.last_telemetry: Optional[ReinforceNSRunTelemetry] = None

    @property
    def nome(self) -> str:
        base = "reinforcens" if self.search_strategy == "rlns" else "reinforcens_pm"
        if self.post_opt_mode == "P3":
            return base
        return f"{base}_{self.post_opt_mode.lower()}"

    def _build_initial_solution(
        self,
        instance: TTPInstance,
        base_lk_tour: Tour,
        start_idx: int,
        deadline: Optional[float] = None,
    ) -> Tuple[Tour, Dict[int, List[int]], IGP]:
        """
        Gera uma solução inicial (Π, P) via CLKH + IGP com semente controlada
        por `start_idx`, garantindo pareamento exato entre variantes.
        Partindo do tour base já refinado por LK (`base_lk_tour`), aplica
        perturbações DoubleBridge seguidas de LKLocalSearch sem repetir o LK
        determinístico inicial sobre o tour de inserção mais barata.
        """
        if self.seed is not None:
            np.random.seed(int(self.seed) + start_idx * 10007)

        best_tour = Tour(order=list(base_lk_tour.order), cost=base_lk_tour.cost)
        if best_tour.length >= 8 and self.clk_num_kicks > 0:
            kicker = DoubleBridge()
            lk = LKLocalSearch(grafo=instance.grafo, num_tries=self.clk_num_tries)
            for _ in range(self.clk_num_kicks):
                if (
                    start_idx > 0
                    and deadline is not None
                    and time.perf_counter() >= deadline
                ):
                    break
                try:
                    kicked_tour = kicker.kick(best_tour, instance.grafo)
                except ValueError:
                    continue
                cand_tour, cand_cost = lk.solve(kicked_tour)
                if cand_cost < best_tour.cost - 1e-9:
                    best_tour = cand_tour

        best_tour.normalize_to_city_1()
        igp = IGP(scenario=instance, tour=best_tour, modelo=self.modelo)
        pack_dict = igp.solve(theta=5.0, num_iterations=self.igp_iterations)
        return best_tour, pack_dict, igp

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        # Linha 3 do Algorithm 1 (Wu et al., 2024): Pré-processamento 1-tree
        prep = build_one_tree_preprocessing(
            grafo=instance.grafo,
            num_candidates=self.num_candidates,
            tau=0.5,
            b=1e-6,
            subgradient_steps=self.subgradient_steps,
            neighbor_ranking=self.neighbor_ranking,
        )

        # Constrói e refina o tour base determinístico (CheapInsertionRegret + LK) uma vez
        cache_attr = f"_cached_base_lk_tour_{self.clk_num_tries}"
        base_lk_tour = getattr(instance.grafo, cache_attr, None)
        if base_lk_tour is None:
            base_ci_tour = CheapInsertionRegret(
                grafo=instance.grafo
            ).find_regret_insertion()
            lk_init = LKLocalSearch(
                grafo=instance.grafo, num_tries=self.clk_num_tries
            )
            base_lk_tour, _ = lk_init.solve(base_ci_tour)
            base_lk_tour.normalize_to_city_1()
            try:
                setattr(instance.grafo, cache_attr, base_lk_tour)
            except Exception:
                pass

        start_clock = time.perf_counter()
        deadline = (
            start_clock + self.time_limit
            if self.time_limit and self.time_limit > 0
            else None
        )

        best_route: Optional[List[int]] = None
        best_plan: Optional[Dict[int, List[int]]] = None
        best_obj = -float("inf")

        telemetry = ReinforceNSRunTelemetry(
            search_strategy=self.search_strategy,
            post_opt_mode=self.post_opt_mode,
            starts_completed=0,
            best_objective=-float("inf"),
        )

        start_idx = 0
        while True:
            if self.max_starts is not None and start_idx >= self.max_starts:
                break
            if (
                start_idx > 0
                and deadline is not None
                and time.perf_counter() >= deadline
            ):
                break

            # Linhas 5-6 do Algorithm 1: Solução inicial (Π, P) <- CLKH + IGP(Π)
            tour, init_plan, igp = self._build_initial_solution(
                instance=instance,
                base_lk_tour=base_lk_tour,
                start_idx=start_idx,
                deadline=deadline,
            )

            k_val = getattr(igp, "best_k", 1.0)
            beta_val = getattr(igp, "best_beta", 1.0)
            gamma_val = getattr(igp, "best_gamma", 1.0)
            delta_val = getattr(igp, "best_delta", 1.0)

            operator = Mov2OptReempOperator(
                instance=instance,
                k=k_val,
                beta=beta_val,
                gamma=gamma_val,
                delta=delta_val,
                modelo=self.modelo,
            )
            init_state = operator.build_initial_state(tour.order, init_plan)

            ls_seed = (
                int(self.seed) + start_idx * 10007 + 1
                if self.seed is not None
                else None
            )
            ls_rng = np.random.default_rng(ls_seed)

            # Linha 7 do Algorithm 1: Busca Local (RLNS ou First Improvement)
            if self.search_strategy == "rlns":
                ls_res: LocalSearchResult = run_rlns_qlearning_search(
                    initial_state=init_state,
                    operator=operator,
                    candidate_neighbors=prep.candidate_neighbors,
                    initial_q_table=prep.q_table,
                    iter_max=self.iter_max,
                    epsilon=self.epsilon,
                    alpha_lr=self.alpha_lr,
                    gamma_discount=self.gamma_discount,
                    deadline=deadline,
                    rng=ls_rng,
                )
            else:
                ls_res = run_first_improvement_search(
                    initial_state=init_state,
                    operator=operator,
                    candidate_neighbors=prep.candidate_neighbors,
                    iter_max=self.iter_max,
                    deadline=deadline,
                    rng=ls_rng,
                )

            # Linhas 8-9 do Algorithm 1: Pós-otimização (IFP + RRP conforme P0..P3)
            # Executa ao fim da busca local do start corrente (custo ~O(m + n^2) < 20ms)
            post_res: PostOptimizationResult = apply_post_optimization(
                instance=instance,
                route=ls_res.best_state.route,
                plan=ls_res.best_state.plan,
                mode=self.post_opt_mode,
                modelo=self.modelo,
                deadline=None,
            )

            # Linhas 10-14 do Algorithm 1: Atualiza melhor solução global (Π*, P*)
            if post_res.objective_final > best_obj:
                best_obj = post_res.objective_final
                best_route = list(post_res.route)
                best_plan = {
                    c: list(ids) for c, ids in post_res.plan.items() if ids
                }

            start_info = StartTelemetry(
                start_index=start_idx,
                init_objective=init_state.objective,
                local_search_objective=ls_res.best_state.objective,
                post_opt_objective=post_res.objective_final,
                local_search_passes=ls_res.passes_completed,
                local_search_moves=ls_res.moves_evaluated,
                ifp_flips=post_res.ifp_flips_accepted,
                rrp_relocations=post_res.rrp_relocations_accepted,
                k=k_val,
                beta=beta_val,
                gamma=gamma_val,
                delta=delta_val,
            )
            telemetry.starts.append(start_info)
            if start_idx == 0:
                telemetry.first_start_init_objective = init_state.objective
                telemetry.first_start_local_objective = (
                    ls_res.best_state.objective
                )
                telemetry.first_start_post_opt_objective = (
                    post_res.objective_final
                )

            start_idx += 1
            telemetry.starts_completed = start_idx
            telemetry.best_objective = best_obj

            if deadline is not None and time.perf_counter() >= deadline:
                break

        telemetry.elapsed_time = time.perf_counter() - start_clock
        self.last_telemetry = telemetry

        return SolucaoTTP(
            rota=best_route if best_route is not None else [1],
            plano_coleta=best_plan if best_plan is not None else {},
        )


@SolverRegistry.register("reinforcens")
class ReinforceNSSolver(_BaseReinforceNSSolver):
    """Solver ReinforceNS original com Q-Learning (Algoritmo 2) e pós-otimização P3."""

    def __init__(self, **kwargs):
        kwargs.setdefault("search_strategy", "rlns")
        kwargs.setdefault("post_opt_mode", "P3")
        super().__init__(**kwargs)


@SolverRegistry.register("reinforcens_pm")
class ReinforceNSPMSolver(_BaseReinforceNSSolver):
    """Solver ReinforceNS-PM com Primeira Melhoria (Algoritmo 1) e pós-otimização P3."""

    def __init__(self, **kwargs):
        kwargs.setdefault("search_strategy", "pm")
        kwargs.setdefault("post_opt_mode", "P3")
        super().__init__(**kwargs)


@SolverRegistry.register("reinforcens_p0")
class ReinforceNSP0Solver(_BaseReinforceNSSolver):
    def __init__(self, **kwargs):
        kwargs.setdefault("search_strategy", "rlns")
        kwargs.setdefault("post_opt_mode", "P0")
        super().__init__(**kwargs)


@SolverRegistry.register("reinforcens_p1")
class ReinforceNSP1Solver(_BaseReinforceNSSolver):
    def __init__(self, **kwargs):
        kwargs.setdefault("search_strategy", "rlns")
        kwargs.setdefault("post_opt_mode", "P1")
        super().__init__(**kwargs)


@SolverRegistry.register("reinforcens_p2")
class ReinforceNSP2Solver(_BaseReinforceNSSolver):
    def __init__(self, **kwargs):
        kwargs.setdefault("search_strategy", "rlns")
        kwargs.setdefault("post_opt_mode", "P2")
        super().__init__(**kwargs)


@SolverRegistry.register("reinforcens_pm_p0")
class ReinforceNSPMP0Solver(_BaseReinforceNSSolver):
    def __init__(self, **kwargs):
        kwargs.setdefault("search_strategy", "pm")
        kwargs.setdefault("post_opt_mode", "P0")
        super().__init__(**kwargs)


@SolverRegistry.register("reinforcens_pm_p1")
class ReinforceNSPMP1Solver(_BaseReinforceNSSolver):
    def __init__(self, **kwargs):
        kwargs.setdefault("search_strategy", "pm")
        kwargs.setdefault("post_opt_mode", "P1")
        super().__init__(**kwargs)


@SolverRegistry.register("reinforcens_pm_p2")
class ReinforceNSPMP2Solver(_BaseReinforceNSSolver):
    def __init__(self, **kwargs):
        kwargs.setdefault("search_strategy", "pm")
        kwargs.setdefault("post_opt_mode", "P2")
        super().__init__(**kwargs)
