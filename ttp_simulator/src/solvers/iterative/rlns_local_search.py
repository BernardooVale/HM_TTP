"""
Procedimentos de Busca Local para o ReinforceNS (Q-Learning / RLNS) e
ReinforceNS-PM (Primeira Melhoria / First Improvement).

Referências:
- Campos et al., "Heurísticas para o Travelling Thief Problem", Seção 3:
  - Algoritmo 1: Busca local por Primeira Melhoria (ReinforceNS-PM).
  - Algoritmo 2: Busca local RLNS original do ReinforceNS [Wu et al. 2024].
- Wu et al. (2024), IJCAI-24, Seção 4.3, Algoritmo 2, Equações (6) e (7).
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from src.solvers.iterative.mov2optreemp import (
    Mov2OptReempOperator,
    RLNSSolutionState,
)


@dataclass
class LocalSearchResult:
    """Resultado detalhado da execução da busca local (RLNS ou First Improvement)."""

    best_state: RLNSSolutionState
    passes_completed: int
    moves_evaluated: int
    improvements_found: int
    stopped_by_timeout: bool
    elapsed_time: float
    q_table: Optional[np.ndarray] = None


def _filter_non_depot_candidates(
    candidate_neighbors: Dict[int, List[int]],
    depot: int = 1,
) -> Dict[int, List[int]]:
    """
    Garante que o depósito (cidade 1, fixa na posição 0 da rota) não seja
    selecionado como alvo de inversão 2-opt nas sub-rotas internas.
    """
    filtered: Dict[int, List[int]] = {}
    for city_id, neighbors in candidate_neighbors.items():
        if city_id == depot:
            continue
        filtered[city_id] = [
            int(nb) for nb in neighbors if nb != depot and nb != city_id
        ]
    return filtered


def run_first_improvement_search(
    initial_state: RLNSSolutionState,
    operator: Mov2OptReempOperator,
    candidate_neighbors: Dict[int, List[int]],
    iter_max: int = 1000,
    deadline: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> LocalSearchResult:
    """
    Executa a busca local por Primeira Melhoria (ReinforceNS-PM) conforme o
    Algoritmo 1 de Campos et al.:
      - Em cada passagem q = 1 .. iter_max:
        - Para cada cidade x_i (i = 2 .. n, excluindo o depósito):
          - Embaralha aleatoriamente N(x_i).
          - Para cada c' em N(x_i):
            - Verifica limite de tempo T_max.
            - Aplica (Π̄, P̄) <- MOV2OPTREEMP(Π, P, x_i, c', δ).
            - Se f(Π̄, P̄) > f(Π, P): aceita (Π, P) <- (Π̄, P̄), marca
              melhoriaPassagem = verdadeiro e interrompe o laço de candidatos.
        - Se melhoriaPassagem == falso ao fim da passagem, encerra no ótimo local.
    """
    start_time = time.perf_counter()
    if rng is None:
        rng = np.random.default_rng()

    current_state = initial_state.copy()
    valid_neighbors = _filter_non_depot_candidates(candidate_neighbors, depot=1)

    passes_completed = 0
    moves_evaluated = 0
    improvements_found = 0
    stopped_by_timeout = False

    n = len(current_state.route)
    if n <= 2:
        return LocalSearchResult(
            best_state=current_state,
            passes_completed=0,
            moves_evaluated=0,
            improvements_found=0,
            stopped_by_timeout=False,
            elapsed_time=time.perf_counter() - start_time,
        )

    for _ in range(iter_max):
        melhoria_passagem = False

        # Percorre i = 2 .. n (índices 1 .. n-1 da rota, excluindo o depósito na posição 0)
        for idx in range(1, n):
            if deadline is not None and time.perf_counter() >= deadline:
                stopped_by_timeout = True
                return LocalSearchResult(
                    best_state=current_state,
                    passes_completed=passes_completed,
                    moves_evaluated=moves_evaluated,
                    improvements_found=improvements_found,
                    stopped_by_timeout=True,
                    elapsed_time=time.perf_counter() - start_time,
                )

            x_i = current_state.route[idx]
            candidates = valid_neighbors.get(x_i, [])
            if not candidates:
                continue

            shuffled_candidates = candidates.copy()
            rng.shuffle(shuffled_candidates)

            for c_prime in shuffled_candidates:
                if deadline is not None and time.perf_counter() >= deadline:
                    stopped_by_timeout = True
                    return LocalSearchResult(
                        best_state=current_state,
                        passes_completed=passes_completed,
                        moves_evaluated=moves_evaluated,
                        improvements_found=improvements_found,
                        stopped_by_timeout=True,
                        elapsed_time=time.perf_counter() - start_time,
                    )

                cand_state = operator.apply_move(
                    current_state, x_i, c_prime, in_place=False
                )
                moves_evaluated += 1

                if cand_state.objective > current_state.objective + 1e-9:
                    current_state = cand_state
                    melhoria_passagem = True
                    improvements_found += 1
                    break

        passes_completed += 1
        if not melhoria_passagem:
            break

    return LocalSearchResult(
        best_state=current_state,
        passes_completed=passes_completed,
        moves_evaluated=moves_evaluated,
        improvements_found=improvements_found,
        stopped_by_timeout=stopped_by_timeout,
        elapsed_time=time.perf_counter() - start_time,
    )


def run_rlns_qlearning_search(
    initial_state: RLNSSolutionState,
    operator: Mov2OptReempOperator,
    candidate_neighbors: Dict[int, List[int]],
    initial_q_table: np.ndarray,
    iter_max: int = 1000,
    epsilon: float = 0.2,
    alpha_lr: float = 0.1,
    gamma_discount: float = 0.9,
    deadline: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    local_init: str = "neg_inf",
) -> LocalSearchResult:
    """
    Executa a busca local RLNS com Q-Learning conforme o Algoritmo 2 de
    Campos et al. e Wu et al. (2024):
      - Inicializa (Π', P') <- (Π, P) e f' <- f(Π, P).
      - Em cada passagem q = 1 .. iter_max:
        - Define (Π_local, P_local) <- (Π, P) e f_local <- f(Π, P).
        - Para cada i = 2 .. n (cidades da rota excluindo o depósito):
          - Verifica T_max.
          - Seleciona x_j em N(x_i) via ε-greedy (ε = 0.2).
          - f_antes <- f(Π, P).
          - (Π, P) <- MOV2OPTREEMP(Π, P, x_i, x_j, δ).
          - r <- f(Π, P) - f_antes.
          - Se f(Π, P) >= f_local: atualiza (Π_local, P_local) e f_local.
          - Se f(Π, P) > f': atualiza (Π', P') e f'.
          - Atualiza Q(x_i, x_j) <- (1 - λ) Q(x_i, x_j) + λ [r + φ max_{a'} Q(s_{t+1}, a')].
        - Ao fim da passagem:
          - Se f_local > f': atualiza (Π', P') e f'.
          - Restaura (Π, P) <- (Π_local, P_local).
    """
    start_time = time.perf_counter()
    if rng is None:
        rng = np.random.default_rng()

    current_state = initial_state.copy()
    best_global_state = initial_state.copy()
    f_prime = best_global_state.objective

    q_table = initial_q_table.copy()
    valid_neighbors = _filter_non_depot_candidates(candidate_neighbors, depot=1)

    passes_completed = 0
    moves_evaluated = 0
    improvements_found = 0
    stopped_by_timeout = False

    n = len(current_state.route)
    if n <= 2:
        return LocalSearchResult(
            best_state=best_global_state,
            passes_completed=0,
            moves_evaluated=0,
            improvements_found=0,
            stopped_by_timeout=False,
            elapsed_time=time.perf_counter() - start_time,
            q_table=q_table,
        )

    for _ in range(iter_max):
        # Wu et al. (2024), Seção 4 (p. 3), Seção 4.3 e Algoritmo 2 (p. 4):
        # A solução corrente da iteração externa é (Π_0, P_0) = base_pass_state.
        # Para cada cidade x_i (i = 2 .. n), seleciona x_j ∈ N(x_i) via ε-greedy
        # e avalia a solução vizinha (Π_1, P_1) = MOV2OPTREEMP(Π_0, P_0, x_i, x_j, δ),
        # com recompensa r(s_t, a_t) = f(Π_1, P_1) - f(Π_0, P_0).
        # Ao final da passagem (linha 17), transiciona (Π_0, P_0) <- (Π_local, P_local).
        base_pass_state = current_state.copy()
        if local_init == "current":
            local_best_state = base_pass_state.copy()
            f_local = base_pass_state.objective
        else:
            local_best_state = None
            f_local = -np.inf

        for idx in range(1, n):
            if deadline is not None and time.perf_counter() >= deadline:
                stopped_by_timeout = True
                return LocalSearchResult(
                    best_state=best_global_state,
                    passes_completed=passes_completed,
                    moves_evaluated=moves_evaluated,
                    improvements_found=improvements_found,
                    stopped_by_timeout=True,
                    elapsed_time=time.perf_counter() - start_time,
                    q_table=q_table,
                )

            x_i = base_pass_state.route[idx]
            candidates = valid_neighbors.get(x_i, [])
            if not candidates:
                continue

            # Seleção ε-greedy em N(x_i)
            if rng.random() < epsilon:
                x_j = int(candidates[int(rng.integers(0, len(candidates)))])
            else:
                best_q = -np.inf
                best_cand = candidates[0]
                for c in candidates:
                    q_val = q_table[x_i, c]
                    if q_val > best_q:
                        best_q = q_val
                        best_cand = c
                x_j = int(best_cand)

            f_antes = base_pass_state.objective
            cand_state = operator.apply_move(
                base_pass_state, x_i, x_j, in_place=False
            )
            moves_evaluated += 1
            f_depois = cand_state.objective
            reward = f_depois - f_antes

            if f_depois >= f_local - 1e-9:
                f_local = f_depois
                local_best_state = cand_state

            if f_depois > f_prime + 1e-9:
                f_prime = f_depois
                best_global_state = cand_state.copy()
                improvements_found += 1

            # Próximo estado s_{t+1}: próxima cidade na rota a escolher um vizinho
            next_idx = idx + 1 if (idx + 1) < n else 1
            s_next = base_pass_state.route[next_idx]
            next_candidates = valid_neighbors.get(s_next, [])
            if next_candidates:
                max_q_next = max(q_table[s_next, a_p] for a_p in next_candidates)
            else:
                max_q_next = 0.0

            # Equação (3) / Equação (7): atualização de Q(s_t, a_t)
            q_table[x_i, x_j] = (1.0 - alpha_lr) * q_table[
                x_i, x_j
            ] + alpha_lr * (reward + gamma_discount * max_q_next)

        if local_best_state is not None:
            if f_local > f_prime + 1e-9:
                f_prime = f_local
                best_global_state = local_best_state.copy()
            current_state = local_best_state.copy()
        passes_completed += 1

    return LocalSearchResult(
        best_state=best_global_state,
        passes_completed=passes_completed,
        moves_evaluated=moves_evaluated,
        improvements_found=improvements_found,
        stopped_by_timeout=stopped_by_timeout,
        elapsed_time=time.perf_counter() - start_time,
        q_table=q_table,
    )
