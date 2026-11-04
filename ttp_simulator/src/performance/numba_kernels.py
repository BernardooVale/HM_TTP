"""
Módulo de aceleração de alta performance para o TTP Simulator:
1. Kernels compilados via Numba JIT (@njit(cache=True)) com fallback transparente caso Numba não esteja instalado.
2. Cache de Programação Dinâmica / Memoização Adaptativa à Memória RAM disponível do sistema.
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator


# ==============================================================================
# 1. GERENCIADOR DE MEMÓRIA ADAPTATIVO (RAM-Aware LRU / Stage Memoization)
# ==============================================================================

class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def detectar_memoria_sistema_bytes() -> Tuple[int, int]:
    """
    Detecta a memória física total e a memória física livre do sistema operacional
    sem depender de bibliotecas externas (psutil).
    Retorna (total_bytes, disponivel_bytes).
    """
    # Windows via ctypes kernel32
    if os.name == "nt":
        try:
            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys), int(stat.ullAvailPhys)
        except Exception:
            pass

    # Linux / macOS / POSIX via /proc/meminfo ou sysconf
    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            total_pages = os.sysconf("SC_PHYS_PAGES")
            avail_pages = os.sysconf("SC_AVPHYS_PAGES") if "SC_AVPHYS_PAGES" in os.sysconf_names else total_pages // 2
            if page_size > 0 and total_pages > 0:
                return int(page_size * total_pages), int(page_size * max(avail_pages, 1))
        except Exception:
            pass

    # Fallback conservador: 8 GB total, 2 GB disponível
    return 8 * (1024 ** 3), 2 * (1024 ** 3)


def estimar_tamanho_bytes(obj: Any) -> int:
    """Estima recursivamente de forma rápida o consumo de memória em bytes de um objeto no cache."""
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes) + 128
    if isinstance(obj, (bytes, bytearray, str)):
        return len(obj) + 64
    if isinstance(obj, (int, float, bool)) or obj is None:
        return 32
    if isinstance(obj, (tuple, list, set, frozenset)):
        if not obj:
            return 64
        # Amostragem rápida para coleções grandes homogêneas
        n = len(obj)
        if n > 32:
            primeiro = next(iter(obj))
            return 64 + n * (estimar_tamanho_bytes(primeiro) + 8)
        return 64 + sum(estimar_tamanho_bytes(x) + 8 for x in obj)
    if isinstance(obj, dict):
        if not obj:
            return 128
        n = len(obj)
        if n > 32:
            k, v = next(iter(obj.items()))
            return 128 + n * (estimar_tamanho_bytes(k) + estimar_tamanho_bytes(v) + 24)
        return 128 + sum(estimar_tamanho_bytes(k) + estimar_tamanho_bytes(v) + 24 for k, v in obj.items())
    return sys.getsizeof(obj, 128)


class AdaptiveRAMCache:
    """
    Cache LRU de Programação Dinâmica / Memoização adaptável à memória RAM do computador.
    - Calcula dinamicamente o orçamento de memória (budget) com base na RAM física livre
      dividida pelo número de processos workers ativos.
    - Monitora tanto os bytes alocados no próprio cache quanto a pressão global da RAM do sistema,
      descartando entradas antigas (LRU) automaticamente caso o orçamento seja atingido ou
      a RAM livre do sistema fique abaixo da margem de segurança.
    """

    def __init__(
        self,
        num_workers: int = 1,
        fracao_ram_livre: float = 0.35,
        min_budget_mb: float = 32.0,
        max_budget_mb: float = 2048.0,
        reserva_minima_sistema_mb: float = 512.0,
    ):
        self.num_workers = max(1, int(num_workers))
        self.fracao_ram_livre = max(0.05, min(0.80, float(fracao_ram_livre)))
        self.min_budget_bytes = int(min_budget_mb * (1024 ** 2))
        self.max_budget_bytes = int(max_budget_mb * (1024 ** 2))
        self.reserva_minima_bytes = int(reserva_minima_sistema_mb * (1024 ** 2))

        self._store: OrderedDict[Any, Tuple[Any, int]] = OrderedDict()
        self._current_bytes: int = 0
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self._op_counter: int = 0

        self.budget_bytes: int = self._calcular_orcamento_bytes()

    def _calcular_orcamento_bytes(self) -> int:
        _, avail_bytes = detectar_memoria_sistema_bytes()
        utilizavel = max(0, avail_bytes - self.reserva_minima_bytes)
        por_worker = int((utilizavel * self.fracao_ram_livre) / self.num_workers)
        return max(self.min_budget_bytes, min(self.max_budget_bytes, por_worker))

    def reconfigurar_workers(self, num_workers: int) -> None:
        self.num_workers = max(1, int(num_workers))
        self.budget_bytes = self._calcular_orcamento_bytes()
        self._evict_se_necessario()

    def _evict_se_necessario(self, bytes_adicionais: int = 0) -> None:
        self._op_counter += 1
        # A cada 256 escritas, reconsulta a RAM física livre do SO para se adaptar a variações externas
        if self._op_counter % 256 == 0:
            self.budget_bytes = self._calcular_orcamento_bytes()

        alvo = self.budget_bytes
        while self._store and (self._current_bytes + bytes_adicionais > alvo):
            _, (_, tam) = self._store.popitem(last=False)
            self._current_bytes = max(0, self._current_bytes - tam)
            self.evictions += 1

    def get(self, key: Any, default: Any = None) -> Any:
        item = self._store.get(key)
        if item is not None:
            self.hits += 1
            self._store.move_to_end(key)
            return item[0]
        self.misses += 1
        return default

    def set(self, key: Any, value: Any) -> None:
        tam = estimar_tamanho_bytes(value) + estimar_tamanho_bytes(key)
        if key in self._store:
            _, tam_antigo = self._store.pop(key)
            self._current_bytes = max(0, self._current_bytes - tam_antigo)

        # Se um único objeto exceder o orçamento inteiro, não armazena
        if tam > self.budget_bytes:
            return

        self._evict_se_necessario(tam)
        self._store[key] = (value, tam)
        self._current_bytes += tam

    def clear(self) -> None:
        self._store.clear()
        self._current_bytes = 0

    def status_resumo(self) -> Dict[str, Any]:
        total_ram, avail_ram = detectar_memoria_sistema_bytes()
        total_req = self.hits + self.misses
        return {
            "ram_total_mb": round(total_ram / (1024 ** 2), 1),
            "ram_disponivel_mb": round(avail_ram / (1024 ** 2), 1),
            "budget_worker_mb": round(self.budget_bytes / (1024 ** 2), 1),
            "uso_atual_mb": round(self._current_bytes / (1024 ** 2), 2),
            "entradas": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round((self.hits / total_req) * 100.0, 1) if total_req > 0 else 0.0,
            "evictions": self.evictions,
            "numba_ativo": NUMBA_AVAILABLE,
        }


# Instância global por processo (cada worker no ProcessPool tem seu próprio espaço de endereçamento)
_GLOBAL_RAM_CACHE = AdaptiveRAMCache(
    num_workers=max(1, int(os.environ.get("TTP_WORKERS", os.cpu_count() or 4)))
)


def obter_cache_ram_global() -> AdaptiveRAMCache:
    return _GLOBAL_RAM_CACHE


# ==============================================================================
# 2. KERNELS NUMBA JIT PARA KNAPSACK 0/1 EXATO (DP / Branch-and-Bound iterativo)
# ==============================================================================

@njit(cache=True)
def numba_knapsack_01_bb(
    pesos: np.ndarray,
    valores: np.ndarray,
    capacidade: float,
) -> Tuple[float, np.ndarray]:
    """
    Branch-and-Bound exato compilado via Numba para o Problema da Mochila 0/1.
    Recebe arrays já ordenados por densidade decrescente (valor / peso).
    Retorna (melhor_valor, mascara_selecionados_bool).
    """
    n = len(pesos)
    best_sel = np.zeros(n, dtype=np.uint8)
    curr_sel = np.zeros(n, dtype=np.uint8)
    best_val = 0.0

    # Pilha explícita para execução iterativa sem recursão em código nativo
    # Estado na pilha: (idx, fase, peso_atual, valor_atual)
    # fase 0: tentar ramo de inclusão (left) e agendar ramo de exclusão (right)
    # fase 1: desfazer inclusão na volta (backtrack) e tentar exclusão
    stack_idx = np.empty(n + 2, dtype=np.int32)
    stack_phase = np.empty(n + 2, dtype=np.int8)
    stack_w = np.empty(n + 2, dtype=np.float64)
    stack_v = np.empty(n + 2, dtype=np.float64)

    top = 0
    stack_idx[0] = 0
    stack_phase[0] = 0
    stack_w[0] = 0.0
    stack_v[0] = 0.0

    while top >= 0:
        idx = stack_idx[top]
        phase = stack_phase[top]
        w_cur = stack_w[top]
        v_cur = stack_v[top]

        if phase == 0:
            if v_cur > best_val:
                best_val = v_cur
                for k in range(n):
                    best_sel[k] = curr_sel[k]

            if idx == n:
                top -= 1
                continue

            # Limite superior fracionário (Dantzig bound)
            ub = v_cur
            w_rem = w_cur
            for i in range(idx, n):
                wi = pesos[i]
                vi = valores[i]
                if w_rem + wi <= capacidade:
                    w_rem += wi
                    ub += vi
                else:
                    if wi > 0.0:
                        ub += (capacidade - w_rem) * (vi / wi)
                    break

            if ub <= best_val:
                top -= 1
                continue

            # Prepara retorno para fase 1 (ramo de exclusão)
            stack_phase[top] = 1

            # Ramo 1: incluir item idx se couber
            if w_cur + pesos[idx] <= capacidade:
                curr_sel[idx] = 1
                top += 1
                stack_idx[top] = idx + 1
                stack_phase[top] = 0
                stack_w[top] = w_cur + pesos[idx]
                stack_v[top] = v_cur + valores[idx]

        elif phase == 1:
            curr_sel[idx] = 0
            # Substitui frame atual pelo ramo 2: não incluir item idx
            stack_idx[top] = idx + 1
            stack_phase[top] = 0
            # stack_w[top] e stack_v[top] permanecem w_cur e v_cur

    return best_val, best_sel


# ==============================================================================
# 3. KERNELS NUMBA JIT PARA DISSIPAÇÃO DE CALOR (MATRIZ 2D)
# ==============================================================================

@njit(cache=True)
def numba_calcular_menores_caminhos(D: np.ndarray, is_metric: bool) -> np.ndarray:
    """
    Retorna a matriz de menores caminhos entre todos os pares (1..n).
    - Se o grafo for métrico completo (EUC_2D ou CEIL_2D), a desigualdade triangular
      é sempre satisfeita (D[u, v] <= D[u, k] + D[k, v]), portanto a distância mínima
      é exatamente a própria matriz D em O(1).
    - Caso contrário (EXPLICIT não-métrico), executa Floyd-Warshall em código nativo.
    """
    if is_metric:
        return D
    dim = D.shape[0]
    dist = D.copy()
    for k in range(1, dim):
        for i in range(1, dim):
            d_ik = dist[i, k]
            for j in range(1, dim):
                if d_ik + dist[k, j] < dist[i, j]:
                    dist[i, j] = d_ik + dist[k, j]
    return dist


@njit(cache=True)
def numba_dissipar_calor_matriz(
    D: np.ndarray,
    SP: np.ndarray,
    temp: np.ndarray,
    modo_dissipacao: int,
    direcionado: bool,
) -> np.ndarray:
    """
    Calcula a matriz de calor (N+1 x N+1, indexada 1..N) em código nativo Numba.
    modo_dissipacao: 0 = adjacent_only, 1 = bfs, 2 = dijkstra.
    Equivalência matemática exata com _dissipar_calor original.
    """
    dim = D.shape[0]
    n = dim - 1
    calor = np.zeros((dim, dim), dtype=np.float64)

    if not direcionado:
        if modo_dissipacao == 0:  # adjacent_only
            for u in range(1, dim):
                t_u = temp[u]
                for v in range(u + 1, dim):
                    t_v = temp[v]
                    if t_u > 0.0 or t_v > 0.0:
                        d = D[u, v]
                        if d < 0.001:
                            d = 0.001
                        val = (t_u + t_v) / d
                        calor[u, v] = val
                        calor[v, u] = val
        elif modo_dissipacao == 1:  # bfs
            t_total = 0.0
            for u in range(1, dim):
                t_total += temp[u]
            for u in range(1, dim):
                t_u = temp[u]
                for v in range(u + 1, dim):
                    val = (t_u + temp[v] + t_total) / (2.0 * (1.0 + D[u, v]))
                    calor[u, v] = val
                    calor[v, u] = val
        else:  # dijkstra
            for s in range(1, dim):
                t_s = temp[s]
                if t_s <= 0.0:
                    continue
                for u in range(1, dim):
                    du = SP[s, u]
                    for v in range(u + 1, dim):
                        dv = SP[s, v]
                        m = du if du < dv else dv
                        d_edge = m + D[u, v] * 0.5
                        calor[u, v] += t_s / (1.0 + d_edge)
            for u in range(1, dim):
                for v in range(u + 1, dim):
                    calor[v, u] = calor[u, v]
    else:
        if modo_dissipacao == 0:  # adjacent_only direcionado
            for u in range(1, dim):
                t_u = temp[u]
                if t_u <= 0.0:
                    continue
                for v in range(1, dim):
                    if u == v:
                        continue
                    d = D[u, v]
                    if d < 0.001:
                        d = 0.001
                    calor[u, v] = t_u / d
        elif modo_dissipacao == 1:  # bfs direcionado
            t_total = 0.0
            for u in range(1, dim):
                t_total += temp[u]
            n_outros = float(n - 2 if n - 2 > 1 else 1)
            for u in range(1, dim):
                t_u = temp[u]
                for v in range(1, dim):
                    if u == v:
                        continue
                    outras = t_total - t_u - temp[v]
                    if outras < 0.0:
                        outras = 0.0
                    calor[u, v] = (t_u + outras / n_outros) / (2.0 * (1.0 + D[u, v]))
        else:  # dijkstra direcionado
            for s in range(1, dim):
                t_s = temp[s]
                if t_s <= 0.0:
                    continue
                for u in range(1, dim):
                    du = SP[s, u]
                    for v in range(1, dim):
                        if u == v or s == v:
                            continue
                        d_edge = du + D[u, v] * 0.5
                        calor[u, v] += t_s / (1.0 + d_edge)

    return calor


# ==============================================================================
# 4. KERNELS NUMBA JIT PARA CONSTRUÇÃO DE ROTA E 2-OPT / HOLE OPTIMIZATION
# ==============================================================================

@njit(cache=True)
def numba_construir_rota_gulosa(
    D: np.ndarray,
    calor: np.ndarray,
    metodo_id: int,
    direcao_final: bool,
    afinidade_termica: bool,
    gamma: float,
) -> np.ndarray:
    """
    Constrói a rota para métodos sequenciais gulosos em O(N^2) nativo:
    metodo_id:
      0 = walk
      2 = backward_edges
      3 = backward_directed
      4 = forward_directed
    Preserva exatamente a mesma regra de desempate (menor ID de cidade ainda não visitada).
    """
    dim = D.shape[0]
    n = dim - 1
    visitado = np.zeros(dim, dtype=np.uint8)
    visitado[1] = 1
    sequencia = np.empty(n, dtype=np.int32)
    sequencia[0] = 1
    atual = 1

    for step in range(1, n):
        melhor_cand = -1
        if metodo_id == 0:  # walk
            if not direcao_final:  # calor_inicio -> max
                melhor_score = -1.0e300
                for cand in range(2, dim):
                    if visitado[cand] == 0:
                        sc = calor[atual, cand]
                        if sc > melhor_score:
                            melhor_score = sc
                            melhor_cand = cand
            else:  # calor_final -> min
                melhor_score = 1.0e300
                for cand in range(2, dim):
                    if visitado[cand] == 0:
                        sc = calor[atual, cand]
                        if sc < melhor_score:
                            melhor_score = sc
                            melhor_cand = cand

        elif metodo_id == 2:  # backward_edges -> max
            melhor_score = -1.0e300
            for cand in range(2, dim):
                if visitado[cand] == 0:
                    c_val = calor[atual, cand]
                    if afinidade_termica:
                        d_val = D[atual, cand]
                        if d_val < 0.001:
                            d_val = 0.001
                        sc = c_val / (d_val ** gamma)
                    else:
                        sc = c_val
                    if sc > melhor_score:
                        melhor_score = sc
                        melhor_cand = cand

        elif metodo_id == 3:  # backward_directed -> max
            melhor_score = -1.0e300
            for cand in range(2, dim):
                if visitado[cand] == 0:
                    c_val = calor[cand, atual]
                    d_val = D[cand, atual]
                    if d_val < 0.001:
                        d_val = 0.001
                    if afinidade_termica:
                        sc = (c_val + 1e-6) / (d_val ** gamma)
                    else:
                        sc = c_val + (1e-6 / d_val)
                    if sc > melhor_score:
                        melhor_score = sc
                        melhor_cand = cand

        else:  # 4 = forward_directed -> min
            melhor_score = 1.0e300
            for cand in range(2, dim):
                if visitado[cand] == 0:
                    c_val = calor[atual, cand]
                    d_val = D[atual, cand]
                    if d_val < 0.001:
                        d_val = 0.001
                    if afinidade_termica:
                        sc = (c_val * (d_val ** gamma)) + (1e-6 * d_val)
                    else:
                        sc = c_val + (1e-6 * d_val)
                    if sc < melhor_score:
                        melhor_score = sc
                        melhor_cand = cand

        visitado[melhor_cand] = 1
        sequencia[step] = melhor_cand
        atual = melhor_cand

    if metodo_id in (2, 3):
        # Inverte os elementos de índice 1..n-1 manteniendo cidade 1 no índice 0
        rota = np.empty(n, dtype=np.int32)
        rota[0] = 1
        for i in range(1, n):
            rota[i] = sequencia[n - i]
        return rota

    return sequencia


@njit(cache=True)
def numba_construir_rota_kruskal(
    D: np.ndarray,
    calor: np.ndarray,
    afinidade_termica: bool,
    gamma: float,
) -> np.ndarray:
    """
    Constrói o ciclo Hamiltoniano pelo método de Kruskal térmico em código nativo.
    Usa ordenação estável (mergesort) na chave negada para preservar exatamente a mesma
    ordem de desempate do Timsort `sorted(..., reverse=True)` do Python sobre (u, v) crescente.
    """
    dim = D.shape[0]
    n = dim - 1
    num_edges = (n * (n - 1)) // 2

    u_arr = np.empty(num_edges, dtype=np.int32)
    v_arr = np.empty(num_edges, dtype=np.int32)
    neg_scores = np.empty(num_edges, dtype=np.float64)

    idx = 0
    for u in range(1, dim):
        for v in range(u + 1, dim):
            u_arr[idx] = u
            v_arr[idx] = v
            c_val = calor[u, v]
            if afinidade_termica:
                d_val = D[u, v]
                if d_val < 0.001:
                    d_val = 0.001
                if gamma == 1.0:
                    sc = c_val / d_val
                else:
                    sc = c_val / (d_val ** gamma)
            else:
                sc = c_val
            neg_scores[idx] = -sc
            idx += 1

    order = np.argsort(neg_scores, kind="mergesort")

    deg = np.zeros(dim, dtype=np.int32)
    adj = np.zeros((dim, 2), dtype=np.int32)
    parent = np.arange(dim, dtype=np.int32)

    arestas_adicionadas = 0
    for k in range(num_edges):
        e_idx = order[k]
        u = u_arr[e_idx]
        v = v_arr[e_idx]
        if deg[u] < 2 and deg[v] < 2:
            # Find com path compression
            ru = u
            while parent[ru] != ru:
                ru = parent[ru]
            curr = u
            while curr != ru:
                nxt = parent[curr]
                parent[curr] = ru
                curr = nxt

            rv = v
            while parent[rv] != rv:
                rv = parent[rv]
            curr = v
            while curr != rv:
                nxt = parent[curr]
                parent[curr] = rv
                curr = nxt

            if ru != rv or arestas_adicionadas == n - 1:
                adj[u, deg[u]] = v
                deg[u] += 1
                adj[v, deg[v]] = u
                deg[v] += 1
                parent[ru] = rv
                arestas_adicionadas += 1
                if arestas_adicionadas == n:
                    break

    rota = np.empty(n, dtype=np.int32)
    rota[0] = 1
    atual = 1
    ant = -1
    for step in range(1, n):
        v0 = adj[atual, 0]
        v1 = adj[atual, 1]
        prox = v0 if v0 != ant else v1
        rota[step] = prox
        ant = atual
        atual = prox

    return rota


@njit(cache=True)
def numba_otimizar_2opt(rota_arr: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Refinamento local 2-opt em código nativo Numba com equivalência exata."""
    n = len(rota_arr)
    melhor_rota = rota_arr.copy()
    if n <= 3:
        return melhor_rota

    melhorou = True
    while melhorou:
        melhorou = False
        for i in range(1, n - 1):
            u1 = melhor_rota[i - 1]
            v1 = melhor_rota[i]
            d_u1_v1 = D[u1, v1]
            for j in range(i + 1, n):
                u2 = melhor_rota[j]
                v2 = melhor_rota[(j + 1) % n]
                d_atual = d_u1_v1 + D[u2, v2]
                d_nova = D[u1, u2] + D[v1, v2]
                if d_nova < d_atual - 1e-6:
                    left = i
                    right = j
                    while left < right:
                        tmp = melhor_rota[left]
                        melhor_rota[left] = melhor_rota[right]
                        melhor_rota[right] = tmp
                        left += 1
                        right -= 1
                    melhorou = True
                    break
            if melhorou:
                break
    return melhor_rota


@njit(cache=True)
def numba_caminho_fixo_bb(A: int, B: int, cidades: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Branch and Bound exato em código nativo Numba para otimização de caminho com
    extremidades fixas A e B (usado em otimizar_trechos_vazios para trechos <= 8 nós).
    """
    n = len(cidades)
    melhor_ordem = cidades.copy()
    if n <= 1:
        return melhor_ordem

    nodes = np.empty(n + 2, dtype=np.int32)
    nodes[0] = A
    for i in range(n):
        nodes[i + 1] = cidades[i]
    nodes[n + 1] = B

    dist_matrix = np.empty((n + 2, n + 2), dtype=np.float64)
    for i in range(n + 2):
        u = nodes[i]
        for j in range(n + 2):
            dist_matrix[i, j] = D[u, nodes[j]]

    d_orig = dist_matrix[0, 1] + dist_matrix[n, n + 1]
    for i in range(1, n):
        d_orig += dist_matrix[i, i + 1]
    melhor_dist = d_orig

    min_out = np.zeros(n + 1, dtype=np.float64)
    for i in range(1, n + 1):
        m_val = 1.0e300
        for j in range(1, n + 2):
            if j != i and dist_matrix[i, j] < m_val:
                m_val = dist_matrix[i, j]
        min_out[i] = m_val

    # Pilha iterativa para o DFS Branch-and-Bound
    # Em cada profundidade depth (0..n), guardamos curr_idx, mask, dist_acc, next_j_to_try
    st_curr = np.empty(n + 1, dtype=np.int32)
    st_mask = np.empty(n + 1, dtype=np.int32)
    st_dist = np.empty(n + 1, dtype=np.float64)
    st_next_j = np.empty(n + 1, dtype=np.int32)
    path = np.empty(n, dtype=np.int32)

    top = 0
    st_curr[0] = 0
    st_mask[0] = (1 << n) - 1
    st_dist[0] = 0.0
    st_next_j[0] = 1

    while top >= 0:
        curr_idx = st_curr[top]
        unvisited_mask = st_mask[top]
        dist_acc = st_dist[top]
        j_start = st_next_j[top]

        if unvisited_mask == 0:
            total = dist_acc + dist_matrix[curr_idx, n + 1]
            if total < melhor_dist - 1e-6:
                melhor_dist = total
                for k in range(n):
                    melhor_ordem[k] = path[k]
            top -= 1
            continue

        if j_start == 1:
            # Avalia Lower Bound na primeira entrada neste nó
            min_from_curr = 1.0e300
            sum_min_out = 0.0
            for j in range(1, n + 1):
                if unvisited_mask & (1 << (j - 1)):
                    d_cj = dist_matrix[curr_idx, j]
                    if d_cj < min_from_curr:
                        min_from_curr = d_cj
                    sum_min_out += min_out[j]
            if dist_acc + min_from_curr + sum_min_out >= melhor_dist - 1e-6:
                top -= 1
                continue

        avancou = False
        for j in range(j_start, n + 1):
            bit = 1 << (j - 1)
            if unvisited_mask & bit:
                d = dist_acc + dist_matrix[curr_idx, j]
                if d >= melhor_dist - 1e-6:
                    continue
                st_next_j[top] = j + 1
                path[top] = nodes[j]
                top += 1
                st_curr[top] = j
                st_mask[top] = unvisited_mask ^ bit
                st_dist[top] = d
                st_next_j[top] = 1
                avancou = True
                break

        if not avancou:
            top -= 1

    return melhor_ordem


@njit(cache=True)
def numba_caminho_fixo_2opt(A: int, B: int, cidades: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Busca local 2-opt + 1-opt (Or-Opt) em código nativo Numba para extremidades fixas A e B.
    """
    m = len(cidades)
    melhor = cidades.copy()
    if m <= 1:
        return melhor

    sub = np.empty(m - 1, dtype=np.int32)
    melhorou = True
    while melhorou:
        melhorou = False
        # 1. Movimentos 2-opt
        for i in range(m):
            u1 = A if i == 0 else melhor[i - 1]
            v1 = melhor[i]
            d_u1_v1 = D[u1, v1]
            for j in range(i + 1, m):
                u2 = melhor[j]
                v2 = B if j == m - 1 else melhor[j + 1]
                d_atual = d_u1_v1 + D[u2, v2]
                d_nova = D[u1, u2] + D[v1, v2]
                if d_nova < d_atual - 1e-6:
                    left = i
                    right = j
                    while left < right:
                        tmp = melhor[left]
                        melhor[left] = melhor[right]
                        melhor[right] = tmp
                        left += 1
                        right -= 1
                    melhorou = True
                    break
            if melhorou:
                break

        if melhorou:
            continue

        # 2. Movimentos de Relocação 1-opt (Or-Opt)
        for i in range(m):
            cid = melhor[i]
            pred = A if i == 0 else melhor[i - 1]
            succ = B if i == m - 1 else melhor[i + 1]
            custo_remover = D[pred, succ] - (D[pred, cid] + D[cid, succ])

            idx_s = 0
            for k in range(m):
                if k != i:
                    sub[idx_s] = melhor[k]
                    idx_s += 1

            melhor_insercao = -1
            maior_economia = custo_remover

            for pos in range(m):
                p_pred = A if pos == 0 else sub[pos - 1]
                p_succ = B if pos == m - 1 else sub[pos]
                custo_inserir = D[p_pred, cid] + D[cid, p_succ] - D[p_pred, p_succ]
                delta_total = custo_remover + custo_inserir
                if delta_total < maior_economia - 1e-6:
                    maior_economia = delta_total
                    melhor_insercao = pos

            if melhor_insercao != -1:
                for k in range(melhor_insercao):
                    melhor[k] = sub[k]
                melhor[melhor_insercao] = cid
                for k in range(melhor_insercao, m - 1):
                    melhor[k + 1] = sub[k]
                melhorou = True
                break

    return melhor


# ==============================================================================
# 5. KERNELS NUMBA JIT PARA DROP & ADD (PACK ITERATIVE / REFINAMENTO DE MOCHILA)
# ==============================================================================

@njit(cache=True)
def numba_refinar_ttp1(
    distancias: np.ndarray,
    item_cidx: np.ndarray,
    item_peso: np.ndarray,
    item_valor: np.ndarray,
    coletado: np.ndarray,
    cap_mochila: float,
    v_min: float,
    v_max: float,
    taxa_aluguel: float,
    max_iter: int,
) -> np.ndarray:
    """
    Busca local Drop & Add (1-bit-flip) para o TTP1 compilada com Numba + Programação Dinâmica
    por somas de sufixo de tempo (Suffix Time Sums) e poda por limite superior de valor.
    Reduz a complexidade de cada avaliação marginal de O(M + N) para O(N - idx_cidade),
    poupando 100% do prefixo da rota anterior à cidade do item e podando itens cujo valor bruto
    não supera o maior ganho já encontrado na iteração.
    """
    n_cidades = len(distancias)
    n_itens = len(item_peso)
    fator_vel = (v_max - v_min) / max(cap_mochila, 1e-6)

    peso_por_cidade = np.zeros(n_cidades, dtype=np.float64)
    peso_atual = 0.0
    for k in range(n_itens):
        if coletado[k]:
            c = item_cidx[k]
            if c >= 0:
                peso_por_cidade[c] += item_peso[k]
                peso_atual += item_peso[k]

    peso_acc_arr = np.zeros(n_cidades, dtype=np.float64)
    tempo_sufixo = np.zeros(n_cidades + 1, dtype=np.float64)

    for _ in range(max_iter):
        mudou = False

        # --- FASE DROP ---
        while True:
            p_acc = 0.0
            for i in range(n_cidades):
                p_acc += peso_por_cidade[i]
                peso_acc_arr[i] = p_acc

            t_suf = 0.0
            for i in range(n_cidades - 1, -1, -1):
                vel = v_max - peso_acc_arr[i] * fator_vel
                if vel < v_min:
                    vel = v_min
                t_suf += distancias[i] / vel
                tempo_sufixo[i] = t_suf

            melhor_k = -1
            maior_ganho_drop = 0.0

            for k in range(n_itens):
                if not coletado[k]:
                    continue
                c = item_cidx[k]
                p = item_peso[k]
                v = item_valor[k]

                t_novo_suf = 0.0
                for i in range(c, n_cidades):
                    vel = v_max - (peso_acc_arr[i] - p) * fator_vel
                    if vel < v_min:
                        vel = v_min
                    t_novo_suf += distancias[i] / vel

                delta_tempo = t_novo_suf - tempo_sufixo[c]
                ganho = -v - taxa_aluguel * delta_tempo
                if ganho > maior_ganho_drop:
                    maior_ganho_drop = ganho
                    melhor_k = k

            if melhor_k != -1 and maior_ganho_drop > 1e-6:
                coletado[melhor_k] = 0
                peso_por_cidade[item_cidx[melhor_k]] -= item_peso[melhor_k]
                peso_atual -= item_peso[melhor_k]
                mudou = True
            else:
                break

        # --- FASE ADD ---
        while True:
            p_acc = 0.0
            for i in range(n_cidades):
                p_acc += peso_por_cidade[i]
                peso_acc_arr[i] = p_acc

            t_suf = 0.0
            for i in range(n_cidades - 1, -1, -1):
                vel = v_max - peso_acc_arr[i] * fator_vel
                if vel < v_min:
                    vel = v_min
                t_suf += distancias[i] / vel
                tempo_sufixo[i] = t_suf

            melhor_k = -1
            maior_ganho_add = 0.0

            for k in range(n_itens):
                if coletado[k]:
                    continue
                p = item_peso[k]
                if peso_atual + p > cap_mochila:
                    continue
                v = item_valor[k]
                # Poda exata: adicionar peso nunca diminui o tempo de viagem, logo ganho <= v
                if v <= maior_ganho_add:
                    continue
                c = item_cidx[k]

                t_novo_suf = 0.0
                for i in range(c, n_cidades):
                    vel = v_max - (peso_acc_arr[i] + p) * fator_vel
                    if vel < v_min:
                        vel = v_min
                    t_novo_suf += distancias[i] / vel

                delta_tempo = t_novo_suf - tempo_sufixo[c]
                ganho = v - taxa_aluguel * delta_tempo
                if ganho > maior_ganho_add:
                    maior_ganho_add = ganho
                    melhor_k = k

            if melhor_k != -1 and maior_ganho_add > 1e-6:
                coletado[melhor_k] = 1
                peso_por_cidade[item_cidx[melhor_k]] += item_peso[melhor_k]
                peso_atual += item_peso[melhor_k]
                mudou = True
            else:
                break

        if not mudou:
            break

    return coletado


@njit(cache=True)
def _numba_avaliar_ttp2_estado(
    distancias: np.ndarray,
    peso_por_cidade: np.ndarray,
    coletado: np.ndarray,
    item_cidx: np.ndarray,
    item_valor: np.ndarray,
    v_min: float,
    v_max: float,
    fator_vel: float,
    const_deg: float,
    tempo_coleta_cidade: np.ndarray,
) -> float:
    n_cidades = len(distancias)
    n_itens = len(item_valor)
    tempo_viagem = 0.0
    peso_acc = 0.0

    for i in range(n_cidades):
        tempo_coleta_cidade[i] = tempo_viagem
        peso_acc += peso_por_cidade[i]
        vel = v_max - peso_acc * fator_vel
        if vel < v_min:
            vel = v_min
        tempo_viagem += distancias[i] / vel

    tempo_fim = tempo_viagem
    valor_degradado_total = 0.0
    for k in range(n_itens):
        if coletado[k]:
            idx = item_cidx[k]
            if idx >= 0:
                tempo_decorrido = tempo_fim - tempo_coleta_cidade[idx]
                degradacao = math.ceil(tempo_decorrido / const_deg)
                valor_degradado_total += (item_valor[k] - degradacao)

    return valor_degradado_total


@njit(cache=True)
def numba_refinar_ttp2(
    distancias: np.ndarray,
    item_cidx: np.ndarray,
    item_peso: np.ndarray,
    item_valor: np.ndarray,
    coletado: np.ndarray,
    cap_mochila: float,
    v_min: float,
    v_max: float,
    const_deg: float,
    max_iter: int,
) -> np.ndarray:
    """Busca local Drop & Add compilada com Numba para o modelo TTP2."""
    n_cidades = len(distancias)
    n_itens = len(item_peso)
    fator_vel = (v_max - v_min) / max(cap_mochila, 1e-6)
    c_deg = max(const_deg, 1e-6)

    peso_por_cidade = np.zeros(n_cidades, dtype=np.float64)
    peso_atual = 0.0
    for k in range(n_itens):
        if coletado[k]:
            c = item_cidx[k]
            if c >= 0:
                peso_por_cidade[c] += item_peso[k]
                peso_atual += item_peso[k]

    tempo_coleta_cidade = np.zeros(n_cidades, dtype=np.float64)

    for _ in range(max_iter):
        mudou = False

        # --- FASE DROP ---
        while True:
            obj_atual = _numba_avaliar_ttp2_estado(
                distancias, peso_por_cidade, coletado, item_cidx, item_valor,
                v_min, v_max, fator_vel, c_deg, tempo_coleta_cidade
            )
            melhor_k = -1
            maior_ganho_drop = 0.0

            for k in range(n_itens):
                if not coletado[k]:
                    continue
                c = item_cidx[k]
                p = item_peso[k]
                coletado[k] = 0
                peso_por_cidade[c] -= p
                obj_sem = _numba_avaliar_ttp2_estado(
                    distancias, peso_por_cidade, coletado, item_cidx, item_valor,
                    v_min, v_max, fator_vel, c_deg, tempo_coleta_cidade
                )
                coletado[k] = 1
                peso_por_cidade[c] += p

                ganho = obj_sem - obj_atual
                if ganho > maior_ganho_drop:
                    maior_ganho_drop = ganho
                    melhor_k = k

            if melhor_k != -1 and maior_ganho_drop > 1e-6:
                coletado[melhor_k] = 0
                peso_por_cidade[item_cidx[melhor_k]] -= item_peso[melhor_k]
                peso_atual -= item_peso[melhor_k]
                mudou = True
            else:
                break

        # --- FASE ADD ---
        while True:
            obj_atual = _numba_avaliar_ttp2_estado(
                distancias, peso_por_cidade, coletado, item_cidx, item_valor,
                v_min, v_max, fator_vel, c_deg, tempo_coleta_cidade
            )
            melhor_k = -1
            maior_ganho_add = 0.0

            for k in range(n_itens):
                if coletado[k]:
                    continue
                p = item_peso[k]
                if peso_atual + p > cap_mochila:
                    continue
                if item_valor[k] <= maior_ganho_add:
                    continue
                c = item_cidx[k]
                coletado[k] = 1
                peso_por_cidade[c] += p
                obj_com = _numba_avaliar_ttp2_estado(
                    distancias, peso_por_cidade, coletado, item_cidx, item_valor,
                    v_min, v_max, fator_vel, c_deg, tempo_coleta_cidade
                )
                coletado[k] = 0
                peso_por_cidade[c] -= p

                ganho = obj_com - obj_atual
                if ganho > maior_ganho_add:
                    maior_ganho_add = ganho
                    melhor_k = k

            if melhor_k != -1 and maior_ganho_add > 1e-6:
                coletado[melhor_k] = 1
                peso_por_cidade[item_cidx[melhor_k]] += item_peso[melhor_k]
                peso_atual += item_peso[melhor_k]
                mudou = True
            else:
                break

        if not mudou:
            break

    return coletado
