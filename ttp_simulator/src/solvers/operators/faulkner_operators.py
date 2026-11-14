"""
Operadores modulares das heurísticas S1–S5 e C1–C6 de Faulkner et al. (2015)
para o Traveling Thief Problem (TTP1):
- Pack(Π, α)
- PackIterative(Π, c, δ, q)
- BitFlip(Π, P) com vetor de pesos por cidade e atualização diferencial por sufixo
- (1+1)-EA(Π, P)
- Insertion(Π, P) com vetor de pesos por cidade e recálculo parcial de segmento
"""

import math
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.core.fast_evaluator import FastTTPEvaluator
from src.core.models import TTPInstance


def _calcular_distancias_restantes_array(
    instance: TTPInstance, rota: List[int]
) -> Dict[int, float]:
    """Calcula a distância restante de cada cidade na rota até o retorno à cidade 1."""
    n = len(rota)
    if n == 0:
        return {}
    dist_mtx = getattr(instance.grafo, "_distance_matrix", None)
    dist_restante: Dict[int, float] = {}
    if dist_mtx is not None:
        dist_acc = float(dist_mtx[rota[-1], rota[0]])
        dist_restante[rota[-1]] = dist_acc
        for i in range(n - 2, -1, -1):
            dist_acc += float(dist_mtx[rota[i], rota[i + 1]])
            dist_restante[rota[i]] = dist_acc
    else:
        dist_acc = float(instance.grafo.distancia(rota[-1], rota[0]))
        dist_restante[rota[-1]] = dist_acc
        for i in range(n - 2, -1, -1):
            dist_acc += float(instance.grafo.distancia(rota[i], rota[i + 1]))
            dist_restante[rota[i]] = dist_acc
    return dist_restante


def _itens_para_plano(
    instance: TTPInstance, itens_ids: Set[int]
) -> Dict[int, List[int]]:
    """Converte um conjunto de IDs de itens para o formato plano_coleta por cidade."""
    itens_by_id = {it.id: it for it in instance.itens}
    plano: Dict[int, List[int]] = {}
    for it_id in itens_ids:
        it = itens_by_id.get(it_id)
        if it is not None:
            plano.setdefault(it.cidade_origem, []).append(it_id)
    return plano


def _plano_para_itens(plano_coleta: Dict[int, List[int]]) -> Set[int]:
    """Extrai o conjunto de IDs de itens a partir do dicionário plano_coleta."""
    return {it_id for ids in plano_coleta.values() for it_id in ids}


def pack(
    instance: TTPInstance,
    rota: List[int],
    alpha: float,
    r: int = 10,
    evaluator: Optional[FastTTPEvaluator] = None,
) -> Tuple[Dict[int, List[int]], float]:
    """
    Procedimento Pack(Π, α) de Faulkner et al. (2015).

    Ordena os itens por s_ik = (p_ik / w_ik)^α / d_i e os insere em lotes
    de tamanho μ = ceil(m / r), revertendo e dividindo μ por 2 quando Z piora.
    """
    if evaluator is None:
        evaluator = FastTTPEvaluator(instance, rota, modelo="TTP1")

    itens_validos = [it for it in instance.itens if it.cidade_origem != 1]
    m = len(itens_validos)
    if m == 0:
        return {}, evaluator.calcular_lucro(set())

    dist_restante = _calcular_distancias_restantes_array(instance, rota)

    def _score(it) -> float:
        d_i = max(dist_restante.get(it.cidade_origem, 1.0), 1e-9)
        if it.peso <= 0:
            return float("inf")
        ratio = it.valor / it.peso
        if ratio <= 0:
            return 0.0
        try:
            log_val = alpha * math.log(ratio)
            if log_val > 700.0:
                return float("inf")
            if log_val < -700.0:
                return 0.0
            return math.exp(log_val) / d_i
        except (ValueError, OverflowError):
            return float("inf") if alpha >= 0 else 0.0

    itens_ordenados = sorted(itens_validos, key=_score, reverse=True)

    mu = max(1, math.ceil(m / max(1, r)))
    cap = instance.capacidade_mochila

    p_atual: List[int] = []
    peso_atual = 0.0
    p_melhor: List[int] = []
    peso_melhor = 0.0
    z_melhor = evaluator.calcular_lucro(set())
    k = 0
    k_melhor = 0
    avaliado_no_fim = True

    while k < m and peso_atual < cap and (mu > 1 or m <= r):
        it = itens_ordenados[k]
        if peso_atual + it.peso <= cap:
            p_atual.append(it.id)
            peso_atual += it.peso
            avaliado_no_fim = False
            if (k + 1) % mu == 0:
                z = evaluator.calcular_lucro(set(p_atual))
                avaliado_no_fim = True
                if z < z_melhor:
                    p_atual = list(p_melhor)
                    peso_atual = peso_melhor
                    k = k_melhor
                    mu = mu // 2
                    if mu <= 1 and m > r:
                        break
                else:
                    p_melhor = list(p_atual)
                    peso_melhor = peso_atual
                    k_melhor = k + 1
                    z_melhor = z
        k += 1

    if not avaliado_no_fim:
        z = evaluator.calcular_lucro(set(p_atual))
        if z > z_melhor:
            p_melhor = list(p_atual)
            z_melhor = z

    return _itens_para_plano(instance, set(p_melhor)), z_melhor


def pack_iterative(
    instance: TTPInstance,
    rota: List[int],
    c: float = 5.0,
    delta: float = 2.5,
    q: int = 20,
    eps: float = 0.1,
    r: int = 10,
    evaluator: Optional[FastTTPEvaluator] = None,
) -> Tuple[Dict[int, List[int]], float]:
    """
    Procedimento PackIterative(Π, c, δ, q) de Faulkner et al. (2015).

    Busca ternária intervalar sobre o expoente α usando Pack(Π, α).
    """
    if evaluator is None:
        evaluator = FastTTPEvaluator(instance, rota, modelo="TTP1")

    p_l, z_l = pack(instance, rota, c - delta, r=r, evaluator=evaluator)
    p_m, z_m = pack(instance, rota, c, r=r, evaluator=evaluator)
    p_r, z_r = pack(instance, rota, c + delta, r=r, evaluator=evaluator)

    p_best, z_best = p_m, z_m
    if z_l > z_best:
        p_best, z_best = p_l, z_l
    if z_r > z_best:
        p_best, z_best = p_r, z_r

    for _ in range(q):
        if z_l > z_m and z_r > z_m:
            if z_l > z_r:
                z_m = z_l
                c = c - delta
                p_best, z_best = p_l, z_l
            else:
                z_m = z_r
                c = c + delta
                p_best, z_best = p_r, z_r
        elif z_l > z_m:
            z_m = z_l
            c = c - delta
            p_best, z_best = p_l, z_l
        elif z_r > z_m:
            z_m = z_r
            c = c + delta
            p_best, z_best = p_r, z_r

        delta = delta / 2.0
        p_l, z_l = pack(instance, rota, c - delta, r=r, evaluator=evaluator)
        p_r, z_r = pack(instance, rota, c + delta, r=r, evaluator=evaluator)

        if z_l > z_best:
            p_best, z_best = p_l, z_l
        if z_r > z_best:
            p_best, z_best = p_r, z_r

        if abs(z_l - z_m) < eps and abs(z_r - z_m) < eps:
            break

    return p_best, z_best


def bit_flip(
    instance: TTPInstance,
    rota: List[int],
    plano_coleta: Dict[int, List[int]],
    max_passes: Optional[int] = None,
    time_limit: Optional[float] = None,
    start_time: Optional[float] = None,
) -> Tuple[Dict[int, List[int]], float, bool]:
    """
    Operador determinístico BitFlip(Π, P) (hill-climbing sequencial sobre os itens).

    Usa vetor de pesos por cidade e atualização diferencial do sufixo da rota
    O(N - c) por avaliação.
    - max_passes=1: executa exatamente uma passagem completa (usado em C1/C2).
    - max_passes=None: repete até convergir (usado em S2).
    """
    n = len(rota)
    if n == 0 or not instance.itens:
        return {}, 0.0, False

    cidade_idx = {rota[i]: i for i in range(n)}
    dist_mtx = getattr(instance.grafo, "_distance_matrix", None)
    if dist_mtx is not None:
        distancias = np.array(
            [dist_mtx[rota[i], rota[(i + 1) % n]] for i in range(n)],
            dtype=np.float64,
        )
    else:
        distancias = np.array(
            [instance.grafo.distancia(rota[i], rota[(i + 1) % n]) for i in range(n)],
            dtype=np.float64,
        )

    cap = max(float(instance.capacidade_mochila), 1e-6)
    v_min = float(instance.v_min)
    v_max = float(instance.v_max)
    taxa_aluguel = float(instance.taxa_aluguel)
    fator_vel = (v_max - v_min) / cap

    itens_coletados = _plano_para_itens(plano_coleta)
    m = len(instance.itens)
    coletado = np.zeros(m, dtype=bool)
    item_cidx = np.empty(m, dtype=np.int32)
    item_peso = np.empty(m, dtype=np.float64)
    item_valor = np.empty(m, dtype=np.float64)

    peso_por_cidade = np.zeros(n, dtype=np.float64)
    peso_atual = 0.0
    valor_bruto = 0.0

    for k, it in enumerate(instance.itens):
        c = cidade_idx.get(it.cidade_origem, -1)
        item_cidx[k] = c
        item_peso[k] = it.peso
        item_valor[k] = it.valor
        if it.id in itens_coletados and c > 0:
            coletado[k] = True
            peso_por_cidade[c] += it.peso
            peso_atual += it.peso
            valor_bruto += it.valor

    peso_acc = np.cumsum(peso_por_cidade)
    tempo_sufixo = np.zeros(n + 1, dtype=np.float64)

    def _recomputar_sufixo(inicio: int = 0) -> None:
        t_suf = tempo_sufixo[n]
        for i in range(n - 1, -1, -1):
            vel = v_max - peso_acc[i] * fator_vel
            if vel < v_min:
                vel = v_min
            t_suf += distancias[i] / vel
            tempo_sufixo[i] = t_suf

    _recomputar_sufixo()
    z_atual = valor_bruto - taxa_aluguel * tempo_sufixo[0]
    melhorou_alguma_vez = False

    passes = 0
    while max_passes is None or passes < max_passes:
        if time_limit is not None and start_time is not None:
            if time.perf_counter() - start_time >= time_limit:
                break

        mudou_na_passagem = False
        for k in range(m):
            c = int(item_cidx[k])
            if c <= 0:
                continue

            p = float(item_peso[k])
            v = float(item_valor[k])

            if coletado[k]:
                dp = -p
                dv = -v
            else:
                if peso_atual + p > instance.capacidade_mochila:
                    continue
                dp = p
                dv = v

            # Recalcula apenas o sufixo [c .. n-1]
            vels = v_max - (peso_acc[c:] + dp) * fator_vel
            np.maximum(vels, v_min, out=vels)
            t_novo_suf = float(np.sum(distancias[c:] / vels))
            delta_tempo = t_novo_suf - tempo_sufixo[c]
            delta_z = dv - taxa_aluguel * delta_tempo

            if delta_z > 1e-9:
                coletado[k] = not coletado[k]
                peso_por_cidade[c] += dp
                peso_atual += dp
                valor_bruto += dv
                z_atual += delta_z
                peso_acc[c:] += dp
                _recomputar_sufixo()
                mudou_na_passagem = True
                melhorou_alguma_vez = True

        passes += 1
        if not mudou_na_passagem:
            break

    novo_plano: Dict[int, List[int]] = {}
    for k in range(m):
        if coletado[k]:
            it = instance.itens[k]
            novo_plano.setdefault(it.cidade_origem, []).append(it.id)

    return novo_plano, z_atual, melhorou_alguma_vez


def one_plus_one_ea(
    instance: TTPInstance,
    rota: List[int],
    plano_coleta: Dict[int, List[int]],
    max_iters: Optional[int] = None,
    time_limit: Optional[float] = None,
    start_time: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    evaluator: Optional[FastTTPEvaluator] = None,
) -> Tuple[Dict[int, List[int]], float, bool]:
    """
    Operador estocástico (1+1)-EA(Π, P) de Faulkner et al. (2015).

    Inverte cada item com probabilidade p = 1/m, descarta mutações inviáveis
    e aceita se Z(Π, P') >= Z(Π, P).
    """
    if evaluator is None:
        evaluator = FastTTPEvaluator(instance, rota, modelo="TTP1")
    if rng is None:
        rng = np.random.default_rng()

    itens_validos = [it for it in instance.itens if it.cidade_origem != 1]
    m = len(itens_validos)
    itens_coletados = _plano_para_itens(plano_coleta)
    z_inicial = evaluator.calcular_lucro(itens_coletados)
    z_atual = z_inicial

    if m == 0:
        return dict(plano_coleta), z_atual, False

    pesos = np.array([it.peso for it in itens_validos], dtype=np.float64)
    ids = [it.id for it in itens_validos]
    estado = np.array([it_id in itens_coletados for it_id in ids], dtype=bool)
    cap = float(instance.capacidade_mochila)
    prob = 1.0 / m

    iters = 0
    while max_iters is None or iters < max_iters:
        if time_limit is not None and start_time is not None:
            if (iters & 63) == 0 and time.perf_counter() - start_time >= time_limit:
                break

        flips = rng.random(m) < prob
        if flips.any():
            novo_estado = np.logical_xor(estado, flips)
            peso_cand = float(np.dot(novo_estado, pesos))
            if peso_cand <= cap:
                cand_ids = {ids[idx] for idx in np.flatnonzero(novo_estado)}
                z_cand = evaluator.calcular_lucro(cand_ids)
                if z_cand >= z_atual:
                    estado = novo_estado
                    z_atual = z_cand

        iters += 1

    novo_ids = {ids[idx] for idx in np.flatnonzero(estado)}
    melhorou = z_atual > z_inicial + 1e-9
    return _itens_para_plano(instance, novo_ids), z_atual, melhorou


def insertion_local_search(
    instance: TTPInstance,
    rota: List[int],
    plano_coleta: Dict[int, List[int]],
    max_passes: Optional[int] = None,
    time_limit: Optional[float] = None,
    start_time: Optional[float] = None,
) -> Tuple[List[int], float, bool]:
    """
    Operador de busca local na rota Insertion(Π, P) de Faulkner et al. (2015).

    Mantém a cidade 1 fixa na posição 0 e varre x_a em ordem reversa (n -> 2),
    testando realocações para posições anteriores b (a-1 -> 1) com recálculo
    parcial apenas do segmento afetado [b-1 .. a].
    """
    n = len(rota)
    if n <= 3:
        ev = FastTTPEvaluator(instance, rota, modelo="TTP1")
        return list(rota), ev.calcular_lucro(_plano_para_itens(plano_coleta)), False

    itens_coletados = _plano_para_itens(plano_coleta)
    itens_by_id = {it.id: it for it in instance.itens}
    valor_bruto = sum(itens_by_id[i].valor for i in itens_coletados if i in itens_by_id)
    peso_cidade_map: Dict[int, float] = {}
    for it_id in itens_coletados:
        it = itens_by_id.get(it_id)
        if it is not None:
            peso_cidade_map[it.cidade_origem] = peso_cidade_map.get(it.cidade_origem, 0.0) + it.peso

    cap = max(float(instance.capacidade_mochila), 1e-6)
    v_min = float(instance.v_min)
    v_max = float(instance.v_max)
    taxa_aluguel = float(instance.taxa_aluguel)
    fator_vel = (v_max - v_min) / cap
    dist_mtx = getattr(instance.grafo, "_distance_matrix", None)

    def _dist(u: int, v: int) -> float:
        if dist_mtx is not None:
            return float(dist_mtx[u, v])
        return float(instance.grafo.distancia(u, v))

    rota_atual = list(rota)
    melhorou_alguma_vez = False

    def _precomputar(r_seq: List[int]):
        w_acc = [0.0] * n
        p_time = [0.0] * (n + 1)
        s_time = [0.0] * (n + 1)
        acc = 0.0
        t_acc = 0.0
        for i in range(n):
            acc += peso_cidade_map.get(r_seq[i], 0.0)
            w_acc[i] = acc
            vel = max(v_min, v_max - acc * fator_vel)
            t_acc += _dist(r_seq[i], r_seq[(i + 1) % n]) / vel
            p_time[i + 1] = t_acc
        for i in range(n - 1, -1, -1):
            vel = max(v_min, v_max - w_acc[i] * fator_vel)
            s_time[i] = s_time[i + 1] + _dist(r_seq[i], r_seq[(i + 1) % n]) / vel
        return w_acc, p_time, s_time

    w_acc, p_time, s_time = _precomputar(rota_atual)
    z_atual = valor_bruto - taxa_aluguel * p_time[n]

    passes = 0
    while max_passes is None or passes < max_passes:
        if time_limit is not None and start_time is not None:
            if time.perf_counter() - start_time >= time_limit:
                break

        mudou_na_passagem = False
        for a in range(n - 1, 1, -1):
            if time_limit is not None and start_time is not None:
                if time.perf_counter() - start_time >= time_limit:
                    break

            cidade_a = rota_atual[a]
            w_a = peso_cidade_map.get(cidade_a, 0.0)
            melhor_b = -1
            melhor_z_a = z_atual
            next_after_a = rota_atual[(a + 1) % n]

            for b in range(a - 1, 0, -1):
                # Segmento alterado: posições b-1 .. a
                t_seg = 0.0
                w_cur = w_acc[b - 1]

                # Perna b-1: rota[b-1] -> cidade_a
                vel = max(v_min, v_max - w_cur * fator_vel)
                t_seg += _dist(rota_atual[b - 1], cidade_a) / vel

                # Perna b: cidade_a -> rota[b]
                w_cur += w_a
                vel = max(v_min, v_max - w_cur * fator_vel)
                t_seg += _dist(cidade_a, rota_atual[b]) / vel

                # Pernas b+1 .. a: rota[b] -> ... -> rota[a-1] -> next_after_a
                for idx in range(b, a):
                    w_cur += peso_cidade_map.get(rota_atual[idx], 0.0)
                    vel = max(v_min, v_max - w_cur * fator_vel)
                    prox = rota_atual[idx + 1] if idx + 1 < a else next_after_a
                    t_seg += _dist(rota_atual[idx], prox) / vel

                tempo_total = p_time[b - 1] + t_seg + s_time[a + 1]
                z_cand = valor_bruto - taxa_aluguel * tempo_total

                if z_cand > melhor_z_a + 1e-9:
                    melhor_z_a = z_cand
                    melhor_b = b

            if melhor_b != -1:
                cidade_movida = rota_atual.pop(a)
                rota_atual.insert(melhor_b, cidade_movida)
                w_acc, p_time, s_time = _precomputar(rota_atual)
                z_atual = melhor_z_a
                mudou_na_passagem = True
                melhorou_alguma_vez = True

        passes += 1
        if not mudou_na_passagem:
            break

    return rota_atual, z_atual, melhorou_alguma_vez
