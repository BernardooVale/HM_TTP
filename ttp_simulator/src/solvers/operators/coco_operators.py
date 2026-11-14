"""
Operadores do solucionador CoCo (Namazi, Sanderson, Newton & Sattar):
- Cálculo dos perfis Prefixo-Mínimo Π(P_{c,p}, k), Posfixo-Máximo Ω(Q_{c,p}, k)
  e identificação dos Itens de Fronteira (Boundary Items)
- PGCH Estendido (uncollection via Π e collection via Ω)
- Boundary Bit-Flip Search (KPSolver)
"""

import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.core.fast_evaluator import FastTTPEvaluator
from src.core.models import Item, TTPInstance


def calcular_perfis_e_fronteira_coco(
    rota: List[int],
    plano_coleta: Dict[int, List[int]],
    itens_por_cidade: Dict[int, List[Item]],
    razoes: Dict[int, float],
    sentinela: float,
) -> Tuple[List[float], List[float], List[Item]]:
    """
    Calcula em O(n + m):
    - pi_p[0..n-1]: Prefixo-Mínimo Π(P_{c,p}, k)
    - omega_q[0..n]: Posfixo-Máximo Ω(Q_{c,p}, k) (com omega_q[n] = 0.0)
    - boundary_items: lista de itens de fronteira (coletados e não-coletados)
    """
    n = len(rota)
    pi_p = [sentinela] * n
    omega_q = [0.0] * (n + 1)
    if n <= 1:
        return pi_p, omega_q, []

    coletados: Set[int] = {it_id for ids in plano_coleta.values() for it_id in ids}
    p_seq = [sentinela] * n
    q_seq = [0.0] * n
    p_item: List[Optional[Item]] = [None] * n
    q_item: List[Optional[Item]] = [None] * n

    for k in range(1, n):
        cid = rota[k]
        itens_cid = itens_por_cidade.get(cid)
        if not itens_cid:
            continue
        # itens_cid está ordenado por rentabilidade decrescente:
        # - primeiro não-coletado = maior r_i (q(k))
        # - último coletado = menor r_i (p(k))
        for it in itens_cid:
            if it.id in coletados:
                p_item[k] = it
                p_seq[k] = razoes[it.id]
            elif q_item[k] is None:
                q_item[k] = it
                q_seq[k] = razoes[it.id]

    for k in range(1, n):
        pi_p[k] = min(pi_p[k - 1], p_seq[k])

    for k in range(n - 1, 0, -1):
        omega_q[k] = max(q_seq[k], omega_q[k + 1])

    boundary_ids: Set[int] = set()
    boundary_items: List[Item] = []

    for k in range(1, n):
        it_p = p_item[k]
        if it_p is not None and pi_p[k] == p_seq[k] and it_p.id not in boundary_ids:
            boundary_ids.add(it_p.id)
            boundary_items.append(it_p)

        it_q = q_item[k]
        if it_q is not None and omega_q[k] == q_seq[k] and it_q.id not in boundary_ids:
            boundary_ids.add(it_q.id)
            boundary_items.append(it_q)

    return pi_p, omega_q, boundary_items


def pgch_estendido_move(
    rota: List[int],
    plano_coleta: Dict[int, List[int]],
    k1: int,
    k2: int,
    pi_p: List[float],
    omega_q: List[float],
    itens_por_cidade: Dict[int, List[Item]],
    razoes: Dict[int, float],
) -> Tuple[List[int], Dict[int, List[int]]]:
    """
    PGCH Estendido do CoCo:
    1. Inverte o segmento rota[k1..k2].
    2. Uncollection (k = k1 .. k2): remove itens coletados com r_j < Π(P_{c,p}, k).
    3. Collection (k = k2 .. k1): coleta itens não coletados com r_j > Ω(Q_{c,p}, k)
       respeitando W_collected <= W_uncollected.
    """
    nova_rota = rota[:k1] + rota[k1 : k2 + 1][::-1] + rota[k2 + 1 :]
    novo_plano: Dict[int, List[int]] = {
        c: list(ids) for c, ids in plano_coleta.items() if ids
    }

    coletados_no_seg: Set[int] = set()
    for k in range(k1, k2 + 1):
        for it_id in novo_plano.get(nova_rota[k], ()):
            coletados_no_seg.add(it_id)

    w_uncollected = 0.0

    # Fase 2: Uncollection (início -> fim do segmento)
    for k in range(k1, k2 + 1):
        cid = nova_rota[k]
        limiar_pi = pi_p[k]
        itens_cid = itens_por_cidade.get(cid)
        if not itens_cid or cid not in novo_plano:
            continue
        restantes = []
        for it in itens_cid:
            if it.id in coletados_no_seg:
                if razoes[it.id] < limiar_pi:
                    coletados_no_seg.remove(it.id)
                    w_uncollected += it.peso
                else:
                    restantes.append(it.id)
        if restantes:
            novo_plano[cid] = restantes
        else:
            novo_plano.pop(cid, None)

    if w_uncollected <= 0.0:
        return nova_rota, novo_plano

    # Fase 3: Collection (fim -> início do segmento) usando Ω(Q_{c,p}, k)
    w_collected = 0.0
    for k in range(k2, k1 - 1, -1):
        if w_collected >= w_uncollected:
            break
        cid = nova_rota[k]
        limiar_omega = omega_q[k]
        itens_cid = itens_por_cidade.get(cid)
        if not itens_cid:
            continue
        for it in itens_cid:
            if razoes[it.id] <= limiar_omega:
                # Ordenado por rentabilidade decrescente
                break
            if it.id not in coletados_no_seg:
                if w_collected + it.peso <= w_uncollected + 1e-9:
                    coletados_no_seg.add(it.id)
                    novo_plano.setdefault(cid, []).append(it.id)
                    w_collected += it.peso

    return nova_rota, novo_plano


def boundary_bit_flip(
    instance: TTPInstance,
    rota: List[int],
    plano_coleta: Dict[int, List[int]],
    itens_por_cidade: Dict[int, List[Item]],
    razoes: Dict[int, float],
    sentinela: float,
    rng: Optional[np.random.Generator] = None,
    time_limit: Optional[float] = None,
    start_time: Optional[float] = None,
    evaluator: Optional[FastTTPEvaluator] = None,
) -> Tuple[Dict[int, List[int]], float, bool]:
    """
    KPSolver do CoCo (Boundary Bit-Flip Search):
    Avalia aleatoriamente itens da bolsa de fronteira b até atingir ótimo local
    (todos os itens de b avaliados sem melhora) ou esgotar o time_limit.
    """
    if evaluator is None:
        evaluator = FastTTPEvaluator(instance, rota, modelo="TTP1")
    if rng is None:
        rng = np.random.default_rng()

    novo_plano: Dict[int, List[int]] = {
        c: list(ids) for c, ids in plano_coleta.items() if ids
    }
    coletados_init: Set[int] = {it_id for ids in novo_plano.values() for it_id in ids}
    evaluator.inicializar_estado(coletados_init)
    z_atual = evaluator.calcular_lucro_atual()
    cap = float(instance.capacidade_mochila)
    melhorou_alguma_vez = False

    _, _, bolsa = calcular_perfis_e_fronteira_coco(
        rota, novo_plano, itens_por_cidade, razoes, sentinela
    )

    while bolsa:
        if time_limit is not None and start_time is not None:
            if time.perf_counter() - start_time >= time_limit:
                break

        ordem = rng.permutation(len(bolsa))
        aceitou = False

        for idx in ordem:
            if time_limit is not None and start_time is not None:
                if time.perf_counter() - start_time >= time_limit:
                    break

            it = bolsa[int(idx)]
            cid = it.cidade_origem
            estava_coletado = it.id in evaluator.itens_coletados

            if estava_coletado:
                evaluator.remover_item(it.id)
                z_cand = evaluator.calcular_lucro_atual()
                if z_cand > z_atual + 1e-9:
                    z_atual = z_cand
                    novo_plano[cid].remove(it.id)
                    if not novo_plano[cid]:
                        novo_plano.pop(cid, None)
                    melhorou_alguma_vez = True
                    aceitou = True
                    break
                else:
                    evaluator.adicionar_item(it.id)
            else:
                if evaluator.peso_total_atual + it.peso <= cap:
                    evaluator.adicionar_item(it.id)
                    z_cand = evaluator.calcular_lucro_atual()
                    if z_cand > z_atual + 1e-9:
                        z_atual = z_cand
                        novo_plano.setdefault(cid, []).append(it.id)
                        melhorou_alguma_vez = True
                        aceitou = True
                        break
                    else:
                        evaluator.remover_item(it.id)

        if aceitou:
            _, _, bolsa = calcular_perfis_e_fronteira_coco(
                rota, novo_plano, itens_por_cidade, razoes, sentinela
            )
        else:
            break

    return novo_plano, z_atual, melhorou_alguma_vez
