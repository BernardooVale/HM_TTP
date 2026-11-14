"""
Operadores das heurísticas baseadas em 2-OPT do artigo Profit-Guided
(Namazi et al., SoCS 2019):
- Cálculo de rentabilidade r_i e função prefixo-mínimo Π_{c,p}(k)
- TwoOptMove (2-OPT Convencional descoordenado)
- PGCH (Profit Guided Coordination Heuristic)
- RandFlipOneItem (busca local aleatória na mochila com orçamento temporal)
"""

import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.core.fast_evaluator import FastTTPEvaluator
from src.core.models import Item, TTPInstance


def preparar_itens_ordenados_por_cidade(
    instance: TTPInstance,
) -> Tuple[Dict[int, List[Item]], Dict[int, float], float]:
    """
    Agrupa e pré-ordena os itens de cada cidade por rentabilidade decrescente
    (r_i = valor / peso, desempate por maior valor) e calcula o sentinela
    1 + max_j(r_j).
    """
    por_cidade: Dict[int, List[Item]] = {}
    razoes: Dict[int, float] = {}
    max_r = 0.0

    for it in instance.itens:
        if it.cidade_origem == 1:
            continue
        r = (it.valor / it.peso) if it.peso > 0 else float("inf")
        razoes[it.id] = r
        if np.isfinite(r) and r > max_r:
            max_r = r
        por_cidade.setdefault(it.cidade_origem, []).append(it)

    sentinela = 1.0 + max_r
    for c_id, lista in por_cidade.items():
        lista.sort(key=lambda it: (razoes[it.id], it.valor), reverse=True)

    return por_cidade, razoes, sentinela


def calcular_prefixo_minimo(
    rota: List[int],
    plano_coleta: Dict[int, List[int]],
    razoes: Dict[int, float],
    sentinela: float,
) -> List[float]:
    """
    Calcula o vetor Π_{c,p}[0..n-1] (0-indexado, com Π_{c,p}[0] = sentinela).
    """
    n = len(rota)
    pi_cp = [sentinela] * n
    if n <= 1:
        return pi_cp

    for k in range(1, n):
        cidade = rota[k]
        itens_cid = plano_coleta.get(cidade)
        if itens_cid:
            r_min = min(razoes.get(it_id, sentinela) for it_id in itens_cid)
        else:
            r_min = sentinela
        pi_cp[k] = min(pi_cp[k - 1], r_min)

    return pi_cp


def two_opt_convencional_move(
    rota: List[int],
    plano_coleta: Dict[int, List[int]],
    k1: int,
    k2: int,
) -> Tuple[List[int], Dict[int, List[int]]]:
    """
    2-OPT Convencional (Descoordenado): inverte o segmento rota[k1..k2]
    mantendo o plano de coleta estritamente congelado.
    """
    nova_rota = rota[:k1] + rota[k1 : k2 + 1][::-1] + rota[k2 + 1 :]
    novo_plano = {c: list(ids) for c, ids in plano_coleta.items() if ids}
    return nova_rota, novo_plano


def pgch_move(
    rota: List[int],
    plano_coleta: Dict[int, List[int]],
    k1: int,
    k2: int,
    pi_cp: List[float],
    itens_por_cidade: Dict[int, List[Item]],
    razoes: Dict[int, float],
) -> Tuple[List[int], Dict[int, List[int]]]:
    """
    Operador PGCH (Profit Guided Coordination Heuristic):
    1. Inverte o segmento rota[k1..k2].
    2. Unpicking (k = k1 .. k2): descarta itens coletados com r_i < Π_{c,p}(k).
    3. Picking (k = k2 .. k1): coleta itens com r_i >= Π_{c,p}(k) até o limite
       do peso descartado na fase anterior.
    """
    nova_rota = rota[:k1] + rota[k1 : k2 + 1][::-1] + rota[k2 + 1 :]
    novo_plano: Dict[int, List[int]] = {
        c: list(ids) for c, ids in plano_coleta.items() if ids
    }

    coletados_no_seg: Set[int] = set()
    for k in range(k1, k2 + 1):
        cid = nova_rota[k]
        for it_id in novo_plano.get(cid, ()):
            coletados_no_seg.add(it_id)

    peso_descartado = 0.0

    # Fase 2: Descarte (Unpicking) do início ao fim do segmento
    for k in range(k1, k2 + 1):
        cid = nova_rota[k]
        limiar = pi_cp[k]
        itens_cid = itens_por_cidade.get(cid)
        if not itens_cid or cid not in novo_plano:
            continue
        novos_ids = []
        for it in itens_cid:
            if it.id in coletados_no_seg:
                if razoes[it.id] < limiar:
                    coletados_no_seg.remove(it.id)
                    peso_descartado += it.peso
                else:
                    novos_ids.append(it.id)
        if novos_ids:
            novo_plano[cid] = novos_ids
        else:
            novo_plano.pop(cid, None)

    if peso_descartado <= 0.0:
        return nova_rota, novo_plano

    # Fase 3: Inclusão compensatória (Picking) do fim para o início do segmento
    peso_adicionado = 0.0
    for k in range(k2, k1 - 1, -1):
        if peso_adicionado >= peso_descartado:
            break
        cid = nova_rota[k]
        limiar = pi_cp[k]
        itens_cid = itens_por_cidade.get(cid)
        if not itens_cid:
            continue
        for it in itens_cid:
            if razoes[it.id] < limiar:
                # Lista está ordenada por rentabilidade decrescente
                break
            if it.id not in coletados_no_seg:
                if peso_adicionado + it.peso <= peso_descartado + 1e-9:
                    coletados_no_seg.add(it.id)
                    novo_plano.setdefault(cid, []).append(it.id)
                    peso_adicionado += it.peso

    return nova_rota, novo_plano


def rand_flip_one_item(
    instance: TTPInstance,
    rota: List[int],
    plano_coleta: Dict[int, List[int]],
    duration_seconds: float,
    rng: Optional[np.random.Generator] = None,
    evaluator: Optional[FastTTPEvaluator] = None,
    min_iters: int = 1,
) -> Tuple[Dict[int, List[int]], float, bool]:
    """
    Operador KPSolver RandFlipOneItem(): mantém a rota fixa e inverte
    aleatoriamente 1 item por vez, aceitando melhorias estritas, durante o
    orçamento temporal especificado.
    """
    if evaluator is None:
        evaluator = FastTTPEvaluator(instance, rota, modelo="TTP1")
    if rng is None:
        rng = np.random.default_rng()

    itens_validos = [it for it in instance.itens if it.cidade_origem != 1]
    coletados_init: Set[int] = {
        it_id for ids in plano_coleta.values() for it_id in ids
    }
    evaluator.inicializar_estado(coletados_init)
    z_atual = evaluator.calcular_lucro_atual()
    z_inicial = z_atual

    m = len(itens_validos)
    if m == 0:
        return {c: list(ids) for c, ids in plano_coleta.items() if ids}, z_atual, False

    cap = float(instance.capacidade_mochila)
    start = time.perf_counter()
    iters = 0

    while iters < min_iters or (time.perf_counter() - start < duration_seconds):
        it = itens_validos[int(rng.integers(0, m))]
        estava_coletado = it.id in evaluator.itens_coletados

        if estava_coletado:
            evaluator.remover_item(it.id)
            z_cand = evaluator.calcular_lucro_atual()
            if z_cand > z_atual + 1e-9:
                z_atual = z_cand
            else:
                evaluator.adicionar_item(it.id)
        else:
            if evaluator.peso_total_atual + it.peso <= cap:
                evaluator.adicionar_item(it.id)
                z_cand = evaluator.calcular_lucro_atual()
                if z_cand > z_atual + 1e-9:
                    z_atual = z_cand
                else:
                    evaluator.remover_item(it.id)

        iters += 1
        if duration_seconds <= 0.0 and iters >= min_iters:
            break

    itens_by_id = {it.id: it for it in itens_validos}
    novo_plano: Dict[int, List[int]] = {}
    for it_id in evaluator.itens_coletados:
        it = itens_by_id.get(it_id)
        if it is not None:
            novo_plano.setdefault(it.cidade_origem, []).append(it_id)

    melhorou = z_atual > z_inicial + 1e-9
    return novo_plano, z_atual, melhorou
