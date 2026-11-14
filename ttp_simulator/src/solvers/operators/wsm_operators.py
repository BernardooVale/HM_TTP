"""
Operadores do Weighted-Sum Method (WSM) de Chagas & Wagner (2021)
para o Bi-Objective Traveling Thief Problem (BITTP):
- Avaliação bi-objetivo (g(z), h(π, z)) e escalarização f(π, z, α)
- Gerenciamento de arquivo Pareto de soluções não-dominadas
- RandomizedPackingAlgorithm(π, ρ, α, γ)
- 2-OPT com filtro geométrico de estiramento (β)
- Bit-Flip amostrado (λ)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.core.models import Item, SolucaoTTP, TTPInstance


@dataclass
class ParetoEntry:
    """Entrada do arquivo de soluções não-dominadas do WSM."""

    rota: List[int]
    itens_ids: Set[int]
    lucro_bruto: float  # g(z)
    tempo_viagem: float  # h(π, z)

    def escalarizado(self, alpha: float, taxa_aluguel: float) -> float:
        return alpha * self.lucro_bruto - (1.0 - alpha) * taxa_aluguel * self.tempo_viagem

    def lucro_ttp1(self, taxa_aluguel: float) -> float:
        return self.lucro_bruto - taxa_aluguel * self.tempo_viagem


def avaliar_biobjetivo(
    instance: TTPInstance,
    rota: List[int],
    itens_ids: Set[int],
    itens_by_id: Optional[Dict[int, Item]] = None,
) -> Tuple[float, float]:
    """Calcula (g(z), h(π, z)) para uma rota e conjunto de itens."""
    n = len(rota)
    if n == 0:
        return 0.0, 0.0

    if itens_by_id is None:
        itens_by_id = {it.id: it for it in instance.itens}

    cidade_idx = {rota[i]: i for i in range(n)}
    peso_por_cidade = np.zeros(n, dtype=np.float64)
    g = 0.0

    for it_id in itens_ids:
        it = itens_by_id.get(it_id)
        if it is not None:
            g += it.valor
            c = cidade_idx.get(it.cidade_origem)
            if c is not None:
                peso_por_cidade[c] += it.peso

    cap = max(float(instance.capacidade_mochila), 1e-6)
    v_min = float(instance.v_min)
    v_max = float(instance.v_max)
    fator_vel = (v_max - v_min) / cap
    dist_mtx = getattr(instance.grafo, "_distance_matrix", None)

    peso_acc = 0.0
    h = 0.0
    for i in range(n):
        peso_acc += float(peso_por_cidade[i])
        vel = max(v_min, v_max - peso_acc * fator_vel)
        u, v = rota[i], rota[(i + 1) % n]
        d = float(dist_mtx[u, v]) if dist_mtx is not None else float(instance.grafo.distancia(u, v))
        h += d / vel

    return g, h


def atualizar_arquivo_pareto(
    arquivo: List[ParetoEntry],
    rota: List[int],
    itens_ids: Set[int],
    g: float,
    h: float,
) -> bool:
    """
    Atualiza o arquivo S mantendo apenas soluções não-dominadas
    (maximiza g, minimiza h). Retorna True se a nova solução foi inserida.
    """
    eps = 1e-9
    for entry in arquivo:
        if entry.lucro_bruto >= g - eps and entry.tempo_viagem <= h + eps:
            if (
                entry.lucro_bruto > g + eps
                or entry.tempo_viagem < h - eps
                or (abs(entry.lucro_bruto - g) <= eps and abs(entry.tempo_viagem - h) <= eps)
            ):
                return False

    nova_lista = [
        entry
        for entry in arquivo
        if not (g >= entry.lucro_bruto - eps and h <= entry.tempo_viagem + eps and (g > entry.lucro_bruto + eps or h < entry.tempo_viagem - eps))
    ]
    nova_lista.append(
        ParetoEntry(
            rota=list(rota),
            itens_ids=set(itens_ids),
            lucro_bruto=g,
            tempo_viagem=h,
        )
    )
    arquivo[:] = nova_lista
    return True


def selecionar_pivo(
    arquivo: List[ParetoEntry],
    alpha: float,
    taxa_aluguel: float,
) -> ParetoEntry:
    """Seleciona no arquivo S a solução que maximiza f(π, z, α)."""
    return max(arquivo, key=lambda e: e.escalarizado(alpha, taxa_aluguel))


def _distancias_restantes(instance: TTPInstance, rota: List[int]) -> Dict[int, float]:
    n = len(rota)
    if n == 0:
        return {}
    dist_mtx = getattr(instance.grafo, "_distance_matrix", None)
    d_rest: Dict[int, float] = {}
    d_acc = (
        float(dist_mtx[rota[-1], rota[0]])
        if dist_mtx is not None
        else float(instance.grafo.distancia(rota[-1], rota[0]))
    )
    d_rest[rota[-1]] = d_acc
    for i in range(n - 2, -1, -1):
        d_acc += (
            float(dist_mtx[rota[i], rota[i + 1]])
            if dist_mtx is not None
            else float(instance.grafo.distancia(rota[i], rota[i + 1]))
        )
        d_rest[rota[i]] = d_acc
    return d_rest


def randomized_packing_algorithm(
    instance: TTPInstance,
    rota: List[int],
    rho: int,
    alpha: float,
    gamma: float,
    rng: Optional[np.random.Generator] = None,
    itens_by_id: Optional[Dict[int, Item]] = None,
) -> Tuple[Set[int], float, float]:
    """
    Algoritmo 2 do WSM (Chagas & Wagner, 2021):
    Gera ρ planos usando expoentes aleatórios normalizados (a, b, c) e janela
    dinâmica de checagem φ = ceil((m / γ) * α + 1e-5).
    Retorna (melhor_itens_ids, g_best, h_best).
    """
    if rng is None:
        rng = np.random.default_rng()
    if itens_by_id is None:
        itens_by_id = {it.id: it for it in instance.itens}

    itens_validos = [it for it in instance.itens if it.cidade_origem != 1]
    m = len(itens_validos)
    g_vazio, h_vazio = avaliar_biobjetivo(instance, rota, set(), itens_by_id)
    if m == 0:
        return set(), g_vazio, h_vazio

    d_rest = _distancias_restantes(instance, rota)
    taxa_aluguel = float(instance.taxa_aluguel)
    cap = float(instance.capacidade_mochila)

    z_best: Set[int] = set()
    g_best, h_best = g_vazio, h_vazio
    f_best = alpha * g_best - (1.0 - alpha) * taxa_aluguel * h_best

    for _ in range(max(1, rho)):
        abc = rng.random(3)
        soma = float(np.sum(abc))
        if soma <= 1e-12:
            a, b, c = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
        else:
            a, b, c = float(abc[0] / soma), float(abc[1] / soma), float(abc[2] / soma)

        def _score(it: Item) -> float:
            p_val = max(it.valor, 1e-9)
            w_val = max(it.peso, 1e-9)
            d_val = max(d_rest.get(it.cidade_origem, 1.0), 1e-9)
            return (p_val ** a) / ((w_val ** b) * (d_val ** c))

        itens_ord = sorted(itens_validos, key=_score, reverse=True)
        phi = max(1, math.ceil((m / max(1e-6, gamma)) * alpha + 1e-5))

        z_estavel: List[int] = []
        peso_estavel = 0.0
        f_estavel = alpha * g_vazio - (1.0 - alpha) * taxa_aluguel * h_vazio
        g_estavel, h_estavel = g_vazio, h_vazio

        z_cand: List[int] = []
        peso_cand = 0.0
        houve_inclusao = False
        k_estavel = 0
        k_cand = 0

        while k_cand < m and phi >= 1:
            it = itens_ord[k_cand]
            if peso_cand + it.peso <= cap:
                z_cand.append(it.id)
                peso_cand += it.peso
                houve_inclusao = True

            if ((k_cand + 1) % phi == 0) and houve_inclusao:
                g_c, h_c = avaliar_biobjetivo(instance, rota, set(z_cand), itens_by_id)
                f_c = alpha * g_c - (1.0 - alpha) * taxa_aluguel * h_c
                if f_c > f_estavel:
                    z_estavel = list(z_cand)
                    peso_estavel = peso_cand
                    f_estavel = f_c
                    g_estavel, h_estavel = g_c, h_c
                    k_estavel = k_cand + 1
                else:
                    z_cand = list(z_estavel)
                    peso_cand = peso_estavel
                    k_cand = k_estavel - 1
                    phi = phi // 2
                houve_inclusao = False

            k_cand += 1

        if houve_inclusao:
            g_c, h_c = avaliar_biobjetivo(instance, rota, set(z_cand), itens_by_id)
            f_c = alpha * g_c - (1.0 - alpha) * taxa_aluguel * h_c
            if f_c > f_estavel:
                z_estavel = list(z_cand)
                f_estavel = f_c
                g_estavel, h_estavel = g_c, h_c

        if f_estavel > f_best:
            z_best = set(z_estavel)
            g_best, h_best = g_estavel, h_estavel
            f_best = f_estavel

    return z_best, g_best, h_best


def calcular_distancia_media_cidades(instance: TTPInstance) -> float:
    """Calcula a distância geográfica média l_médio entre todos os pares de cidades."""
    cidades = sorted(instance.grafo.cidades.keys())
    n = len(cidades)
    if n <= 1:
        return 0.0
    dist_mtx = getattr(instance.grafo, "_distance_matrix", None)
    if dist_mtx is not None:
        sub = dist_mtx[np.ix_(cidades, cidades)]
        soma = float(np.sum(sub) - np.trace(sub))
        return soma / (n * (n - 1))
    soma = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            soma += 2.0 * float(instance.grafo.distancia(cidades[i], cidades[j]))
    return soma / (n * (n - 1))


def explorar_2opt_com_filtro_beta(
    instance: TTPInstance,
    pivo: ParetoEntry,
    alpha: float,
    beta: float,
    l_medio: float,
    itens_by_id: Dict[int, Item],
) -> Optional[Tuple[List[int], float, float]]:
    """
    Aplica 2-OPT filtrando movimentos com Δd <= l_medio * β e retorna
    a melhor rota (rota_melhor, g, h) segundo f(π, z', α), ou None.
    """
    if beta == float("-inf"):
        return None

    rota = pivo.rota
    n = len(rota)
    if n <= 3:
        return None

    dist_mtx = getattr(instance.grafo, "_distance_matrix", None)

    def _d(u: int, v: int) -> float:
        return float(dist_mtx[u, v]) if dist_mtx is not None else float(instance.grafo.distancia(u, v))

    taxa = float(instance.taxa_aluguel)
    limiar_dist = l_medio * beta
    melhor_f = pivo.escalarizado(alpha, taxa)
    melhor_res: Optional[Tuple[List[int], float, float]] = None

    for i in range(1, n - 1):
        u1, v1 = rota[i - 1], rota[i]
        d_u1_v1 = _d(u1, v1)
        for j in range(i + 1, n):
            u2, v2 = rota[j], rota[(j + 1) % n]
            delta_d = _d(u1, u2) + _d(v1, v2) - d_u1_v1 - _d(u2, v2)
            if delta_d <= limiar_dist + 1e-12:
                r_cand = rota[:i] + rota[i : j + 1][::-1] + rota[j + 1 :]
                g_c, h_c = avaliar_biobjetivo(instance, r_cand, pivo.itens_ids, itens_by_id)
                f_c = alpha * g_c - (1.0 - alpha) * taxa * h_c
                if f_c > melhor_f + 1e-9:
                    melhor_f = f_c
                    melhor_res = (r_cand, g_c, h_c)

    return melhor_res


def explorar_bit_flip_amostrado(
    instance: TTPInstance,
    pivo: ParetoEntry,
    lambd: float,
    arquivo: List[ParetoEntry],
    rng: np.random.Generator,
    itens_by_id: Dict[int, Item],
) -> None:
    """
    Aplica Bit-Flip amostrado com probabilidade λ sobre os itens de pivo,
    atualizando o arquivo de soluções não-dominadas S.
    """
    cap = float(instance.capacidade_mochila)
    peso_pivo = sum(itens_by_id[i].peso for i in pivo.itens_ids if i in itens_by_id)

    for it in instance.itens:
        if it.cidade_origem == 1:
            continue
        if rng.random() <= lambd:
            if it.id in pivo.itens_ids:
                cand_ids = set(pivo.itens_ids)
                cand_ids.remove(it.id)
            else:
                if peso_pivo + it.peso > cap:
                    continue
                cand_ids = set(pivo.itens_ids)
                cand_ids.add(it.id)

            g_c, h_c = avaliar_biobjetivo(instance, pivo.rota, cand_ids, itens_by_id)
            atualizar_arquivo_pareto(arquivo, pivo.rota, cand_ids, g_c, h_c)


def itens_para_plano_coleta(
    itens_ids: Set[int],
    itens_by_id: Dict[int, Item],
) -> Dict[int, List[int]]:
    plano: Dict[int, List[int]] = {}
    for it_id in itens_ids:
        it = itens_by_id.get(it_id)
        if it is not None:
            plano.setdefault(it.cidade_origem, []).append(it_id)
    return plano


def extrair_melhor_ttp1_do_arquivo(
    instance: TTPInstance,
    arquivo: List[ParetoEntry],
    itens_by_id: Dict[int, Item],
) -> SolucaoTTP:
    """Seleciona no arquivo S a solução que maximiza o objetivo TTP1 (α = 0.5)."""
    taxa = float(instance.taxa_aluguel)
    melhor = max(arquivo, key=lambda e: e.lucro_ttp1(taxa))
    return SolucaoTTP(
        rota=list(melhor.rota),
        plano_coleta=itens_para_plano_coleta(melhor.itens_ids, itens_by_id),
    )
