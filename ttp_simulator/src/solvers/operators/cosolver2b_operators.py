"""
Operadores da heurística Cosolver2B (El Yafrani & Ahiod, 2015):
- Triangulação de Delaunay para poda de vizinhança 2-OPT
- Inicialização por Inserção & Eliminação
- Recuperação incremental de objetivo (tacc / wacc)
- 2-OPT restrito por Delaunay (First-Fit e Best-Fit)
- Bit-Flip estruturado por cidade (First-Fit e Best-Fit)
"""

from typing import Dict, List, Set, Tuple

import numpy as np

from src.core.fast_evaluator import FastTTPEvaluator
from src.core.models import Grafo, Item, TTPInstance


def gerar_grafo_delaunay(
    grafo: Grafo,
    fallback_k: int = 15,
) -> Dict[int, Set[int]]:
    """
    Pré-computa o grafo de adjacência da Triangulação de Delaunay sobre as
    coordenadas 2D das cidades. Faz fallback seguro para os K vizinhos mais
    próximos se houver menos de 3 cidades ou pontos colineares/degenerados.
    """
    cidades_ids = sorted(grafo.cidades.keys())
    adj: Dict[int, Set[int]] = {cid: set() for cid in cidades_ids}

    if len(cidades_ids) >= 3:
        try:
            from matplotlib.tri import Triangulation

            xs = np.array([grafo.cidades[cid].x for cid in cidades_ids], dtype=np.float64)
            ys = np.array([grafo.cidades[cid].y for cid in cidades_ids], dtype=np.float64)
            tri = Triangulation(xs, ys)
            for u_idx, v_idx in tri.edges:
                u = cidades_ids[int(u_idx)]
                v = cidades_ids[int(v_idx)]
                adj[u].add(v)
                adj[v].add(u)
            return adj
        except Exception:
            pass

    closest = grafo.pre_calculate_closest_nodes(num_closest=fallback_k)
    for cid in cidades_ids:
        for viz in closest.get(cid, []):
            if viz != 0 and viz != cid:
                adj[cid].add(viz)
                adj[viz].add(cid)
    return adj


def indexar_itens_por_cidade(
    instance: TTPInstance,
) -> Dict[int, List[Item]]:
    """Agrupa os itens da instância por cidade de origem (excluindo o depósito 1)."""
    por_cidade: Dict[int, List[Item]] = {}
    for it in instance.itens:
        if it.cidade_origem != 1:
            por_cidade.setdefault(it.cidade_origem, []).append(it)
    return por_cidade


def insercao_e_eliminacao(
    instance: TTPInstance,
    rota: List[int],
) -> Tuple[Dict[int, List[int]], float]:
    """
    Heurística inicial de Inserção e Eliminação (El Yafrani & Ahiod, 2015):
    1. Inserção: ordena os itens por razão valor / (peso * d_restante) e insere
       gulosamente até o limite de capacidade da mochila.
    2. Eliminação: executa bit-flip restrito exclusivamente aos itens inseridos,
       desmarcando os que aumentam o benefício líquido G(x_0, z).
    """
    n = len(rota)
    if n == 0 or not instance.itens:
        return {}, 0.0

    dist_mtx = getattr(instance.grafo, "_distance_matrix", None)
    dist_restante: Dict[int, float] = {}
    d_acc = (
        float(dist_mtx[rota[-1], rota[0]])
        if dist_mtx is not None
        else float(instance.grafo.distancia(rota[-1], rota[0]))
    )
    dist_restante[rota[-1]] = d_acc
    for i in range(n - 2, -1, -1):
        d_acc += (
            float(dist_mtx[rota[i], rota[i + 1]])
            if dist_mtx is not None
            else float(instance.grafo.distancia(rota[i], rota[i + 1]))
        )
        dist_restante[rota[i]] = d_acc

    def _score(it: Item) -> float:
        if it.cidade_origem == 1:
            return -1.0
        if it.peso <= 0:
            return float("inf")
        d_rest = max(dist_restante.get(it.cidade_origem, 1.0), 1e-6)
        return it.valor / (it.peso * d_rest)

    itens_ordenados = sorted(instance.itens, key=_score, reverse=True)
    selecionados: List[Item] = []
    peso_atual = 0.0
    cap = float(instance.capacidade_mochila)

    for it in itens_ordenados:
        if it.cidade_origem == 1:
            continue
        if peso_atual + it.peso <= cap:
            selecionados.append(it)
            peso_atual += it.peso

    # Fase de Eliminação restrita aos itens inseridos
    evaluator = FastTTPEvaluator(instance, rota, modelo="TTP1")
    coletados_ids: Set[int] = {it.id for it in selecionados}
    evaluator.inicializar_estado(coletados_ids)
    z_atual = evaluator.calcular_lucro_atual()

    for it in selecionados:
        if it.id not in evaluator.itens_coletados:
            continue
        evaluator.remover_item(it.id)
        z_sem = evaluator.calcular_lucro_atual()
        if z_sem > z_atual + 1e-9:
            z_atual = z_sem
        else:
            evaluator.adicionar_item(it.id)

    plano: Dict[int, List[int]] = {}
    for it in selecionados:
        if it.id in evaluator.itens_coletados:
            plano.setdefault(it.cidade_origem, []).append(it.id)

    return plano, z_atual


def inicializar_acumuladores(
    instance: TTPInstance,
    rota: List[int],
    plano_coleta: Dict[int, List[int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Inicializa os vetores acumuladores (wacc, tacc), pesos por cidade,
    distâncias das pernas, valor bruto e lucro líquido G(x, z).
    """
    n = len(rota)
    cidade_idx = {rota[i]: i for i in range(n)}
    itens_by_id = {it.id: it for it in instance.itens}
    dist_mtx = getattr(instance.grafo, "_distance_matrix", None)

    distancias = np.empty(n, dtype=np.float64)
    for i in range(n):
        u, v = rota[i], rota[(i + 1) % n]
        distancias[i] = float(dist_mtx[u, v]) if dist_mtx is not None else float(instance.grafo.distancia(u, v))

    peso_por_cidade = np.zeros(n, dtype=np.float64)
    valor_bruto = 0.0
    for ids in plano_coleta.values():
        for it_id in ids:
            it = itens_by_id.get(it_id)
            if it is not None and it.cidade_origem in cidade_idx:
                peso_por_cidade[cidade_idx[it.cidade_origem]] += it.peso
                valor_bruto += it.valor

    cap = max(float(instance.capacidade_mochila), 1e-6)
    v_min = float(instance.v_min)
    v_max = float(instance.v_max)
    fator_vel = (v_max - v_min) / cap

    wacc = np.cumsum(peso_por_cidade)
    tacc = np.empty(n, dtype=np.float64)
    t_soma = 0.0
    for i in range(n):
        vel = v_max - wacc[i] * fator_vel
        if vel < v_min:
            vel = v_min
        t_soma += distancias[i] / vel
        tacc[i] = t_soma

    z_atual = valor_bruto - float(instance.taxa_aluguel) * t_soma
    return peso_por_cidade, wacc, tacc, distancias, valor_bruto, z_atual


def two_opt_delaunay(
    instance: TTPInstance,
    rota: List[int],
    plano_coleta: Dict[int, List[int]],
    grafo_delaunay: Dict[int, Set[int]],
    estrategia: str = "best_fit",
) -> Tuple[List[int], float, bool]:
    """
    Passo TSKP do Cosolver2B: busca local 2-OPT restrita às arestas da
    Triangulação de Delaunay com recuperação incremental (wacc, tacc).
    Suporta estrategia='first_fit' ou 'best_fit'.
    """
    n = len(rota)
    if n <= 3:
        _, _, tacc, _, _, z = inicializar_acumuladores(instance, rota, plano_coleta)
        return list(rota), z, False

    cap = max(float(instance.capacidade_mochila), 1e-6)
    v_min = float(instance.v_min)
    v_max = float(instance.v_max)
    taxa_aluguel = float(instance.taxa_aluguel)
    fator_vel = (v_max - v_min) / cap
    dist_mtx = getattr(instance.grafo, "_distance_matrix", None)

    def _dist(u: int, v: int) -> float:
        return float(dist_mtx[u, v]) if dist_mtx is not None else float(instance.grafo.distancia(u, v))

    rota_atual = list(rota)
    peso_por_cidade, wacc, tacc, _, valor_bruto, z_atual = inicializar_acumuladores(
        instance, rota_atual, plano_coleta
    )
    melhorou_alguma_vez = False

    while True:
        pos_map = {cid: idx for idx, cid in enumerate(rota_atual)}
        t_atual = float(tacc[n - 1])
        melhor_t = t_atual
        melhor_par: Tuple[int, int] | None = None
        aplicado_first_fit = False

        for i in range(1, n - 1):
            cid_i = rota_atual[i]
            vizinhos = grafo_delaunay.get(cid_i, set())
            candidatos_j = sorted(
                pos_map[v] for v in vizinhos if v in pos_map and pos_map[v] > i
            )
            if not candidatos_j:
                continue

            t_prefixo = float(tacc[i - 2]) if i >= 2 else 0.0
            w_prefixo = float(wacc[i - 1])

            for j in candidatos_j:
                # Reversão de rota_atual[i..j]: recálculo incremental a partir de i-1
                t_seg = _dist(rota_atual[i - 1], rota_atual[j]) / max(
                    v_min, v_max - w_prefixo * fator_vel
                )
                w_cur = w_prefixo
                for k in range(j, i, -1):
                    w_cur += float(peso_por_cidade[k])
                    vel = max(v_min, v_max - w_cur * fator_vel)
                    t_seg += _dist(rota_atual[k], rota_atual[k - 1]) / vel

                # Perna saindo de rota_atual[i] (nova posição j) para rota_atual[(j+1)%n]
                w_cur += float(peso_por_cidade[i])
                vel = max(v_min, v_max - w_cur * fator_vel)
                t_seg += _dist(rota_atual[i], rota_atual[(j + 1) % n]) / vel

                # Segmento posterior j+1 .. n-1 (pesos e arestas intactos, apenas
                # já temos wacc[k] e distâncias originais)
                t_sufixo = float(tacc[n - 1] - tacc[j])
                t_cand = t_prefixo + t_seg + t_sufixo

                if t_cand < melhor_t - 1e-9:
                    melhor_t = t_cand
                    melhor_par = (i, j)
                    if estrategia == "first_fit":
                        rota_atual[i : j + 1] = rota_atual[i : j + 1][::-1]
                        peso_por_cidade, wacc, tacc, _, valor_bruto, z_atual = (
                            inicializar_acumuladores(instance, rota_atual, plano_coleta)
                        )
                        melhorou_alguma_vez = True
                        aplicado_first_fit = True
                        break
            if aplicado_first_fit:
                break

        if aplicado_first_fit:
            continue

        if melhor_par is not None:
            bi, bj = melhor_par
            rota_atual[bi : bj + 1] = rota_atual[bi : bj + 1][::-1]
            peso_por_cidade, wacc, tacc, _, valor_bruto, z_atual = (
                inicializar_acumuladores(instance, rota_atual, plano_coleta)
            )
            melhorou_alguma_vez = True
        else:
            break

    z_final = valor_bruto - taxa_aluguel * float(tacc[n - 1])
    return rota_atual, z_final, melhorou_alguma_vez


def bit_flip_cosolver2b(
    instance: TTPInstance,
    rota: List[int],
    plano_coleta: Dict[int, List[int]],
    itens_por_cidade: Dict[int, List[Item]],
    estrategia: str = "best_fit",
) -> Tuple[Dict[int, List[int]], float, bool]:
    """
    Passo KRP do Cosolver2B: busca local Bit-Flip estruturada por cidade
    com recuperação incremental de tempo a partir da cidade modificada.
    Suporta estrategia='first_fit' ou 'best_fit'.
    """
    n = len(rota)
    if n == 0 or not instance.itens:
        return {}, 0.0, False

    cap = max(float(instance.capacidade_mochila), 1e-6)
    v_min = float(instance.v_min)
    v_max = float(instance.v_max)
    taxa_aluguel = float(instance.taxa_aluguel)
    fator_vel = (v_max - v_min) / cap

    novo_plano: Dict[int, List[int]] = {
        c: list(ids) for c, ids in plano_coleta.items() if ids
    }
    coletados: Set[int] = {it_id for ids in novo_plano.values() for it_id in ids}
    peso_por_cidade, wacc, tacc, distancias, valor_bruto, z_atual = (
        inicializar_acumuladores(instance, rota, novo_plano)
    )
    peso_total = float(wacc[n - 1]) if n > 0 else 0.0
    melhorou_alguma_vez = False

    while True:
        melhor_delta_g = 0.0
        melhor_acao: Tuple[int, Item, bool] | None = None
        aplicado_first_fit = False

        for c_idx in range(1, n):
            cid = rota[c_idx]
            itens_cid = itens_por_cidade.get(cid)
            if not itens_cid:
                continue

            t_antes = float(tacc[c_idx - 1])
            t_sufixo_atual = float(tacc[n - 1]) - t_antes

            for it in itens_cid:
                esta_coletado = it.id in coletados
                if esta_coletado:
                    dp = -it.peso
                    dv = -it.valor
                else:
                    if peso_total + it.peso > instance.capacidade_mochila:
                        continue
                    dp = it.peso
                    dv = it.valor

                vels = v_max - (wacc[c_idx:] + dp) * fator_vel
                np.maximum(vels, v_min, out=vels)
                t_sufixo_novo = float(np.sum(distancias[c_idx:] / vels))
                delta_g = dv - taxa_aluguel * (t_sufixo_novo - t_sufixo_atual)

                if delta_g > melhor_delta_g + 1e-9:
                    melhor_delta_g = delta_g
                    melhor_acao = (c_idx, it, esta_coletado)
                    if estrategia == "first_fit":
                        if esta_coletado:
                            coletados.remove(it.id)
                            novo_plano[cid].remove(it.id)
                            if not novo_plano[cid]:
                                novo_plano.pop(cid, None)
                        else:
                            coletados.add(it.id)
                            novo_plano.setdefault(cid, []).append(it.id)
                        peso_total += dp
                        peso_por_cidade, wacc, tacc, distancias, valor_bruto, z_atual = (
                            inicializar_acumuladores(instance, rota, novo_plano)
                        )
                        melhorou_alguma_vez = True
                        aplicado_first_fit = True
                        break
            if aplicado_first_fit:
                break

        if aplicado_first_fit:
            continue

        if melhor_acao is not None:
            _, it, esta_coletado = melhor_acao
            cid = it.cidade_origem
            if esta_coletado:
                coletados.remove(it.id)
                novo_plano[cid].remove(it.id)
                if not novo_plano[cid]:
                    novo_plano.pop(cid, None)
                peso_total -= it.peso
            else:
                coletados.add(it.id)
                novo_plano.setdefault(cid, []).append(it.id)
                peso_total += it.peso

            peso_por_cidade, wacc, tacc, distancias, valor_bruto, z_atual = (
                inicializar_acumuladores(instance, rota, novo_plano)
            )
            melhorou_alguma_vez = True
        else:
            break

    return novo_plano, z_atual, melhorou_alguma_vez
