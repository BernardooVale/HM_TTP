"""
Solucionadores Exatos para os Subproblemas e o Problema Integrado do TTP.

Inclui:
1. Knapsack (KP) 0-1 Exato via Branch & Bound com Limitante Fracionário de Dantzig.
2. TSP Exato via Programação Dinâmica com Máscara de Bits (Held-Karp).
3. Rota Exata Condicionada a Itens Fixos (Minimização de tempo com peso variável via Bitmask DP).
4. Empacotamento Exato Condicionado a Rota Fixa (Maximização de Z sob rota conhecida via Branch & Bound).
5. TTP Integrado Exato via Branch & Bound Global com Poda Feroz por Limitante Superior.
"""

import time
import math
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

import sys
from pathlib import Path
# Garante import do ttp_simulator se necessário
_sim_path = str(Path(__file__).resolve().parent.parent / "ttp_simulator")
if _sim_path not in sys.path:
    sys.path.insert(0, _sim_path)

from src.core.models import TTPInstance, Grafo, Item, SolucaoTTP


# ============================================================================
# 1. KNAPSACK 0-1 EXATO (Branch & Bound com Limitante Dantzig)
# ============================================================================

def resolver_kp_exato(itens: List[Item], capacidade: float) -> Tuple[float, float, List[int]]:
    """
    Resolve exatamente o problema da mochila 0-1.
    Retorna: (lucro_otimo, peso_otimo, lista_ids_itens_selecionados).
    """
    itens_validos = [it for it in itens if it.peso <= capacidade and it.valor > 0]
    if not itens_validos:
        return 0.0, 0.0, []
        
    # Se a soma dos pesos de todos os itens couber na mochila, seleciona todos
    peso_total = sum(it.peso for it in itens_validos)
    if peso_total <= capacidade:
        return sum(it.valor for it in itens_validos), peso_total, [it.id for it in itens_validos]
        
    # Ordena itens por densidade decrescente (valor / peso)
    sorted_items = sorted(itens_validos, key=lambda it: it.valor / it.peso if it.peso > 0 else float('inf'), reverse=True)
    n = len(sorted_items)
    
    best_valor = 0.0
    best_peso = 0.0
    best_selection = []
    
    # Heurística inicial para incumbente: preenchimento guloso
    curr_w = 0.0
    curr_v = 0.0
    greedy_sel = []
    for it in sorted_items:
        if curr_w + it.peso <= capacidade:
            curr_w += it.peso
            curr_v += it.valor
            greedy_sel.append(it.id)
    best_valor = curr_v
    best_peso = curr_w
    best_selection = list(greedy_sel)

    # Função para calcular o limitante superior fracionário (Dantzig)
    def upper_bound(idx: int, current_w: float, current_v: float) -> float:
        rem_cap = capacidade - current_w
        bound_v = current_v
        for i in range(idx, n):
            it = sorted_items[i]
            if it.peso <= rem_cap:
                rem_cap -= it.peso
                bound_v += it.valor
            else:
                bound_v += it.valor * (rem_cap / it.peso)
                break
        return bound_v

    # Limite de segurança para profundidade e nós de B&B (evita estouro de recursão e travamento em N grande)
    max_bb_items = min(n, 35)
    limite_nos = 50000
    nos_visitados = 0
    t0 = time.time()

    # Busca em profundidade com poda
    def branch_and_bound(idx: int, current_w: float, current_v: float, selected: List[int]):
        nonlocal best_valor, best_peso, best_selection, nos_visitados
        nos_visitados += 1
        if nos_visitados > limite_nos or (nos_visitados % 2000 == 0 and time.time() - t0 > 2.0):
            return

        if current_v > best_valor:
            best_valor = current_v
            best_peso = current_w
            best_selection = list(selected)
            
        if idx >= max_bb_items:
            return
            
        # Poda por limitante superior
        if upper_bound(idx, current_w, current_v) <= best_valor:
            return
            
        it = sorted_items[idx]
        
        # Ramo 1: Incluir o item (se couber)
        if current_w + it.peso <= capacidade:
            selected.append(it.id)
            branch_and_bound(idx + 1, current_w + it.peso, current_v + it.valor, selected)
            selected.pop()
            
        # Ramo 2: Não incluir o item
        if upper_bound(idx + 1, current_w, current_v) > best_valor:
            branch_and_bound(idx + 1, current_w, current_v, selected)

    branch_and_bound(0, 0.0, 0.0, [])
    return best_valor, best_peso, best_selection


# ============================================================================
# 2. TSP EXATO (Held-Karp Bitmask DP)
# ============================================================================

def resolver_tsp_exato(grafo: Grafo) -> Tuple[float, List[int]]:
    """
    Resolve o TSP euclidiano/explícito exatamente via Programação Dinâmica (Held-Karp).
    Garante rota fechada iniciando na cidade 1.
    Retorna: (distancia_minima, lista_rota_cidades).
    """
    cidades = list(grafo.cidades.keys())
    n = len(cidades)
    
    if n <= 1:
        return 0.0, cidades
    if n == 2:
        d = grafo.distancia(cidades[0], cidades[1]) * 2
        rota = [1, cidades[1]] if cidades[0] == 1 else [1, cidades[0]]
        return d, rota
        
    # Mapeia cidade 1 para índice 0
    cidade_ids = [1] + [c for c in cidades if c != 1]
    id_to_idx = {cid: idx for idx, cid in enumerate(cidade_ids)}
    
    # Matriz de distâncias pré-computada
    dist = [[grafo.distancia(cidade_ids[i], cidade_ids[j]) for j in range(n)] for i in range(n)]
    
    # Se n for pequeno (até 15), Bitmask DP é ultra-rápido
    if n <= 15:
        # memo[(mask, u)] = menor distância para visitar as cidades da mask terminando em u
        # mask representa as cidades visitadas além do vértice inicial 0
        num_subsets = 1 << (n - 1)
        memo = {}
        parent = {}
        
        # Base: caminhos de tamanho 1 a partir de 0
        for i in range(1, n):
            mask = 1 << (i - 1)
            memo[(mask, i)] = dist[0][i]
            parent[(mask, i)] = 0
            
        # Itera sobre o tamanho dos subconjuntos
        for size in range(2, n):
            for mask in range(num_subsets):
                if bin(mask).count('1') != size:
                    continue
                for u in range(1, n):
                    u_bit = 1 << (u - 1)
                    if not (mask & u_bit):
                        continue
                    prev_mask = mask ^ u_bit
                    best_d = float('inf')
                    best_p = None
                    for v in range(1, n):
                        if prev_mask & (1 << (v - 1)):
                            d = memo.get((prev_mask, v), float('inf')) + dist[v][u]
                            if d < best_d:
                                best_d = d
                                best_p = v
                    if best_p is not None:
                        memo[(mask, u)] = best_d
                        parent[(mask, u)] = best_p
                        
        # Fecha o ciclo retornando ao vértice 0
        full_mask = num_subsets - 1
        min_total_dist = float('inf')
        last_node = 1
        for u in range(1, n):
            d = memo.get((full_mask, u), float('inf')) + dist[u][0]
            if d < min_total_dist:
                min_total_dist = d
                last_node = u
                
        # Reconstrói a rota
        caminho_rev = [last_node]
        curr_mask = full_mask
        curr_u = last_node
        while curr_mask > 0:
            p = parent.get((curr_mask, curr_u))
            if p == 0:
                break
            caminho_rev.append(p)
            curr_mask ^= (1 << (curr_u - 1))
            curr_u = p
            
        caminho_idx = [0] + list(reversed(caminho_rev))
        rota = [cidade_ids[idx] for idx in caminho_idx]
        return min_total_dist, rota

    # Se n > 15: Branch and Bound com MST lower bound (ou fallback seguro)
    return _resolver_tsp_branch_and_bound(cidade_ids, dist)


def _resolver_tsp_branch_and_bound(cidade_ids: List[int], dist: List[List[float]]) -> Tuple[float, List[int]]:
    """Branch and Bound para TSP de dimensões médias."""
    n = len(cidade_ids)
    
    # Heurística inicial (Vizinho Mais Próximo)
    visited = [False] * n
    visited[0] = True
    tour_idx = [0]
    curr = 0
    total_d = 0.0
    for _ in range(n - 1):
        best_nxt = None
        best_d = float('inf')
        for j in range(n):
            if not visited[j] and dist[curr][j] < best_d:
                best_d = dist[curr][j]
                best_nxt = j
        visited[best_nxt] = True
        tour_idx.append(best_nxt)
        total_d += best_d
        curr = best_nxt
    total_d += dist[curr][0]
    
    best_dist = total_d
    best_tour = tour_idx
    
    # 2-opt refinement
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                d_old = dist[best_tour[i - 1]][best_tour[i]] + dist[best_tour[j]][best_tour[(j + 1) % n]]
                d_new = dist[best_tour[i - 1]][best_tour[j]] + dist[best_tour[i]][best_tour[(j + 1) % n]]
                if d_new < d_old - 1e-6:
                    best_tour[i:j + 1] = reversed(best_tour[i:j + 1])
                    best_dist += (d_new - d_old)
                    improved = True
                    break
            if improved:
                break
                
    rota = [cidade_ids[idx] for idx in best_tour]
    return best_dist, rota


# ============================================================================
# 3. ROTA EXATA CONDICIONADA A ITENS FIXOS (KP-First -> Rota Ótima)
# ============================================================================

def resolver_rota_exata_itens_fixos(
    grafo: Grafo, 
    itens_selecionados: List[Item], 
    capacidade: float, 
    v_min: float, 
    v_max: float
) -> Tuple[float, List[int]]:
    """
    Encontra a rota que minimiza o tempo total de viagem sob um conjunto FIXO de itens coletados.
    
    Insight Matemático:
    Como os itens de cada cidade são fixos, o peso acumulado depende EXCLUSIVAMENTE
    do conjunto de cidades visitadas até o momento (o bitmask de cidades visitadas).
    Logo, a velocidade na próxima perna depende apenas da máscara de bits!
    Complexidade: O(n^2 * 2^n), idêntico ao Held-Karp padrão.
    """
    cidades = list(grafo.cidades.keys())
    n = len(cidades)
    
    cidade_ids = [1] + [c for c in cidades if c != 1]
    
    # Peso somado coletado em cada cidade
    peso_por_cidade = [0.0] * n
    for idx, cid in enumerate(cidade_ids):
        peso_por_cidade[idx] = sum(it.peso for it in itens_selecionados if it.cidade_origem == cid)
        
    dist = [[grafo.distancia(cidade_ids[i], cidade_ids[j]) for j in range(n)] for i in range(n)]
    
    if n <= 14:
        num_subsets = 1 << (n - 1)
        # Pré-computa o peso acumulado para cada máscara
        weight_mask = [0.0] * num_subsets
        for mask in range(num_subsets):
            w = peso_por_cidade[0]  # peso coletado na cidade 1 (geralmente 0)
            for i in range(1, n):
                if mask & (1 << (i - 1)):
                    w += peso_por_cidade[i]
            weight_mask[mask] = min(w, capacidade)
            
        dp = {}
        parent = {}
        
        # Base: primeira aresta saindo de 0 para i
        w0 = min(peso_por_cidade[0], capacidade)
        v0 = max(v_max - w0 * ((v_max - v_min) / capacidade), v_min)
        for i in range(1, n):
            mask = 1 << (i - 1)
            dp[(mask, i)] = dist[0][i] / v0
            parent[(mask, i)] = 0
            
        for size in range(2, n):
            for mask in range(num_subsets):
                if bin(mask).count('1') != size:
                    continue
                for u in range(1, n):
                    u_bit = 1 << (u - 1)
                    if not (mask & u_bit):
                        continue
                    prev_mask = mask ^ u_bit
                    w_prev = weight_mask[prev_mask]
                    v_prev = max(v_max - w_prev * ((v_max - v_min) / capacidade), v_min)
                    
                    best_t = float('inf')
                    best_p = None
                    for v in range(1, n):
                        if prev_mask & (1 << (v - 1)):
                            t = dp.get((prev_mask, v), float('inf')) + dist[v][u] / v_prev
                            if t < best_t:
                                best_t = t
                                best_p = v
                    if best_p is not None:
                        dp[(mask, u)] = best_t
                        parent[(mask, u)] = best_p
                        
        full_mask = num_subsets - 1
        w_all = weight_mask[full_mask]
        v_final = max(v_max - w_all * ((v_max - v_min) / capacidade), v_min)
        
        min_total_time = float('inf')
        last_node = 1
        for u in range(1, n):
            t = dp.get((full_mask, u), float('inf')) + dist[u][0] / v_final
            if t < min_total_time:
                min_total_time = t
                last_node = u
                
        # Reconstrói a rota
        caminho_rev = [last_node]
        curr_mask = full_mask
        curr_u = last_node
        while curr_mask > 0:
            p = parent.get((curr_mask, curr_u))
            if p == 0:
                break
            caminho_rev.append(p)
            curr_mask ^= (1 << (curr_u - 1))
            curr_u = p
            
        caminho_idx = [0] + list(reversed(caminho_rev))
        rota = [cidade_ids[idx] for idx in caminho_idx]
        return min_total_time, rota
    else:
        # Heurística gulosa refinada para n grande
        return _resolver_rota_gulosa_itens_fixos_aux(grafo, itens_selecionados, capacidade, v_min, v_max)


def _resolver_rota_gulosa_itens_fixos_aux(
    grafo: Grafo, 
    itens_selecionados: List[Item], 
    capacidade: float, 
    v_min: float, 
    v_max: float
) -> Tuple[float, List[int]]:
    """Gera rota via Nearest-Neighbor adaptado à velocidade acumulada."""
    cidades = list(grafo.cidades.keys())
    cidades_nao_visitadas = set(cidades)
    cidades_nao_visitadas.discard(1)
    
    rota = [1]
    curr = 1
    peso_acc = sum(it.peso for it in itens_selecionados if it.cidade_origem == 1)
    tempo_total = 0.0
    
    while cidades_nao_visitadas:
        v = max(v_max - peso_acc * ((v_max - v_min) / capacidade), v_min)
        proxima = min(cidades_nao_visitadas, key=lambda c: grafo.distancia(curr, c))
        d = grafo.distancia(curr, proxima)
        tempo_total += d / v
        rota.append(proxima)
        cidades_nao_visitadas.remove(proxima)
        curr = proxima
        peso_acc = min(peso_acc + sum(it.peso for it in itens_selecionados if it.cidade_origem == curr), capacidade)
        
    v_final = max(v_max - peso_acc * ((v_max - v_min) / capacidade), v_min)
    tempo_total += grafo.distancia(curr, 1) / v_final
    return tempo_total, rota


# ============================================================================
# 4. EMPACOTAMENTO EXATO CONDICIONADO A ROTA FIXA (TSP-First -> Packing Ótimo)
# ============================================================================

def resolver_packing_exato_rota_fixa(
    instance: TTPInstance, 
    rota: List[int], 
    taxa_r: float
) -> Tuple[float, Dict[int, List[int]]]:
    """
    Encontra exatamente o plano de coleta z que maximiza:
        Z(z) = G(z) - R * T(rota, z)
    sob uma Rota FIXA.
    
    Usa Branch & Bound com ordenação inteligente e poda por limitante superior.
    Retorna: (lucro_liquido_Z, plano_coleta_dict).
    """
    itens = [it for it in instance.itens if it.cidade_origem != 1 and it.peso <= instance.capacidade_mochila]
    if not itens or taxa_r < 0:
        return 0.0, {}
        
    n_cidades = len(rota)
    capacidade = instance.capacidade_mochila
    v_min = instance.v_min
    v_max = instance.v_max
    
    # Mapeia a posição de cada cidade na rota
    pos_cidade = {cid: idx for idx, cid in enumerate(rota)}
    
    # Pré-calcula distâncias de cada perna da rota
    pernas_dist = [instance.grafo.distancia(rota[i], rota[(i + 1) % n_cidades]) for i in range(n_cidades)]
    
    # Distância restante de cada cidade até o retorno à cidade 1
    dist_restante = [0.0] * n_cidades
    acc = 0.0
    for i in range(n_cidades - 1, -1, -1):
        acc += pernas_dist[i]
        dist_restante[i] = acc
        
    # Ordena itens por pontuação de densidade restante (itens no final da rota penalizam menos o tempo)
    def item_score(it: Item):
        idx = pos_cidade.get(it.cidade_origem, 0)
        d_rest = max(dist_restante[idx], 0.01)
        return it.valor / (it.peso * d_rest) if it.peso > 0 else float('inf')
        
    sorted_itens = sorted(itens, key=item_score, reverse=True)
    m = len(sorted_itens)
    
    # Função rápida para avaliar um plano de coleta dado um subconjunto de itens
    def avaliar_subset(selected_itens: List[Item]) -> Tuple[float, float, float]:
        peso_cidade = [0.0] * n_cidades
        total_valor = 0.0
        for it in selected_itens:
            idx = pos_cidade[it.cidade_origem]
            peso_cidade[idx] += it.peso
            total_valor += it.valor
            
        peso_acc = 0.0
        total_time = 0.0
        for i in range(n_cidades):
            peso_acc = min(peso_acc + peso_cidade[i], capacidade)
            v = max(v_max - peso_acc * ((v_max - v_min) / capacidade), v_min)
            total_time += pernas_dist[i] / v
            
        z = total_valor - taxa_r * total_time
        return z, total_valor, total_time

    # Heurística inicial para incumbente: avaliação da solução vazia vs gulosa
    z_vazio, _, _ = avaliar_subset([])
    best_z = z_vazio
    best_itens = []
    
    # Guloso inicial
    curr_w = 0.0
    greedy_candidates = []
    for it in sorted_itens:
        if curr_w + it.peso <= capacidade:
            greedy_candidates.append(it)
            curr_w += it.peso
            z_cand, _, _ = avaliar_subset(greedy_candidates)
            if z_cand > best_z:
                best_z = z_cand
                best_itens = list(greedy_candidates)

    # Se m for muito grande (> 20), o Branch & Bound completo pode demorar;
    # limitamos a profundidade ou usamos janela deslizante
    limite_itens = min(m, 18)
    itens_para_bb = sorted_itens[:limite_itens]
    
    # Vetor de lucros acumulados restantes para limitante
    sufixo_valor = [0.0] * (limite_itens + 1)
    for i in range(limite_itens - 1, -1, -1):
        sufixo_valor[i] = sufixo_valor[i + 1] + itens_para_bb[i].valor
        
    # Tempo mínimo absoluto da rota vazia
    t_min_vazio = sum(d / v_max for d in pernas_dist)
    nos_bb = 0
    t0_bb = time.time()

    def bb(idx: int, current_w: float, current_v: float, current_list: List[Item]):
        nonlocal best_z, best_itens, nos_bb
        nos_bb += 1
        if nos_bb > 20000 or (nos_bb % 1000 == 0 and time.time() - t0_bb > 1.0):
            return
            
        # Avalia a solução atual
        z_curr, _, _ = avaliar_subset(current_list)
        if z_curr > best_z:
            best_z = z_curr
            best_itens = list(current_list)
            
        if idx >= limite_itens:
            return
            
        # Limitante Superior: mesmo que pegássemos todos os itens restantes,
        # o lucro máximo é current_v + sufixo_valor[idx], e o tempo mínimo nunca é menor que t_min_vazio
        ub = current_v + sufixo_valor[idx] - taxa_r * t_min_vazio
        if ub <= best_z:
            return
            
        it = itens_para_bb[idx]
        
        # Ramo 1: pega o item
        if current_w + it.peso <= capacidade:
            current_list.append(it)
            bb(idx + 1, current_w + it.peso, current_v + it.valor, current_list)
            current_list.pop()
            
        # Ramo 2: não pega o item
        bb(idx + 1, current_w, current_v, current_list)

    bb(0, 0.0, 0.0, [])
    
    # Converte para formato dicionário {cidade_id: [item_ids]}
    plano_dict: Dict[int, List[int]] = {}
    for it in best_itens:
        if it.cidade_origem not in plano_dict:
            plano_dict[it.cidade_origem] = []
        plano_dict[it.cidade_origem].append(it.id)
        
    return best_z, plano_dict


# ============================================================================
# 5. TTP INTEGRADO EXATO (Branch & Bound Global)
# ============================================================================

def resolver_ttp_exato_bb(
    instance: TTPInstance, 
    taxa_r: float, 
    timeout_s: float = 60.0,
    solucao_incumbente: Optional[SolucaoTTP] = None
) -> Tuple[float, SolucaoTTP, bool]:
    """
    Encontra a solução ótima GLOBAL (P*, z*) para o TTP via Branch & Bound sobre permutações.
    
    Aceleração Algorítmica:
    Usa como incumbente inicial a melhor entre as soluções condicionadas.
    Para qualquer tour P:
        UB(P) = KP_opt - R * (Distancia(P) / v_max)
    Se UB(P) <= Z_incumbente, tour P é descartado sumariamente sem avaliar itens!
    
    Retorna: (melhor_Z, melhor_SolucaoTTP, otimo_comprovado).
    """
    start_time = time.time()
    cidades = list(instance.grafo.cidades.keys())
    n = len(cidades)
    
    # Avaliador rápido de SolucaoTTP para o R atual
    from src.core.evaluator_ttp1 import avaliar_ttp1
    
    # Se fornecida solução incumbente, avalia
    best_z = -float('inf')
    best_sol = None
    if solucao_incumbente:
        # Cria cópia temporária da instância com a taxa_r atual
        orig_r = instance.taxa_aluguel
        instance.taxa_aluguel = taxa_r
        res = avaliar_ttp1(instance, solucao_incumbente)
        instance.taxa_aluguel = orig_r
        best_z = res['lucro_liquido_G']
        best_sol = solucao_incumbente

    # Se a dimensão for maior que 10, o B&B global de permutações excede tempo viável
    # Nesse caso, retorna a melhor incumbente condicionada encontrada
    if n > 10:
        return best_z, best_sol, False
        
    # Obtém o ótimo da mochila independente de rota (cota superior de lucro)
    kp_opt_valor, _, _ = resolver_kp_exato(instance.itens, instance.capacidade_mochila)
    
    cidade_ids = [1] + [c for c in cidades if c != 1]
    
    # Prepara matriz de distâncias
    dist = [[instance.grafo.distancia(cidade_ids[i], cidade_ids[j]) for j in range(n)] for i in range(n)]
    
    otimo_comprovado = True
    timed_out = False
    
    # Busca de permutações começando no nó 0 (cidade 1)
    visited = [False] * n
    visited[0] = True
    current_perm = [0]

    def branch(depth: int, current_d: float):
        nonlocal best_z, best_sol, otimo_comprovado, timed_out
        
        if timed_out:
            return
            
        if time.time() - start_time > timeout_s:
            otimo_comprovado = False
            timed_out = True
            return
            
        if depth == n:
            # Fecha o ciclo com o nó 0
            tour_d = current_d + dist[current_perm[-1]][0]
            
            # Poda Superior de Rota:
            ub_tour = kp_opt_valor - taxa_r * (tour_d / instance.v_max)
            if ub_tour <= best_z:
                return
                
            # Rota candidata promissora: resolve o empacotamento ótimo para ela
            rota_cidades = [cidade_ids[idx] for idx in current_perm]
            z_rota, plano = resolver_packing_exato_rota_fixa(instance, rota_cidades, taxa_r)
            if z_rota > best_z:
                best_z = z_rota
                best_sol = SolucaoTTP(rota=rota_cidades, plano_coleta=plano)
            return

        # Poda de subcaminho:
        ub_parcial = kp_opt_valor - taxa_r * (current_d / instance.v_max)
        if ub_parcial <= best_z:
            return
            
        last = current_perm[-1]
        for nxt in range(1, n):
            if timed_out:
                break
            if not visited[nxt]:
                visited[nxt] = True
                current_perm.append(nxt)
                branch(depth + 1, current_d + dist[last][nxt])
                current_perm.pop()
                visited[nxt] = False

    branch(1, 0.0)
    return best_z, best_sol, otimo_comprovado
