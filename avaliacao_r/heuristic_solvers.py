"""
Solucionadores Gulosos e Heurísticos Auxiliares para o TTP.

Usados para comparar:
- Solução Ótima de um + Gulosa do outro vs.
- Solução Ótima de um + Ótima Condicionada do outro.
"""

from typing import List, Dict, Tuple
import math

import sys
from pathlib import Path
_sim_path = str(Path(__file__).resolve().parent.parent / "ttp_simulator")
if _sim_path not in sys.path:
    sys.path.insert(0, _sim_path)

from src.core.models import TTPInstance, Grafo, Item, SolucaoTTP


def resolver_kp_guloso(itens: List[Item], capacidade: float) -> Tuple[float, float, List[int]]:
    """
    Resolve o KP de forma gulosa baseado na densidade de valor (valor / peso).
    Retorna: (lucro_total, peso_total, lista_ids_itens).
    """
    itens_validos = [it for it in itens if it.peso <= capacidade and it.valor > 0]
    sorted_items = sorted(
        itens_validos, 
        key=lambda it: it.valor / it.peso if it.peso > 0 else float('inf'), 
        reverse=True
    )
    
    peso_atual = 0.0
    lucro_total = 0.0
    selecionados = []
    
    for it in sorted_items:
        if peso_atual + it.peso <= capacidade:
            peso_atual += it.peso
            lucro_total += it.valor
            selecionados.append(it.id)
            
    return lucro_total, peso_atual, selecionados


def resolver_tsp_guloso(grafo: Grafo) -> Tuple[float, List[int]]:
    """
    Gera uma rota TSP via Vizinho Mais Próximo (Nearest Neighbor) iniciando na cidade 1.
    Retorna: (distancia_total, rota).
    """
    cidades = list(grafo.cidades.keys())
    nao_visitadas = set(cidades)
    nao_visitadas.discard(1)
    
    rota = [1]
    curr = 1
    total_dist = 0.0
    
    while nao_visitadas:
        proxima = min(nao_visitadas, key=lambda c: grafo.distancia(curr, c))
        total_dist += grafo.distancia(curr, proxima)
        rota.append(proxima)
        nao_visitadas.remove(proxima)
        curr = proxima
        
    total_dist += grafo.distancia(curr, 1)
    return total_dist, rota


def resolver_rota_gulosa_itens_fixos(
    grafo: Grafo, 
    itens_selecionados: List[Item], 
    capacidade: float, 
    v_min: float, 
    v_max: float
) -> Tuple[float, List[int]]:
    """
    Constrói uma rota gulosa (Nearest Neighbor) a partir da cidade 1 para visitar
    todas as cidades coletando os itens pré-determinados.
    """
    cidades = list(grafo.cidades.keys())
    nao_visitadas = set(cidades)
    nao_visitadas.discard(1)
    
    rota = [1]
    curr = 1
    peso_acc = sum(it.peso for it in itens_selecionados if it.cidade_origem == 1)
    tempo_total = 0.0
    
    while nao_visitadas:
        v = max(v_max - peso_acc * ((v_max - v_min) / capacidade), v_min)
        proxima = min(nao_visitadas, key=lambda c: grafo.distancia(curr, c))
        d = grafo.distancia(curr, proxima)
        tempo_total += d / v
        rota.append(proxima)
        nao_visitadas.remove(proxima)
        curr = proxima
        peso_acc = min(peso_acc + sum(it.peso for it in itens_selecionados if it.cidade_origem == curr), capacidade)
        
    v_final = max(v_max - peso_acc * ((v_max - v_min) / capacidade), v_min)
    tempo_total += grafo.distancia(curr, 1) / v_final
    return tempo_total, rota


def resolver_packing_guloso_rota_fixa(
    instance: TTPInstance, 
    rota: List[int], 
    taxa_r: float
) -> Tuple[float, Dict[int, List[int]]]:
    """
    Gera um plano de coleta guloso sobre uma rota fixa usando a heurística S2
    (densidade ajustada pela distância restante até o fim do tour).
    """
    itens = [it for it in instance.itens if it.cidade_origem != 1 and it.peso <= instance.capacidade_mochila]
    if not itens:
        return 0.0, {}
        
    n_cidades = len(rota)
    pos_cidade = {cid: idx for idx, cid in enumerate(rota)}
    
    # Pré-calcula distâncias restantes
    pernas_dist = [instance.grafo.distancia(rota[i], rota[(i + 1) % n_cidades]) for i in range(n_cidades)]
    dist_restante = [0.0] * n_cidades
    acc = 0.0
    for i in range(n_cidades - 1, -1, -1):
        acc += pernas_dist[i]
        dist_restante[i] = acc

    # Escore S2: v_i / (w_i * dist_restante)
    def escore_s2(it: Item) -> float:
        idx = pos_cidade.get(it.cidade_origem, 0)
        d_rest = max(dist_restante[idx], 0.01)
        return it.valor / (it.peso * d_rest) if it.peso > 0 else float('inf')
        
    sorted_itens = sorted(itens, key=escore_s2, reverse=True)
    
    peso_atual = 0.0
    plano_dict: Dict[int, List[int]] = {}
    itens_escolhidos = []
    
    # Coleta gulosamente enquanto for vantajoso
    for it in sorted_itens:
        if peso_atual + it.peso <= instance.capacidade_mochila:
            plano_dict.setdefault(it.cidade_origem, []).append(it.id)
            itens_escolhidos.append(it)
            peso_atual += it.peso

    # Avaliação do lucro líquido com avaliador padrão
    solucao = SolucaoTTP(rota=rota, plano_coleta=plano_dict)
    from src.core.evaluator_ttp1 import avaliar_ttp1
    
    orig_r = instance.taxa_aluguel
    instance.taxa_aluguel = taxa_r
    res = avaliar_ttp1(instance, solucao)
    instance.taxa_aluguel = orig_r
    
    return res['lucro_liquido_G'], plano_dict
