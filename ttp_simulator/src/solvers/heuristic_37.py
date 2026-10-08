import math
from typing import Dict, List
from src.core.models import TTPInstance, SolucaoTTP, Item
from src.solvers.base import BaseSolver, SolverRegistry

@SolverRegistry.register
class Heuristic37Solver(BaseSolver):
    @property
    def nome(self) -> str:
        return "heuristic_37"
    
    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        cidades_nao_visitadas = set(instance.grafo.cidades.keys())
        cidades_nao_visitadas.discard(1)
        rota = [1]
        cidade_atual = 1
        
        while cidades_nao_visitadas:
            proxima_cidade = min(cidades_nao_visitadas, key=lambda c: instance.grafo.distancia(cidade_atual, c))
            rota.append(proxima_cidade)
            cidades_nao_visitadas.remove(proxima_cidade)
            cidade_atual = proxima_cidade
            
        distancias_acumuladas = {1: 0.0}
        dist_atual = 0.0
        for i in range(len(rota) - 1):
            dist_atual += instance.grafo.distancia(rota[i], rota[i+1])
            distancias_acumuladas[rota[i+1]] = dist_atual
            
        alpha = 0.5
        def calcular_score(item: Item) -> float:
            dist_impact = distancias_acumuladas.get(item.cidade_origem, 0.0)
            return (item.valor / item.peso) - alpha * (dist_impact * item.peso / instance.capacidade_mochila) if item.peso > 0 else float('inf')

        itens_ordenados = sorted(instance.itens, key=calcular_score, reverse=True)
        
        m = len(itens_ordenados)
        limite_calibracao = int(0.37 * m)
        itens_calibracao = itens_ordenados[:limite_calibracao]
        itens_coleta = itens_ordenados[limite_calibracao:]
        
        melhor_threshold = float('-inf')
        for item in itens_calibracao:
            score = calcular_score(item)
            if score > melhor_threshold:
                melhor_threshold = score
                
        peso_atual = 0.0
        plano_coleta: Dict[int, List[int]] = {}
        
        for item in itens_coleta:
            if calcular_score(item) >= melhor_threshold:
                if peso_atual + item.peso <= instance.capacidade_mochila:
                    peso_atual += item.peso
                    if item.cidade_origem not in plano_coleta:
                        plano_coleta[item.cidade_origem] = []
                    plano_coleta[item.cidade_origem].append(item.id)
                    
        return SolucaoTTP(rota=rota, plano_coleta=plano_coleta)
