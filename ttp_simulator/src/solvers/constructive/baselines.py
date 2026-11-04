import random
from src.core.models import TTPInstance, SolucaoTTP, Item
from src.solvers.base import BaseSolver, SolverRegistry

@SolverRegistry.register
class RandomSolver(BaseSolver):
    @property
    def nome(self) -> str:
        return "random"
    
    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        cidades = list(instance.grafo.cidades.keys())
        if 1 in cidades:
            cidades.remove(1)
        random.shuffle(cidades)
        rota = [1] + cidades
        
        itens_embaralhados = list(instance.itens)
        random.shuffle(itens_embaralhados)
        
        peso_atual = 0.0
        plano_coleta = {}
        for item in itens_embaralhados:
            if peso_atual + item.peso <= instance.capacidade_mochila:
                peso_atual += item.peso
                if item.cidade_origem not in plano_coleta:
                    plano_coleta[item.cidade_origem] = []
                plano_coleta[item.cidade_origem].append(item.id)
                
        return SolucaoTTP(rota=rota, plano_coleta=plano_coleta)

@SolverRegistry.register
class GreedyIsolatedSolver(BaseSolver):
    @property
    def nome(self) -> str:
        return "greedy"
    
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
            
        itens_ordenados = sorted(instance.itens, key=lambda x: x.valor / x.peso if x.peso > 0 else float('inf'), reverse=True)
        
        peso_atual = 0.0
        plano_coleta = {}
        for item in itens_ordenados:
            if peso_atual + item.peso <= instance.capacidade_mochila:
                peso_atual += item.peso
                if item.cidade_origem not in plano_coleta:
                    plano_coleta[item.cidade_origem] = []
                plano_coleta[item.cidade_origem].append(item.id)
                
        return SolucaoTTP(rota=rota, plano_coleta=plano_coleta)
