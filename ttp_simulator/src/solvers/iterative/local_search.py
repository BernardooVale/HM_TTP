import random
import time
from typing import Dict, List, Optional
from src.core.models import TTPInstance, SolucaoTTP, Item
from src.solvers.base import BaseSolver, SolverRegistry
from src.core.evaluator_ttp1 import avaliar_ttp1
from src.solvers.constructive.baselines import RandomSolver

@SolverRegistry.register
class AlternateLocalSearchSolver(BaseSolver):
    def __init__(self, time_limit: float = 5.0):
        self.time_limit = time_limit

    @property
    def nome(self) -> str:
        return "local_search"
    
    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        random_solver = RandomSolver()
        solucao = random_solver.solve(instance)

        t0 = time.perf_counter()
        deadline = (t0 + self.time_limit) if (self.time_limit and self.time_limit > 0) else None
        
        max_iterations = 100
        
        for _ in range(max_iterations):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            melhorou = False
            
            rota_melhorada, rota_mudou = self._otimizar_rota(instance, solucao, deadline=deadline)
            if rota_mudou:
                solucao.rota = rota_melhorada
                melhorou = True

            if deadline is not None and time.perf_counter() >= deadline:
                break
                
            plano_melhorado, plano_mudou = self._otimizar_plano_coleta(instance, solucao, deadline=deadline)
            if plano_mudou:
                solucao.plano_coleta = plano_melhorado
                melhorou = True
                
            if not melhorou:
                break
                
        return solucao
        
    def _otimizar_rota(
        self,
        instance: TTPInstance,
        solucao: SolucaoTTP,
        deadline: Optional[float] = None,
    ) -> tuple[List[int], bool]:
        rota = list(solucao.rota)
        n = len(rota)
        melhorou = False
        obj_atual = avaliar_ttp1(instance, SolucaoTTP(rota=rota, plano_coleta=solucao.plano_coleta)).get("lucro_liquido_G", float('-inf'))
        
        for i in range(1, n - 1):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            for j in range(i + 1, n):
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                nova_rota = rota[:i] + rota[i:j+1][::-1] + rota[j+1:]
                
                obj_novo = avaliar_ttp1(instance, SolucaoTTP(rota=nova_rota, plano_coleta=solucao.plano_coleta)).get("lucro_liquido_G", float('-inf'))
                
                if obj_novo > obj_atual:
                    rota = nova_rota
                    melhorou = True
                    break
            if melhorou or (deadline is not None and time.perf_counter() >= deadline):
                break
                
        return rota, melhorou
        
    def _otimizar_plano_coleta(
        self,
        instance: TTPInstance,
        solucao: SolucaoTTP,
        deadline: Optional[float] = None,
    ) -> tuple[Dict[int, List[int]], bool]:
        plano = {k: list(v) for k, v in solucao.plano_coleta.items()}
        itens_coletados_ids = set(solucao.itens_coletados_ids())
        peso_atual = sum(item.peso for item in instance.itens if item.id in itens_coletados_ids)
        
        melhorou = False
        melhor_obj = avaliar_ttp1(instance, solucao).get("lucro_liquido_G", float('-inf'))
        
        for item in instance.itens:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if item.id in itens_coletados_ids:
                plano[item.cidade_origem].remove(item.id)
                obj_novo = avaliar_ttp1(instance, SolucaoTTP(rota=solucao.rota, plano_coleta=plano)).get("lucro_liquido_G", float('-inf'))
                
                if obj_novo > melhor_obj:
                    melhor_obj = obj_novo
                    itens_coletados_ids.remove(item.id)
                    peso_atual -= item.peso
                    melhorou = True
                else:
                    plano[item.cidade_origem].append(item.id)
            else:
                if peso_atual + item.peso <= instance.capacidade_mochila:
                    if item.cidade_origem not in plano:
                        plano[item.cidade_origem] = []
                    plano[item.cidade_origem].append(item.id)
                    
                    obj_novo = avaliar_ttp1(instance, SolucaoTTP(rota=solucao.rota, plano_coleta=plano)).get("lucro_liquido_G", float('-inf'))
                    
                    if obj_novo > melhor_obj:
                        melhor_obj = obj_novo
                        itens_coletados_ids.add(item.id)
                        peso_atual += item.peso
                        melhorou = True
                    else:
                        plano[item.cidade_origem].remove(item.id)
                        
        return plano, melhorou
