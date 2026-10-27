"""
Solucionador PackIterative para o Traveling Thief Problem (TTP).

Referência bibliográfica:
Faulkner, H., Polyakovskiy, S., Schultz, T., & Wagner, M. (2015).
"Solving the Traveling Thief Problem: PackIterative and Co-Solver Approaches".
In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO 2015), pp. 385-392.
IEEE Congress on Evolutionary Computation (CEC 2015).
"""

from typing import Dict, List, Optional, Set
from src.core.models import TTPInstance, SolucaoTTP, Item
from src.core.fast_evaluator import FastTTPEvaluator
from src.solvers.base import BaseSolver, SolverRegistry
from src.solvers.tsp import obter_melhor_rota_tsp


def refinar_plano_coleta(
    instance: TTPInstance,
    rota: List[int],
    plano_coleta: Dict[int, List[int]],
    max_iter: int = 50,
    modelo: str = "TTP1",
) -> Dict[int, List[int]]:
    """
    Busca local 1-bit-flip (Drop & Add / PackIterative refinement).
    
    1. Fase DROP: Remove qualquer item da mochila cujo lucro marginal líquido seja negativo
       (ou seja, cuja remoção estritamente aumente o objetivo G).
    2. Fase ADD: Insere itens que caibam na capacidade restante e tragam ganho líquido estritamente positivo.
    3. Itera até convergência ou atingir max_iter. Suporta TTP1 e TTP2.
    """
    evaluator = FastTTPEvaluator(instance, rota, modelo=modelo)
    itens_by_id = {it.id: it for it in instance.itens}

    itens_coletados: Set[int] = set()
    for it_ids in plano_coleta.values():
        for i_id in it_ids:
            if i_id in itens_by_id:
                itens_coletados.add(i_id)

    peso_atual = sum(itens_by_id[i_id].peso for i_id in itens_coletados)

    for _ in range(max_iter):
        mudou = False

        # --- FASE DROP: Elimina itens deficitários ---
        while True:
            melhor_item_drop: Optional[int] = None
            maior_ganho_drop = 0.0
            obj_atual = evaluator.calcular_lucro(itens_coletados)

            for it_id in list(itens_coletados):
                itens_coletados.remove(it_id)
                obj_sem = evaluator.calcular_lucro(itens_coletados)
                itens_coletados.add(it_id)

                ganho = obj_sem - obj_atual
                if ganho > maior_ganho_drop:
                    maior_ganho_drop = ganho
                    melhor_item_drop = it_id

            if melhor_item_drop is not None and maior_ganho_drop > 1e-6:
                itens_coletados.remove(melhor_item_drop)
                peso_atual -= itens_by_id[melhor_item_drop].peso
                mudou = True
            else:
                break

        # --- FASE ADD: Adiciona itens lucrativos que cabem na mochila ---
        while True:
            melhor_item_add: Optional[int] = None
            maior_ganho_add = 0.0
            obj_atual = evaluator.calcular_lucro(itens_coletados)

            for it in instance.itens:
                if it.id in itens_coletados:
                    continue
                if peso_atual + it.peso > instance.capacidade_mochila:
                    continue

                itens_coletados.add(it.id)
                obj_com = evaluator.calcular_lucro(itens_coletados)
                itens_coletados.remove(it.id)

                ganho = obj_com - obj_atual
                if ganho > maior_ganho_add:
                    maior_ganho_add = ganho
                    melhor_item_add = it.id

            if melhor_item_add is not None and maior_ganho_add > 1e-6:
                itens_coletados.add(melhor_item_add)
                peso_atual += itens_by_id[melhor_item_add].peso
                mudou = True
            else:
                break

        if not mudou:
            break

    # Reconstrução do plano_coleta
    novo_plano: Dict[int, List[int]] = {}
    for it_id in itens_coletados:
        origem = itens_by_id[it_id].cidade_origem
        if origem not in novo_plano:
            novo_plano[origem] = []
        novo_plano[origem].append(it_id)

    return novo_plano


def resolver_pack_iterative(
    instance: TTPInstance,
    rota: List[int],
    max_refinamento_iter: int = 50,
    modelo: str = "TTP1",
) -> Dict[int, List[int]]:
    """
    Algoritmo PackIterative (Faulkner et al., 2015).
    Gera um plano de coleta inicial baseado em escore marginal e aplica busca local Drop & Add.
    """
    n = len(rota)
    dist_restante: Dict[int, float] = {}
    dist_acc = instance.grafo.distancia(rota[-1], rota[0])
    dist_restante[rota[-1]] = dist_acc

    for i in range(n - 2, -1, -1):
        dist_acc += instance.grafo.distancia(rota[i], rota[i + 1])
        dist_restante[rota[i]] = dist_acc

    # Ordena itens por razão de valor dividido pela distância restante até o destino
    itens_ordenados = sorted(
        instance.itens,
        key=lambda it: it.valor / (it.peso * max(dist_restante.get(it.cidade_origem, 1.0), 0.001))
        if it.peso > 0 else float("inf"),
        reverse=True,
    )

    evaluator = FastTTPEvaluator(instance, rota, modelo=modelo)
    itens_coletados: Set[int] = set()
    peso_atual = 0.0
    obj_atual = evaluator.calcular_lucro(itens_coletados)

    # Inserção inicial baseada em ganho positivo
    for it in itens_ordenados:
        if it.cidade_origem == 1:
            continue
        if peso_atual + it.peso <= instance.capacidade_mochila:
            itens_coletados.add(it.id)
            obj_com = evaluator.calcular_lucro(itens_coletados)
            if obj_com > obj_atual:
                peso_atual += it.peso
                obj_atual = obj_com
            else:
                itens_coletados.remove(it.id)

    plano_inicial: Dict[int, List[int]] = {}
    for it_id in itens_coletados:
        it = evaluator.itens_by_id[it_id]
        if it.cidade_origem not in plano_inicial:
            plano_inicial[it.cidade_origem] = []
        plano_inicial[it.cidade_origem].append(it_id)

    # Refinamento Drop & Add
    return refinar_plano_coleta(
        instance, rota, plano_inicial, max_iter=max_refinamento_iter, modelo=modelo
    )


@SolverRegistry.register
class PackIterativeSolver(BaseSolver):
    """
    Solucionador PackIterative para o TTP (Faulkner et al., 2015).
    """

    def __init__(self, modelo: str = "TTP1", nome: Optional[str] = None):
        self.modelo = modelo
        self._nome = nome or "pack_iterative"

    @property
    def nome(self) -> str:
        return self._nome

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        rota = obter_melhor_rota_tsp(instance.grafo)
        plano_coleta = resolver_pack_iterative(instance, rota, modelo=self.modelo)
        return SolucaoTTP(rota=rota, plano_coleta=plano_coleta)

