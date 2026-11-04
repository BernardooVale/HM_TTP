"""
Template Limpo para Implementação de uma Nova Heurística no TTP.

COMO USAR:
1. Copie este arquivo para um novo nome em ttp_simulator/src/solvers/, ex:
   'minha_heuristica.py'.
2. Altere o nome no decorator @SolverRegistry.register("minha_heuristica")
   e na propriedade `nome`.
3. Implemente a lógica do Tour e do Plano de Coleta dentro de solve().
4. Teste imediatamente no terminal com:
   python testar_heuristica.py --solver minha_heuristica --comparar
"""

from typing import Dict, List
from src.core.models import SolucaoTTP, TTPInstance
from src.solvers.base import BaseSolver, SolverRegistry


# @SolverRegistry.register("minha_heuristica")  # Descomente para registrar seu solver
class TemplateHeuristicaSolver(BaseSolver):
    """Exemplo prático e documentado para criar sua própria heurística TTP."""

    @property
    def nome(self) -> str:
        return "template_heuristica"

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        """
        Método principal de resolução.

        Parâmetros disponíveis em `instance`:
        - instance.grafo: Grafo de cidades
          * instance.grafo.cidades: dict {id_cidade: Cidade(id, x, y, itens)}
          * instance.grafo.distancia(c1, c2): distância euclidiana entre duas cidades
          * instance.num_cities: total de cidades (N)
        - instance.itens: Lista de todos os itens disponíveis
          * item.id: id único
          * item.peso: peso em kg
          * item.valor: valor monetário
          * item.cidade_origem: id da cidade onde o item está localizado
        - instance.capacidade_mochila: Capacidade máxima de peso (W)
        - instance.v_min, instance.v_max: Velocidades mínima e máxima do ladrão
        - instance.taxa_aluguel: Taxa de aluguel por unidade de tempo (R)

        Retorno esperado:
        - SolucaoTTP(rota=tour, plano_coleta=plano_coleta)
          * rota: lista com os IDs de todas as cidades [1, ..., N] iniciando em 1.
          * plano_coleta: dict {cidade_id: [item_id_1, item_id_2, ...]} com itens roubados.
        """
        num_cidades = instance.num_cities
        cap_max = instance.capacidade_mochila

        # =====================================================================
        # 1. CONSTRUÇÃO DO TOUR (ROTA)
        # Exemplo simples: Visita as cidades na ordem sequencial [1, 2, ..., N]
        # (Substitua por Nearest Neighbor, 2-opt, Lin-Kernighan, etc.)
        # =====================================================================
        tour = list(range(1, num_cidades + 1))

        # =====================================================================
        # 2. CONSTRUÇÃO DO PLANO DE COLETA (PACKING PLAN)
        # Exemplo: Seleciona itens gulosamente por densidade (valor / peso)
        # agrupando por cidade e respeitando a capacidade máxima W.
        # =====================================================================
        plano_coleta: Dict[int, List[int]] = {c: [] for c in tour}
        peso_acumulado = 0.0

        # Ordena itens da instância por maior razão valor/peso
        itens_ordenados = sorted(
            instance.itens,
            key=lambda it: it.valor / it.peso if it.peso > 0 else 0,
            reverse=True,
        )

        for item in itens_ordenados:
            if peso_acumulado + item.peso <= cap_max:
                plano_coleta[item.cidade_origem].append(item.id)
                peso_acumulado += item.peso

        # =====================================================================
        # 3. RETORNO DA SOLUÇÃO
        # =====================================================================
        return SolucaoTTP(rota=tour, plano_coleta=plano_coleta)
