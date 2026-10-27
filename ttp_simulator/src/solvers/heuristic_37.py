import math
from typing import Dict, List, Optional
from src.core.models import TTPInstance, SolucaoTTP, Item, Grafo
from src.solvers.base import BaseSolver, SolverRegistry


from src.solvers.tsp import (
    resolver_tsp_farthest_insertion,
    resolver_tsp_multistart_nn,
    obter_melhor_rota_tsp,
)


@SolverRegistry.register
class Heuristic37Solver(BaseSolver):
    """
    Solucionador baseado na teoria do Problema da Secretária (regra dos 37%):
    - Constrói uma rota TSP de alta qualidade (Farthest Insertion / Multi-start) iniciando em 1.
    - Avalia os itens na ordem sequencial da rota (ordem de chegada/descoberta).
    - Utiliza o custo marginal real de transporte (distância restante até a cidade 1).
    - Observa os primeiros 37% dos itens para calibrar o limiar de qualidade (benchmark).
    - A partir dos 37%, seleciona itens que superem esse benchmark enquanto couber na mochila.
    """

    def __init__(
        self,
        refinamento: bool = False,
        limiar_simetrico: bool = False,
        modelo: str = "TTP1",
        nome: Optional[str] = None,
    ):
        self.refinamento = refinamento
        self.limiar_simetrico = limiar_simetrico
        self.modelo = modelo
        self._nome = nome or ("heuristic_37_refinada" if refinamento else "heuristic_37")

    @property
    def nome(self) -> str:
        return self._nome

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        # 1. Rota TSP de menor distância iniciando na cidade 1
        rota = obter_melhor_rota_tsp(instance.grafo)
        n = len(rota)

        # 2. Distâncias restantes de cada cidade até o fechamento do ciclo na cidade 1
        distancias_restantes = {}
        dist_acc = instance.grafo.distancia(rota[-1], rota[0])
        distancias_restantes[rota[-1]] = dist_acc

        for i in range(n - 2, -1, -1):
            dist_acc += instance.grafo.distancia(rota[i], rota[i + 1])
            distancias_restantes[rota[i]] = dist_acc

        # 3. Cálculo de velocidade física e impacto marginal no tempo de viagem
        cap_mochila = max(instance.capacidade_mochila, 1e-6)
        v_min = instance.v_min
        v_max = instance.v_max
        taxa_aluguel = instance.taxa_aluguel
        const_deg = max(instance.constante_degradacao, 1e-6)

        def velocidade(peso: float) -> float:
            return max(v_min, v_max - peso * ((v_max - v_min) / cap_mochila))

        def calcular_lucro_marginal(item: Item, peso_mochila: float) -> float:
            """
            Calcula o lucro líquido adicional exato ao adicionar este item à mochila
            já carregando peso_mochila. Suporta tanto TTP1 (aluguel) quanto TTP2 (degradação).
            """
            if item.peso <= 0:
                return item.valor
            d_restante = distancias_restantes.get(item.cidade_origem, 0.0)

            if self.modelo == "TTP2":
                v_depois = velocidade(peso_mochila + item.peso)
                tempo_restante = d_restante / v_depois
                degradacao = math.ceil(tempo_restante / const_deg)
                return item.valor - degradacao
            else:
                v_antes = velocidade(peso_mochila)
                v_depois = velocidade(peso_mochila + item.peso)
                delta_tempo = d_restante * (1.0 / v_depois - 1.0 / v_antes)
                custo_transporte = taxa_aluguel * delta_tempo
                return item.valor - custo_transporte

        def calcular_score(item: Item, peso_mochila: float = 0.0) -> float:
            if item.peso <= 0:
                return float("inf")
            lucro = calcular_lucro_marginal(item, peso_mochila)
            return lucro / item.peso

        # 4. Sequência da secretária:
        # Itens dispostos na ordem natural em que são visitados ao longo da rota
        itens_sequenciais: List[Item] = []
        for cidade_id in rota:
            cidade = instance.grafo.cidades[cidade_id]
            itens_sequenciais.extend(cidade.itens)

        m = len(itens_sequenciais)
        limite_calibracao = int(0.37 * m)
        itens_calibracao = itens_sequenciais[:limite_calibracao]
        itens_coleta = itens_sequenciais[limite_calibracao:]

        # Fase de calibração: observa os scores intrínsecos dos primeiros 37% (avaliados em mochila vazia)
        scores_calibracao = [calcular_score(item, 0.0) for item in itens_calibracao]
        melhor_threshold = max(scores_calibracao) if scores_calibracao else 0.0
        limiar = max(melhor_threshold, 0.0)

        # Fase de decisão/coleta da secretária:
        peso_atual = 0.0
        plano_coleta: Dict[int, List[int]] = {}
        itens_coletados_ids = set()

        for item in itens_coleta:
            if peso_atual + item.peso <= instance.capacidade_mochila:
                if self.limiar_simetrico:
                    # Modo Simétrico: compara score intrínseco (mochila vazia) com o limiar
                    score_intrinseco = calcular_score(item, 0.0)
                    lucro_marginal = calcular_lucro_marginal(item, peso_atual)
                    if score_intrinseco > limiar and lucro_marginal > 0:
                        peso_atual += item.peso
                        itens_coletados_ids.add(item.id)
                        if item.cidade_origem not in plano_coleta:
                            plano_coleta[item.cidade_origem] = []
                        plano_coleta[item.cidade_origem].append(item.id)
                else:
                    # Modo Padrão: avalia score dinâmico dependente do peso acumulado
                    score = calcular_score(item, peso_atual)
                    if score > limiar:
                        peso_atual += item.peso
                        itens_coletados_ids.add(item.id)
                        if item.cidade_origem not in plano_coleta:
                            plano_coleta[item.cidade_origem] = []
                        plano_coleta[item.cidade_origem].append(item.id)

        # 5. Preenchimento guloso pós-secretária:
        itens_candidatos = [it for it in instance.itens if it.id not in itens_coletados_ids]

        while peso_atual < instance.capacidade_mochila:
            melhor_item = None
            melhor_densidade = 0.0

            for it in itens_candidatos:
                if peso_atual + it.peso <= instance.capacidade_mochila:
                    lucro_adicional = calcular_lucro_marginal(it, peso_atual)
                    if lucro_adicional > 0:
                        densidade = lucro_adicional / it.peso if it.peso > 0 else float("inf")
                        if densidade > melhor_densidade:
                            melhor_densidade = densidade
                            melhor_item = it

            if melhor_item is not None:
                peso_atual += melhor_item.peso
                itens_coletados_ids.add(melhor_item.id)
                itens_candidatos.remove(melhor_item)
                if melhor_item.cidade_origem not in plano_coleta:
                    plano_coleta[melhor_item.cidade_origem] = []
                plano_coleta[melhor_item.cidade_origem].append(melhor_item.id)
            else:
                break

        if self.refinamento:
            from src.solvers.pack_iterative import refinar_plano_coleta
            plano_coleta = refinar_plano_coleta(
                instance, rota, plano_coleta, modelo=self.modelo
            )

        return SolucaoTTP(rota=rota, plano_coleta=plano_coleta)


# Registro de variações para comparação direta
SolverRegistry.register(
    type(
        "Heuristic37RefinadaSolver",
        (Heuristic37Solver,),
        {"nome": "heuristic_37_refinada", "__init__": lambda self: Heuristic37Solver.__init__(self, refinamento=True, nome="heuristic_37_refinada")},
    )
)
SolverRegistry.register(
    type(
        "Heuristic37SimetricaSolver",
        (Heuristic37Solver,),
        {"nome": "heuristic_37_simetrica", "__init__": lambda self: Heuristic37Solver.__init__(self, limiar_simetrico=True, nome="heuristic_37_simetrica")},
    )
)
SolverRegistry.register(
    type(
        "Heuristic37TTP2Solver",
        (Heuristic37Solver,),
        {"nome": "heuristic_37_ttp2", "__init__": lambda self: Heuristic37Solver.__init__(self, modelo="TTP2", nome="heuristic_37_ttp2")},
    )
)