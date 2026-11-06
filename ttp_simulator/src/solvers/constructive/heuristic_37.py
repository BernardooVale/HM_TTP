import math
import random
from typing import Dict, List, Optional, Set, Tuple
from src.core.models import TTPInstance, SolucaoTTP, Item, Grafo
from src.solvers.base import BaseSolver, SolverRegistry
from src.config.config_manager import obter_melhor_configuracao


from src.solvers.tsp import (
    resolver_tsp_farthest_insertion,
    resolver_tsp_multistart_nn,
    obter_melhor_rota_tsp,
    otimizar_2opt,
)


@SolverRegistry.register
class Heuristic37Solver(BaseSolver):
    """
    Solucionador baseado na teoria do Problema da Secretária (regra dos 37%):
    - Constrói uma rota TSP de alta qualidade (Farthest Insertion / Multi-start / 2-opt) iniciando em 1.
    - Avalia os itens na ordem sequencial da rota (ordem de chegada/descoberta).
    - Utiliza o custo marginal real de transporte (distância restante até a cidade 1).
    - Observa os primeiros 37% dos itens para calibrar o limiar de qualidade (benchmark).
    - Suporta estratégias de amostragem não-espacial: 'estratificada' (por quartis de d_rest),
      'aleatoria' (amostragem uniforme pura) e 'rota' (legada ao longo dos primeiros 37% do TSP).
    - Suporta limiar dinâmico adaptativo decrescente (Knapsack Secretary): L(W) = L0 * (W_rest / C)^alpha.
    - A partir da observação, seleciona itens que superem esse benchmark enquanto couber na mochila.
    - Opcionalmente realiza preenchimento guloso pós-secretária e refinamento Drop & Add (PackIterative).
    """

    def __init__(
        self,
        refinamento: Optional[bool] = None,
        limiar_simetrico: Optional[bool] = None,
        taxa_amostragem: Optional[float] = None,
        metodo_tsp: Optional[str] = None,
        refinamento_rota: Optional[bool] = None,
        preenchimento_pos: Optional[bool] = None,
        modo_amostragem: Optional[str] = None,
        limiar_adaptativo: Optional[bool] = None,
        alpha_adaptativo: Optional[float] = None,
        modelo: str = "TTP1",
        nome: Optional[str] = None,
    ):
        cfg = obter_melhor_configuracao("heuristic_37")
        self.refinamento = refinamento if refinamento is not None else cfg.get("refinamento", True)
        self.limiar_simetrico = limiar_simetrico if limiar_simetrico is not None else cfg.get("limiar_simetrico", False)
        self.taxa_amostragem = taxa_amostragem if taxa_amostragem is not None else cfg.get("taxa_amostragem", 0.37)
        self.metodo_tsp = metodo_tsp if metodo_tsp is not None else cfg.get("metodo_tsp", "melhor")
        self.refinamento_rota = refinamento_rota if refinamento_rota is not None else cfg.get("refinamento_rota", False)
        self.preenchimento_pos = preenchimento_pos if preenchimento_pos is not None else cfg.get("preenchimento_pos", True)
        self.modo_amostragem = modo_amostragem if modo_amostragem is not None else cfg.get("modo_amostragem", "estratificada")
        self.limiar_adaptativo = limiar_adaptativo if limiar_adaptativo is not None else cfg.get("limiar_adaptativo", True)
        self.alpha_adaptativo = alpha_adaptativo if alpha_adaptativo is not None else cfg.get("alpha_adaptativo", 0.75)
        if self.modo_amostragem not in ("rota", "aleatoria", "estratificada"):
            raise ValueError(f"Modo de amostragem inválido: {self.modo_amostragem}")
        self.modelo = modelo
        self._nome = nome or "heuristic_37"

    @property
    def nome(self) -> str:
        return self._nome

    def solve(self, instance: TTPInstance, rota: Optional[List[int]] = None) -> SolucaoTTP:
        # 1. Rota TSP iniciando na cidade 1
        if rota is not None:
            rota = list(rota)
        else:
            if self.metodo_tsp == "farthest_insertion":
                rota = resolver_tsp_farthest_insertion(instance.grafo)
            elif self.metodo_tsp == "multistart_nn":
                rota = resolver_tsp_multistart_nn(instance.grafo)
            else:
                rota = obter_melhor_rota_tsp(instance.grafo)

            if self.refinamento_rota:
                rota = otimizar_2opt(rota, instance.grafo)

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
        if m == 0:
            return SolucaoTTP(rota=rota, plano_coleta={})

        itens_coleta: List[Tuple[Item, float]] = []

        if self.modo_amostragem == "rota":
            # Modo legado: primeiros 37% ao longo da rota TSP
            limite_calibracao = max(1, int(self.taxa_amostragem * m))
            itens_calibracao = itens_sequenciais[:limite_calibracao]
            scores_calibracao = [calcular_score(item, 0.0) for item in itens_calibracao]
            limiar_base = max(0.0, max(scores_calibracao)) if scores_calibracao else 0.0
            itens_coleta = [(item, limiar_base) for item in itens_sequenciais[limite_calibracao:]]

        elif self.modo_amostragem == "aleatoria":
            # Amostragem desacoplada da rota (amostragem aleatória uniforme com seed estável)
            qtd_amostra = max(1, int(self.taxa_amostragem * m))
            rng = random.Random(42)
            indices_amostra = set(rng.sample(range(m), qtd_amostra))
            itens_calibracao = [itens_sequenciais[i] for i in indices_amostra]
            scores_calibracao = [calcular_score(item, 0.0) for item in itens_calibracao]
            limiar_base = max(0.0, max(scores_calibracao)) if scores_calibracao else 0.0
            itens_coleta = [
                (item, limiar_base)
                for i, item in enumerate(itens_sequenciais)
                if i not in indices_amostra
            ]

        elif self.modo_amostragem == "estratificada":
            # Modo Estratificado: divide a rota em 4 quartis de distância ao depósito e calibra limiares independentes por quartil
            d_max = max(distancias_restantes.values()) if distancias_restantes else 1.0
            if d_max <= 0:
                d_max = 1.0

            itens_por_quartil: Dict[int, List[Item]] = {q: [] for q in range(4)}
            for item in itens_sequenciais:
                d_r = distancias_restantes.get(item.cidade_origem, 0.0)
                norm = d_r / d_max
                q = min(3, max(0, int(norm * 4)))
                itens_por_quartil[q].append(item)

            limiar_quartil: Dict[int, float] = {}
            calibracao_ids: Set[int] = set()

            for q in range(4):
                itens_q = itens_por_quartil[q]
                m_q = len(itens_q)
                if m_q > 0:
                    limite_q = max(1, int(self.taxa_amostragem * m_q))
                    amostra_q = itens_q[:limite_q]
                    for it in amostra_q:
                        calibracao_ids.add(it.id)
                    scores_q = [calcular_score(it, 0.0) for it in amostra_q]
                    limiar_quartil[q] = max(0.0, max(scores_q)) if scores_q else 0.0
                else:
                    limiar_quartil[q] = 0.0

            for item in itens_sequenciais:
                if item.id not in calibracao_ids:
                    d_r = distancias_restantes.get(item.cidade_origem, 0.0)
                    norm = d_r / d_max
                    q = min(3, max(0, int(norm * 4)))
                    itens_coleta.append((item, limiar_quartil[q]))

        # Fase de decisão/coleta da secretária:
        peso_atual = 0.0
        plano_coleta: Dict[int, List[int]] = {}
        itens_coletados_ids = set()

        for item, limiar_base in itens_coleta:
            if peso_atual + item.peso <= instance.capacidade_mochila:
                # Limiar dinâmico adaptativo decrescente (Knapsack Secretary): L(W) = L0 * (W_rest / C)^alpha
                if self.limiar_adaptativo:
                    w_rest = max(0.0, instance.capacidade_mochila - peso_atual)
                    fator_adaptativo = (w_rest / cap_mochila) ** self.alpha_adaptativo
                    limiar_efetivo = limiar_base * fator_adaptativo
                else:
                    limiar_efetivo = limiar_base

                lucro_marginal = calcular_lucro_marginal(item, peso_atual)
                if lucro_marginal <= 0:
                    continue

                if self.limiar_simetrico:
                    score = calcular_score(item, 0.0)
                else:
                    score = calcular_score(item, peso_atual)

                if score > limiar_efetivo:
                    peso_atual += item.peso
                    itens_coletados_ids.add(item.id)
                    if item.cidade_origem not in plano_coleta:
                        plano_coleta[item.cidade_origem] = []
                    plano_coleta[item.cidade_origem].append(item.id)

        # 5. Preenchimento guloso pós-secretária:
        if self.preenchimento_pos:
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
            from src.solvers.iterative.pack_iterative import refinar_plano_coleta
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
SolverRegistry.register(
    type(
        "Heuristic37SemRefinamentoSolver",
        (Heuristic37Solver,),
        {"nome": "heuristic_37_sem_refinamento", "__init__": lambda self: Heuristic37Solver.__init__(self, refinamento=False, nome="heuristic_37_sem_refinamento")},
    )
)
SolverRegistry.register(
    type(
        "Heuristic37EstratificadaSolver",
        (Heuristic37Solver,),
        {"nome": "heuristic_37_estratificada", "__init__": lambda self: Heuristic37Solver.__init__(self, modo_amostragem="estratificada", limiar_adaptativo=True, nome="heuristic_37_estratificada")},
    )
)
SolverRegistry.register(
    type(
        "Heuristic37AleatoriaSolver",
        (Heuristic37Solver,),
        {"nome": "heuristic_37_aleatoria", "__init__": lambda self: Heuristic37Solver.__init__(self, modo_amostragem="aleatoria", limiar_adaptativo=True, nome="heuristic_37_aleatoria")},
    )
)
SolverRegistry.register(
    type(
        "Heuristic37AdaptativaSolver",
        (Heuristic37Solver,),
        {"nome": "heuristic_37_adaptativa", "__init__": lambda self: Heuristic37Solver.__init__(self, limiar_adaptativo=True, nome="heuristic_37_adaptativa")},
    )
)
