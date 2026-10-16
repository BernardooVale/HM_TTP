import heapq
import math
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from src.core.models import Item, SolucaoTTP, TTPInstance
from src.solvers.base import BaseSolver, SolverRegistry


def resolver_knapsack_01_exato(itens: List[Item], capacidade: float) -> Tuple[float, List[Item]]:
    """
    Resolve o Problema da Mochila 0/1 de forma exata usando Branch and Bound com poda.
    Retorna (valor_otimo, lista_itens_selecionados).
    """
    if not itens or capacidade <= 0:
        return 0.0, []

    peso_total = sum(it.peso for it in itens)
    if peso_total <= capacidade:
        return sum(it.valor for it in itens), list(itens)

    # Ordena por densidade decrescente para limites superiores eficientes
    itens_ordenados = sorted(
        itens,
        key=lambda it: it.valor / it.peso if it.peso > 0 else float("inf"),
        reverse=True,
    )
    n = len(itens_ordenados)

    melhor_valor = [0.0]
    melhor_selecao = [[]]

    def limite_superior(idx: int, peso_atual: float, valor_atual: float) -> float:
        limite = valor_atual
        peso = peso_atual
        for i in range(idx, n):
            it = itens_ordenados[i]
            if peso + it.peso <= capacidade:
                peso += it.peso
                limite += it.valor
            else:
                if it.peso > 0:
                    limite += (capacidade - peso) * (it.valor / it.peso)
                break
        return limite

    def branch_and_bound(idx: int, peso_atual: float, valor_atual: float, selecionados: List[Item]):
        if valor_atual > melhor_valor[0]:
            melhor_valor[0] = valor_atual
            melhor_selecao[0] = list(selecionados)

        if idx == n:
            return

        if limite_superior(idx, peso_atual, valor_atual) <= melhor_valor[0]:
            return

        it = itens_ordenados[idx]

        # Ramo 1: incluir o item (se couber)
        if peso_atual + it.peso <= capacidade:
            selecionados.append(it)
            branch_and_bound(idx + 1, peso_atual + it.peso, valor_atual + it.valor, selecionados)
            selecionados.pop()

        # Ramo 2: não incluir o item
        branch_and_bound(idx + 1, peso_atual, valor_atual, selecionados)

    branch_and_bound(0, 0.0, 0.0, [])
    return melhor_valor[0], melhor_selecao[0]


def resolver_knapsack_guloso(itens: List[Item], capacidade: float) -> Tuple[float, List[Item]]:
    """
    Resolve a mochila de forma gulosa (ordenada por valor/peso).
    Retorna (valor_total, lista_itens_selecionados).
    """
    itens_ordenados = sorted(
        itens,
        key=lambda it: it.valor / it.peso if it.peso > 0 else float("inf"),
        reverse=True,
    )
    peso_atual = 0.0
    valor_total = 0.0
    selecionados: List[Item] = []

    for it in itens_ordenados:
        if peso_atual + it.peso <= capacidade:
            peso_atual += it.peso
            valor_total += it.valor
            selecionados.append(it)

    return valor_total, selecionados


@SolverRegistry.register
class QueimadaSolver(BaseSolver):
    """
    Heurística da Queimada (Heat / Thermal Heuristic) para o TTP.

    Etapas:
    1. Temperatura dos Vértices: calculada a partir dos itens de cada cidade.
    2. Dissipação de Calor: propaga a temperatura dos nós pelas arestas do grafo.
    3. Construção do Caminho: ordena a rota a partir do calor acumulado nas arestas
       (direcionando as cidades de maior calor para o final ou início da viagem).
    4. Seleção de Itens: avalia o valor relativo do item contra o custo marginal de
       transporte até a cidade 1 e seleciona a melhor carga.
    """

    def __init__(
        self,
        temperatura: str = "knapsack_greedy",
        dissipacao: str = "dijkstra",
        direcao: str = "calor_final",
        selecao: str = "greedy_relative",
        metodo_construcao: str = "walk",
        nome: Optional[str] = None,
    ):
        """
        Parâmetros:
        - temperatura: 'total_value', 'knapsack_exact', 'knapsack_greedy'
        - dissipacao: 'adjacent_only', 'dijkstra', 'bfs'
        - direcao: 'calor_final' (maior calor no final da rota), 'calor_inicio' (maior calor no início)
        - selecao: 'greedy_relative', 'knapsack_relative'
        - metodo_construcao: 'walk' (caminhada sequencial gulosa por calor), 'kruskal' (ciclo Kruskal orientado)
        """
        self.temperatura_modo = temperatura
        self.dissipacao_modo = dissipacao
        self.direcao_modo = direcao
        self.selecao_modo = selecao
        self.metodo_construcao = metodo_construcao
        self._nome = nome or "queimada"

    @property
    def nome(self) -> str:
        return self._nome

    def _calcular_temperaturas(self, instance: TTPInstance) -> Dict[int, float]:
        """Calcula a temperatura inicial de cada vértice."""
        temperaturas: Dict[int, float] = {}

        for cid, cidade in instance.grafo.cidades.items():
            if cid == 1 or not cidade.itens:
                # Cidade 1 é o depósito inicial / final (geralmente sem itens)
                temperaturas[cid] = 0.0
                continue

            if self.temperatura_modo == "total_value":
                temperaturas[cid] = sum(it.valor for it in cidade.itens)
            elif self.temperatura_modo == "knapsack_exact":
                val, _ = resolver_knapsack_01_exato(cidade.itens, instance.capacidade_mochila)
                temperaturas[cid] = val
            elif self.temperatura_modo == "knapsack_greedy":
                val, _ = resolver_knapsack_guloso(cidade.itens, instance.capacidade_mochila)
                temperaturas[cid] = val
            else:
                raise ValueError(f"Modo de temperatura inválido: {self.temperatura_modo}")

        return temperaturas

    def _dissipar_calor(
        self, instance: TTPInstance, temperaturas: Dict[int, float]
    ) -> Dict[Tuple[int, int], float]:
        """
        Dissipa o calor dos vértices pelas arestas do grafo.
        Retorna dicionário (min(u, v), max(u, v)) -> calor acumulado na aresta.
        """
        cidades = list(instance.grafo.cidades.keys())
        n = len(cidades)

        arestas: Set[Tuple[int, int]] = set()
        for i in range(n):
            for j in range(i + 1, n):
                u, v = cidades[i], cidades[j]
                arestas.add((min(u, v), max(u, v)))

        calor: Dict[Tuple[int, int], float] = {e: 0.0 for e in arestas}

        if self.dissipacao_modo == "adjacent_only":
            for s in cidades:
                t_s = temperaturas.get(s, 0.0)
                if t_s <= 0.0:
                    continue
                for u in cidades:
                    if u == s:
                        continue
                    d = instance.grafo.distancia(s, u)
                    e = (min(s, u), max(s, u))
                    calor[e] += t_s / max(d, 0.001)

        elif self.dissipacao_modo == "dijkstra":
            for s in cidades:
                t_s = temperaturas.get(s, 0.0)
                if t_s <= 0.0:
                    continue

                dist = {c: float("inf") for c in cidades}
                dist[s] = 0.0
                pq = [(0.0, s)]
                while pq:
                    d_u, u = heapq.heappop(pq)
                    if d_u > dist[u]:
                        continue
                    for v in cidades:
                        if v == u:
                            continue
                        w = instance.grafo.distancia(u, v)
                        if dist[u] + w < dist[v]:
                            dist[v] = dist[u] + w
                            heapq.heappush(pq, (dist[v], v))

                for u, v in arestas:
                    # Distância de s até a aresta (u, v) considerando o ponto médio
                    d_edge = min(dist[u], dist[v]) + instance.grafo.distancia(u, v) / 2.0
                    calor[(u, v)] += t_s / (1.0 + d_edge)

        elif self.dissipacao_modo == "bfs":
            for s in cidades:
                t_s = temperaturas.get(s, 0.0)
                if t_s <= 0.0:
                    continue

                level = {c: -1 for c in cidades}
                level[s] = 0
                queue = deque([s])
                while queue:
                    u = queue.popleft()
                    for v in cidades:
                        if v == u:
                            continue
                        if level[v] == -1:
                            level[v] = level[u] + 1
                            queue.append(v)

                for u, v in arestas:
                    h = min(level[u], level[v]) + 1
                    d = instance.grafo.distancia(u, v)
                    calor[(u, v)] += t_s / (h * (1.0 + d))

        else:
            raise ValueError(f"Modo de dissipação inválido: {self.dissipacao_modo}")

        return calor

    def _construir_rota(
        self,
        instance: TTPInstance,
        calor: Dict[Tuple[int, int], float],
        temperaturas: Dict[int, float],
    ) -> List[int]:
        """Constrói a rota iniciando na cidade 1 a partir do calor das arestas."""
        cidades = list(instance.grafo.cidades.keys())
        n = len(cidades)
        if n <= 2:
            return cidades

        if self.metodo_construcao == "walk":
            # Caminhada sequencial partindo da cidade 1
            nao_visitadas = set(cidades)
            nao_visitadas.remove(1)
            rota = [1]
            atual = 1

            while nao_visitadas:
                if self.direcao_modo == "calor_inicio":
                    # No início busca arestas de maior calor
                    prox = max(
                        nao_visitadas,
                        key=lambda c: calor.get((min(atual, c), max(atual, c)), 0.0),
                    )
                elif self.direcao_modo == "calor_final":
                    # Prioriza deixar cidades com maior calor para o final
                    prox = min(
                        nao_visitadas,
                        key=lambda c: calor.get((min(atual, c), max(atual, c)), 0.0),
                    )
                else:
                    raise ValueError(f"Direção inválida: {self.direcao_modo}")

                rota.append(prox)
                nao_visitadas.remove(prox)
                atual = prox

            return rota

        elif self.metodo_construcao == "kruskal":
            # Kruskal para formar ciclo TSP usando arestas de maior calor
            arestas_ordenadas = sorted(calor.keys(), key=lambda e: calor[e], reverse=True)
            adj: Dict[int, List[int]] = {c: [] for c in cidades}
            parent = {c: c for c in cidades}

            def find(i: int) -> int:
                if parent[i] == i:
                    return i
                parent[i] = find(parent[i])
                return parent[i]

            arestas_adicionadas = 0
            for u, v in arestas_ordenadas:
                if len(adj[u]) < 2 and len(adj[v]) < 2:
                    root_u = find(u)
                    root_v = find(v)
                    if root_u != root_v or arestas_adicionadas == n - 1:
                        adj[u].append(v)
                        adj[v].append(u)
                        parent[root_u] = root_v
                        arestas_adicionadas += 1
                        if arestas_adicionadas == n:
                            break

            # Reconstrução do ciclo iniciando em 1
            rota = [1]
            atual = 1
            ant = None
            while len(rota) < n:
                vizinhos = adj[atual]
                prox = vizinhos[0] if vizinhos[0] != ant else vizinhos[1]
                rota.append(prox)
                ant = atual
                atual = prox

            # Duas orientações do ciclo (horário e anti-horário)
            rota_rev = [1] + list(reversed(rota[1:]))

            # Avalia a concentração de calor ao longo da rota
            score_rota = sum(i * temperaturas.get(rota[i], 0.0) for i in range(n))
            score_rev = sum(i * temperaturas.get(rota_rev[i], 0.0) for i in range(n))

            if self.direcao_modo == "calor_final":
                return rota if score_rota >= score_rev else rota_rev
            else:
                return rota if score_rota <= score_rev else rota_rev

        else:
            raise ValueError(f"Método de construção inválido: {self.metodo_construcao}")

    def _selecionar_itens(
        self, instance: TTPInstance, rota: List[int]
    ) -> Dict[int, List[int]]:
        """Seleciona os itens com base no valor relativo frente ao caminho percorrido."""
        n = len(rota)
        dist_restante: Dict[int, float] = {}
        dist_acc = instance.grafo.distancia(rota[-1], rota[0])
        dist_restante[rota[-1]] = dist_acc

        for i in range(n - 2, -1, -1):
            dist_acc += instance.grafo.distancia(rota[i], rota[i + 1])
            dist_restante[rota[i]] = dist_acc

        v_avg = (instance.v_max + instance.v_min) / 2.0
        cap_mochila = max(instance.capacidade_mochila, 1e-6)

        # Cálculo do lucro líquido estimado para cada item
        itens_com_lucro: List[Tuple[float, Item]] = []
        for item in instance.itens:
            if item.peso <= 0:
                continue
            d_rest = dist_restante.get(item.cidade_origem, 0.0)
            custo_transporte = (
                instance.taxa_aluguel
                * d_rest
                * ((instance.v_max - instance.v_min) / (cap_mochila * (v_avg ** 2)))
                * item.peso
            )
            lucro_liquido = item.valor - custo_transporte
            if lucro_liquido > 0:
                itens_com_lucro.append((lucro_liquido, item))

        plano_coleta: Dict[int, List[int]] = {}

        if self.selecao_modo == "greedy_relative":
            # Ordena por densidade de lucro líquido (lucro_liquido / peso)
            itens_com_lucro.sort(key=lambda x: x[0] / x[1].peso, reverse=True)
            peso_atual = 0.0
            for _, item in itens_com_lucro:
                if peso_atual + item.peso <= instance.capacidade_mochila:
                    peso_atual += item.peso
                    if item.cidade_origem not in plano_coleta:
                        plano_coleta[item.cidade_origem] = []
                    plano_coleta[item.cidade_origem].append(item.id)

        elif self.selecao_modo == "knapsack_relative":
            # Constrói itens com valor ajustado para o KP exato
            itens_adaptados = [
                Item(id=it.id, peso=it.peso, valor=lucro, cidade_origem=it.cidade_origem)
                for lucro, it in itens_com_lucro
            ]
            _, selecionados = resolver_knapsack_01_exato(
                itens_adaptados, instance.capacidade_mochila
            )
            for it in selecionados:
                if it.cidade_origem not in plano_coleta:
                    plano_coleta[it.cidade_origem] = []
                plano_coleta[it.cidade_origem].append(it.id)

        else:
            raise ValueError(f"Modo de seleção de itens inválido: {self.selecao_modo}")

        return plano_coleta

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        # 1. Calcular temperatura de cada vértice
        temperaturas = self._calcular_temperaturas(instance)

        # 2. Dissipar calor pelas arestas
        calor = self._dissipar_calor(instance, temperaturas)

        # 3. Construir a rota a partir do calor acumulado
        rota = self._construir_rota(instance, calor, temperaturas)

        # 4. Selecionar itens de acordo com o valor relativo ao caminho
        plano_coleta = self._selecionar_itens(instance, rota)

        return SolucaoTTP(rota=rota, plano_coleta=plano_coleta)


# Registro de variações convenientes para execução via CLI
SolverRegistry.register(
    type(
        "QueimadaAdjSolver",
        (QueimadaSolver,),
        {"nome": "queimada_adj", "__init__": lambda self: QueimadaSolver.__init__(self, dissipacao="adjacent_only", nome="queimada_adj")},
    )
)
SolverRegistry.register(
    type(
        "QueimadaBfsSolver",
        (QueimadaSolver,),
        {"nome": "queimada_bfs", "__init__": lambda self: QueimadaSolver.__init__(self, dissipacao="bfs", nome="queimada_bfs")},
    )
)
SolverRegistry.register(
    type(
        "QueimadaInicioSolver",
        (QueimadaSolver,),
        {"nome": "queimada_inicio", "__init__": lambda self: QueimadaSolver.__init__(self, direcao="calor_inicio", nome="queimada_inicio")},
    )
)
SolverRegistry.register(
    type(
        "QueimadaKpSolver",
        (QueimadaSolver,),
        {"nome": "queimada_kp", "__init__": lambda self: QueimadaSolver.__init__(self, selecao="knapsack_relative", nome="queimada_kp")},
    )
)
SolverRegistry.register(
    type(
        "QueimadaExactSolver",
        (QueimadaSolver,),
        {"nome": "queimada_exact", "__init__": lambda self: QueimadaSolver.__init__(self, temperatura="knapsack_exact", nome="queimada_exact")},
    )
)
SolverRegistry.register(
    type(
        "QueimadaKruskalSolver",
        (QueimadaSolver,),
        {"nome": "queimada_kruskal", "__init__": lambda self: QueimadaSolver.__init__(self, metodo_construcao="kruskal", nome="queimada_kruskal")},
    )
)
