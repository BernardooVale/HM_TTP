import heapq
import math
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from src.core.models import Item, SolucaoTTP, TTPInstance
from src.solvers.base import BaseSolver, SolverRegistry
from .config_manager import obter_melhor_configuracao
from src.solvers.tsp import (
    otimizar_2opt,
    resolver_caminho_fixo_bb,
    resolver_caminho_fixo_2opt,
)

def resolver_knapsack_01_exato(itens: List[Item], capacidade: float) -> Tuple[float, List[Item]]:
    """
    Resolve o Problema da Mochila 0/1 de forma exata usando Branch and Bound com poda.
    Retorna (valor_otimo, lista_itens_selecionados).
    """
    if not itens or capacidade <= 0:
        return 0.0, []

    peso_total = sum(item.peso for item in itens)
    if peso_total <= capacidade:
        return sum(item.valor for item in itens), list(itens)

    # Ordena por densidade decrescente para limites superiores eficientes
    itens_ordenados = sorted(
        itens,
        key=lambda item: item.valor / item.peso if item.peso > 0 else float("inf"),
        reverse=True,
    )
    n = len(itens_ordenados)

    melhor_valor = [0.0]
    melhor_selecao = [[]]

    def limite_superior(idx: int, peso_atual: float, valor_atual: float) -> float:
        limite = valor_atual
        peso = peso_atual
        for i in range(idx, n):
            item = itens_ordenados[i]
            if peso + item.peso <= capacidade:
                peso += item.peso
                limite += item.valor
            else:
                if item.peso > 0:
                    limite += (capacidade - peso) * (item.valor / item.peso)
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

        item = itens_ordenados[idx]

        # Ramo 1: incluir o item (se couber)
        if peso_atual + item.peso <= capacidade:
            selecionados.append(item)
            branch_and_bound(idx + 1, peso_atual + item.peso, valor_atual + item.valor, selecionados)
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
        key=lambda item: item.valor / item.peso if item.peso > 0 else float("inf"),
        reverse=True,
    )
    peso_atual = 0.0
    valor_total = 0.0
    selecionados: List[Item] = []

    for item in itens_ordenados:
        if peso_atual + item.peso <= capacidade:
            peso_atual += item.peso
            valor_total += item.valor
            selecionados.append(item)

    return valor_total, selecionados


def otimizar_trechos_vazios(
    instance: TTPInstance, rota: List[int], plano_coleta: Dict[int, List[int]], max_bb_size: int = 8
) -> List[int]:
    """
    Identifica segmentos contíguos de cidades sem itens coletados ("buracos")
    e otimiza a ordem de visitação mantendo as cidades de coleta e a Cidade 1 ancoradas.

    Como o peso carregado pelo ladrão é constante em cada um desses trechos,
    a velocidade é constante e minimizar o tempo é rigorosamente equivalente a
    minimizar a distância percorrida no trecho.
    """
    cidades_com_coleta = {cid for cid, itens in plano_coleta.items() if len(itens) > 0}
    n = len(rota)
    if n <= 3:
        return list(rota)

    ancoras = [0]
    for i in range(1, n):
        if rota[i] in cidades_com_coleta:
            ancoras.append(i)

    nova_rota = list(rota)

    for k in range(len(ancoras)):
        idx_A = ancoras[k]
        if k + 1 < len(ancoras):
            idx_B = ancoras[k + 1]
            cidades_trecho = nova_rota[idx_A + 1 : idx_B]
            if len(cidades_trecho) >= 2:
                A = nova_rota[idx_A]
                B = nova_rota[idx_B]
                if len(cidades_trecho) <= max_bb_size:
                    ordem = resolver_caminho_fixo_bb(A, B, cidades_trecho, instance.grafo)
                else:
                    ordem = resolver_caminho_fixo_2opt(A, B, cidades_trecho, instance.grafo)
                nova_rota[idx_A + 1 : idx_B] = ordem
        else:
            # Do último ponto de coleta até o retorno à cidade 1
            idx_B = n
            cidades_trecho = nova_rota[idx_A + 1 : idx_B]
            if len(cidades_trecho) >= 2:
                A = nova_rota[idx_A]
                B = nova_rota[0]  # Retorno ao depósito (cidade 1)
                if len(cidades_trecho) <= max_bb_size:
                    ordem = resolver_caminho_fixo_bb(A, B, cidades_trecho, instance.grafo)
                else:
                    ordem = resolver_caminho_fixo_2opt(A, B, cidades_trecho, instance.grafo)
                nova_rota[idx_A + 1 : idx_B] = ordem

    return nova_rota


@SolverRegistry.register
class QueimadaSolver(BaseSolver):
    """
    Heurística da Queimada (Wildfire Heuristic) para o TTP.

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
        temperatura: Optional[str] = None,
        dissipacao: Optional[str] = None,
        direcao: Optional[str] = None,
        selecao: Optional[str] = None,
        metodo_construcao: Optional[str] = None,
        ordem_selecao: Optional[str] = None,
        refinamento: Optional[bool] = None,
        refinamento_rota: Optional[bool] = None,
        afinidade_termica: Optional[bool] = None,
        gamma_afinidade: Optional[float] = None,
        adaptar_regime_r: Optional[bool] = None,
        modelo: str = "TTP1",
        otimizar_buracos: Optional[bool] = None,
        grafo_direcionado: Optional[bool] = None,
        nome: Optional[str] = None,
    ):
        # Carrega dinamicamente a melhor configuração conhecida persistida
        cfg = obter_melhor_configuracao("queimada")
        """
        Parâmetros:
        - temperatura:              'total_value', 'knapsack_exact', 'knapsack_greedy' (padrão ótimo: 'total_value')
        - dissipacao:               'adjacent_only', 'dijkstra', 'bfs' (padrão ótimo: 'bfs')
        - direcao (ordem de calor): 'calor_final', 'calor_inicio' (padrão ótimo: 'calor_final')
        - selecao:                  'greedy_relative', 'knapsack_relative' (padrão ótimo: 'greedy_relative')
        - metodo_construcao:        'walk', 'kruskal', 'backward_edges', 'backward_directed', 'forward_directed' (padrão ótimo: 'kruskal')
        - ordem_selecao:            'densidade', 'reversa', 'sequencial' (padrão ótimo: 'densidade')
        - refinamento:              se True, executa busca local Drop & Add no empacotamento (padrão: True)
        - refinamento_rota:         se True, executa busca local 2-opt para desatar cruzamentos da rota (padrão: True)
        - afinidade_termica:        se True, Kruskal prioriza arestas por calor/distância^gamma (padrão: True)
        - gamma_afinidade:          expoente de penalização de distância na afinidade térmica (padrão: 1.0)
        - adaptar_regime_r:         se True, ajusta dinamicamente a afinidade ao valor de R da instância (padrão: False)
        - modelo:                   'TTP1' (aluguel de viagem) ou 'TTP2' (degradação temporal) (padrão: 'TTP1')
        - otimizar_buracos:         se True, otimiza segmentos vazios sem coleta pós-mochila via B&B / 2-opt (padrão: False)
        - grafo_direcionado:        se True, modela calor direcionado em arcos (u, v) (padrão: True se backward_directed/forward_directed)
        """
        self.temperatura_modo = temperatura if temperatura is not None else cfg.get("temperatura", "total_value")
        self.dissipacao_modo = dissipacao if dissipacao is not None else cfg.get("dissipacao", "dijkstra")
        self.direcao_modo = direcao if direcao is not None else cfg.get("direcao", "calor_final")
        self.selecao_modo = selecao if selecao is not None else cfg.get("selecao", "greedy_relative")
        self.metodo_construcao = metodo_construcao if metodo_construcao is not None else cfg.get("metodo_construcao", "backward_edges")
        self.ordem_selecao = ordem_selecao if ordem_selecao is not None else cfg.get("ordem_selecao", "densidade")
        self.refinamento = refinamento if refinamento is not None else cfg.get("refinamento", True)
        self.refinamento_rota = refinamento_rota if refinamento_rota is not None else cfg.get("refinamento_rota", True)
        self.afinidade_termica = afinidade_termica if afinidade_termica is not None else cfg.get("afinidade_termica", True)
        self.gamma_afinidade = gamma_afinidade if gamma_afinidade is not None else cfg.get("gamma_afinidade", 1.0)
        self.adaptar_regime_r = adaptar_regime_r if adaptar_regime_r is not None else cfg.get("adaptar_regime_r", False)
        self.otimizar_buracos = otimizar_buracos if otimizar_buracos is not None else cfg.get("otimizar_buracos", True)
        if grafo_direcionado is not None:
            self.grafo_direcionado = grafo_direcionado
        else:
            self.grafo_direcionado = cfg.get(
                "grafo_direcionado",
                self.metodo_construcao in ("backward_directed", "forward_directed"),
            )
        self.modelo = modelo
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
                temperaturas[cid] = sum(item.valor for item in cidade.itens)
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
        Dissipa o calor dos vértices pelas arestas/arcos do grafo em alta performance.
        - Não-direcionado: retorna (min(u, v), max(u, v)) -> calor acumulado na aresta.
        - Direcionado: retorna (u, v) -> calor transmitido no arco direcionado u -> v.
        """
        cidades = list(instance.grafo.cidades.keys())
        n = len(cidades)

        calor: Dict[Tuple[int, int], float] = {}

        if not self.grafo_direcionado:
            # Modo não-direcionado clássico (min(u, v), max(u, v))
            if self.dissipacao_modo == "adjacent_only":
                for i in range(n):
                    cidade_atual = cidades[i]
                    temperatura_cidade = temperaturas.get(cidade_atual, 0.0)

                    for j in range(i + 1, n):
                        prox_cidade = cidades[j]
                        temperatura_prox_cidade = temperaturas.get(prox_cidade, 0.0)
                        aresta = (cidade_atual, prox_cidade) if cidade_atual < prox_cidade else (prox_cidade, cidade_atual)

                        if temperatura_cidade > 0.0 or temperatura_prox_cidade > 0.0:
                            distancia = instance.grafo.distancia(cidade_atual, prox_cidade)
                            calor[aresta] = (temperatura_cidade + temperatura_prox_cidade) / max(distancia, 0.001)
                        else:
                            calor[aresta] = 0.0

            elif self.dissipacao_modo == "bfs":
                T_total = sum(temperaturas.values())

                for i in range(n):
                    cidade = cidades[i]
                    temperatura_cidade = temperaturas.get(cidade, 0.0)

                    for j in range(i + 1, n):
                        prox_cidade = cidades[j]
                        temperatura_prox_cidade = temperaturas.get(prox_cidade, 0.0)
                        distancia = instance.grafo.distancia(cidade, prox_cidade)
                        aresta = (cidade, prox_cidade) if cidade < prox_cidade else (prox_cidade, cidade)
                        calor[aresta] = (temperatura_cidade + temperatura_prox_cidade + T_total) / (2.0 * (1.0 + distancia))

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

                    for i in range(n):
                        u = cidades[i]
                        for j in range(i + 1, n):
                            v = cidades[j]
                            e = (u, v) if u < v else (v, u)
                            d_edge = min(dist[u], dist[v]) + instance.grafo.distancia(u, v) / 2.0
                            calor[e] = calor.get(e, 0.0) + t_s / (1.0 + d_edge)
            else:
                raise ValueError(f"Modo de dissipação inválido: {self.dissipacao_modo}")

        else:
            # Modo direcionado: arcos ordenados (origem, destino) com u != v
            if self.dissipacao_modo == "adjacent_only":
                for u in cidades:
                    t_u = temperaturas.get(u, 0.0)
                    for v in cidades:
                        if u == v:
                            continue
                        if t_u > 0.0:
                            distancia = instance.grafo.distancia(u, v)
                            calor[(u, v)] = t_u / max(distancia, 0.001)
                        else:
                            calor[(u, v)] = 0.0

            elif self.dissipacao_modo == "bfs":
                T_total = sum(temperaturas.values())
                n_outros = max(n - 2, 1)
                for u in cidades:
                    t_u = temperaturas.get(u, 0.0)
                    for v in cidades:
                        if u == v:
                            continue
                        t_v = temperaturas.get(v, 0.0)
                        # Calor de u somado ao calor das outras fontes s, excluindo expressamente o destino v
                        outras_fontes = max(T_total - t_u - t_v, 0.0)
                        distancia = instance.grafo.distancia(u, v)
                        calor[(u, v)] = (t_u + outras_fontes / n_outros) / (2.0 * (1.0 + distancia))

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

                    for u in cidades:
                        for v in cidades:
                            if u == v or s == v:
                                # Exclui o próprio destino v como fonte de calor para o arco u -> v
                                continue
                            d_edge = dist[u] + instance.grafo.distancia(u, v) / 2.0
                            calor[(u, v)] = calor.get((u, v), 0.0) + t_s / (1.0 + d_edge)
            else:
                raise ValueError(f"Modo de dissipação inválido: {self.dissipacao_modo}")

        return calor

    def _construir_rota(
        self,
        instance: TTPInstance,
        calor: Dict[Tuple[int, int], float],
        temperaturas: Dict[int, float],
    ) -> List[int]:
        """Constrói a rota iniciando na cidade 1 a partir do calor das arestas/arcos."""
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
            # Ordenação de arestas: puramente por calor ou por afinidade térmica
            if self.afinidade_termica:
                gamma = self.gamma_afinidade
                if self.adaptar_regime_r and self.modelo == "TTP1":
                    # Modula gamma em função de R da instância:
                    gamma = min(2.5, max(0.5, 0.5 + instance.taxa_aluguel / 2.0))

                def score_aresta(e: Tuple[int, int]) -> float:
                    d = max(instance.grafo.distancia(e[0], e[1]), 0.001)
                    return calor[e] / (d ** gamma)

                arestas_ordenadas = sorted(calor.keys(), key=score_aresta, reverse=True)
            else:
                # Kruskal clássico sem afinidade térmica (puramente pelo calor)
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

        elif self.metodo_construcao == "backward_edges":
            # Construção regressiva de trás para frente a partir da Cidade 1
            # Prioriza encadear as arestas de maior calor relativo/afinidade na cauda do percurso.
            gamma = self.gamma_afinidade
            if self.adaptar_regime_r and self.modelo == "TTP1":
                gamma = min(2.5, max(0.5, 0.5 + instance.taxa_aluguel / 2.0))

            def score_aresta_backward(u_node: int, v_node: int) -> float:
                e_key = (min(u_node, v_node), max(u_node, v_node))
                c_val = calor.get(e_key, 0.0)
                if self.afinidade_termica:
                    d_val = max(instance.grafo.distancia(u_node, v_node), 0.001)
                    return c_val / (d_val ** gamma)
                return c_val

            nao_visitadas = set(cidades)
            nao_visitadas.remove(1)

            cadeia_reversa = [1]
            atual = 1

            while nao_visitadas:
                prox = max(nao_visitadas, key=lambda v_cand: score_aresta_backward(atual, v_cand))
                cadeia_reversa.append(prox)
                nao_visitadas.remove(prox)
                atual = prox

            # Rota direta: as arestas adicionadas primeiro na cadeia reversa ficam no final da rota!
            rota = [1] + list(reversed(cadeia_reversa[1:]))

        elif self.metodo_construcao == "backward_directed":
            # Construção regressiva com arcos direcionados de entrada: cand -> atual
            gamma = self.gamma_afinidade
            if self.adaptar_regime_r and self.modelo == "TTP1":
                gamma = min(2.5, max(0.5, 0.5 + instance.taxa_aluguel / 2.0))

            def score_arco_backward(cand: int, target: int) -> float:
                c_val = calor.get((cand, target), 0.0)
                d_val = max(instance.grafo.distancia(cand, target), 0.001)
                if self.afinidade_termica:
                    return (c_val + 1e-6) / (d_val ** gamma)
                return c_val + (1e-6 / d_val)

            nao_visitadas = set(cidades)
            nao_visitadas.remove(1)

            cadeia_reversa = [1]
            atual = 1

            while nao_visitadas:
                prox = max(nao_visitadas, key=lambda cand: score_arco_backward(cand, atual))
                cadeia_reversa.append(prox)
                nao_visitadas.remove(prox)
                atual = prox

            # Inverte para que os arcos de maior calor de entrada fiquem no final da viagem:
            rota = [1] + list(reversed(cadeia_reversa[1:]))

        elif self.metodo_construcao == "forward_directed":
            # Construção progressiva a partir da Cidade 1 com arcos direcionados de saída: atual -> cand
            # Prioriza arcos mais frios no início da viagem
            gamma = self.gamma_afinidade
            if self.adaptar_regime_r and self.modelo == "TTP1":
                gamma = min(2.5, max(0.5, 0.5 + instance.taxa_aluguel / 2.0))

            def score_arco_forward(orig: int, cand: int) -> float:
                c_val = calor.get((orig, cand), 0.0)
                d_val = max(instance.grafo.distancia(orig, cand), 0.001)
                if self.afinidade_termica:
                    # Menor calor tem prioridade; empate em frias desempata por menor distância
                    return (c_val * (d_val ** gamma)) + (1e-6 * d_val)
                return c_val + (1e-6 * d_val)

            nao_visitadas = set(cidades)
            nao_visitadas.remove(1)

            rota = [1]
            atual = 1

            while nao_visitadas:
                prox = min(nao_visitadas, key=lambda cand: score_arco_forward(atual, cand))
                rota.append(prox)
                nao_visitadas.remove(prox)
                atual = prox

        else:
            raise ValueError(f"Método de construção inválido: {self.metodo_construcao}")

        # Refinamento de rota via 2-opt (se ativado)
        if self.refinamento_rota and n > 3:
            rota = otimizar_2opt(rota, instance.grafo)

        # Duas orientações do ciclo (horário e anti-horário)
        rota_rev = [1] + list(reversed(rota[1:]))

        if self.metodo_construcao in ("backward_edges", "backward_directed", "forward_directed"):
            # Para métodos de arestas/arcos, avalia a concentração de calor ao longo da viagem
            if self.grafo_direcionado:
                score_rota = sum(
                    calor.get((rota[i], rota[(i + 1) % n]), 0.0) * i
                    for i in range(n)
                )
                score_rev = sum(
                    calor.get((rota_rev[i], rota_rev[(i + 1) % n]), 0.0) * i
                    for i in range(n)
                )
            else:
                score_rota = sum(
                    calor.get((min(rota[i], rota[(i + 1) % n]), max(rota[i], rota[(i + 1) % n])), 0.0) * i
                    for i in range(n)
                )
                score_rev = sum(
                    calor.get((min(rota_rev[i], rota_rev[(i + 1) % n]), max(rota_rev[i], rota_rev[(i + 1) % n])), 0.0) * i
                    for i in range(n)
                )
        else:
            # Avalia a concentração de calor nos vértices ao longo da rota
            score_rota = sum(i * temperaturas.get(rota[i], 0.0) for i in range(n))
            score_rev = sum(i * temperaturas.get(rota_rev[i], 0.0) for i in range(n))

        if self.direcao_modo == "calor_final":
            return rota if score_rota >= score_rev else rota_rev
        else:
            return rota if score_rota <= score_rev else rota_rev

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
        const_deg = max(instance.constante_degradacao, 1e-6)

        # Cálculo do lucro líquido estimado para cada item
        itens_com_lucro: List[Tuple[float, Item]] = []
        for item in instance.itens:
            if item.peso <= 0:
                continue
            d_rest = dist_restante.get(item.cidade_origem, 0.0)

            if self.modelo == "TTP2":
                # TTP2: degradação temporal do valor
                v_depois = max(
                    instance.v_min,
                    instance.v_max
                    - item.peso * ((instance.v_max - instance.v_min) / cap_mochila),
                )
                tempo_restante = d_rest / v_depois
                degradacao = math.ceil(tempo_restante / const_deg)
                lucro_liquido = item.valor - degradacao
            else:
                # TTP1: custo de aluguel por atraso
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
            if self.ordem_selecao == "densidade":
                itens_com_lucro.sort(key=lambda x: x[0] / x[1].peso, reverse=True)
                peso_atual = 0.0
                for _, item in itens_com_lucro:
                    if peso_atual + item.peso <= instance.capacidade_mochila:
                        peso_atual += item.peso
                        if item.cidade_origem not in plano_coleta:
                            plano_coleta[item.cidade_origem] = []
                        plano_coleta[item.cidade_origem].append(item.id)
            elif self.ordem_selecao == "reversa":
                cidades_reversas = [c for c in reversed(rota) if c != 1]
                peso_atual = 0.0
                for cid in cidades_reversas:
                    itens_cid = [x for x in itens_com_lucro if x[1].cidade_origem == cid]
                    itens_cid.sort(key=lambda x: x[0] / x[1].peso, reverse=True)
                    for _, item in itens_cid:
                        if peso_atual + item.peso <= instance.capacidade_mochila:
                            peso_atual += item.peso
                            if cid not in plano_coleta:
                                plano_coleta[cid] = []
                            plano_coleta[cid].append(item.id)
            elif self.ordem_selecao == "sequencial":
                cidades_seq = [c for c in rota if c != 1]
                peso_atual = 0.0
                for cid in cidades_seq:
                    itens_cid = [x for x in itens_com_lucro if x[1].cidade_origem == cid]
                    itens_cid.sort(key=lambda x: x[0] / x[1].peso, reverse=True)
                    for _, item in itens_cid:
                        if peso_atual + item.peso <= instance.capacidade_mochila:
                            peso_atual += item.peso
                            if cid not in plano_coleta:
                                plano_coleta[cid] = []
                            plano_coleta[cid].append(item.id)
            else:
                raise ValueError(f"Ordem de seleção de cidades inválida: {self.ordem_selecao}")

        elif self.selecao_modo == "knapsack_relative":
            itens_adaptados = [
                Item(id=item.id, peso=item.peso, valor=lucro, cidade_origem=item.cidade_origem)
                for lucro, item in itens_com_lucro
            ]
            _, selecionados = resolver_knapsack_01_exato(
                itens_adaptados, instance.capacidade_mochila
            )
            for item in selecionados:
                if item.cidade_origem not in plano_coleta:
                    plano_coleta[item.cidade_origem] = []
                plano_coleta[item.cidade_origem].append(item.id)

        else:
            raise ValueError(f"Modo de seleção de itens inválido: {self.selecao_modo}")

        if self.refinamento:
            from src.solvers.pack_iterative import refinar_plano_coleta
            plano_coleta = refinar_plano_coleta(
                instance, rota, plano_coleta, modelo=self.modelo
            )

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

        # 5. Otimização de trechos vazios ("buracos") pós-mochila
        if self.otimizar_buracos:
            rota = otimizar_trechos_vazios(instance, rota, plano_coleta)

        return SolucaoTTP(rota=rota, plano_coleta=plano_coleta)


# Registro de variações convenientes para execução via CLI e comparações
SolverRegistry.register(
    type(
        "QueimadaCom2OptSolver",
        (QueimadaSolver,),
        {"nome": "queimada_com_2opt", "__init__": lambda self: QueimadaSolver.__init__(self, refinamento_rota=True, nome="queimada_com_2opt")},
    )
)
SolverRegistry.register(
    type(
        "QueimadaSem2OptSolver",
        (QueimadaSolver,),
        {"nome": "queimada_sem_2opt", "__init__": lambda self: QueimadaSolver.__init__(self, refinamento_rota=False, nome="queimada_sem_2opt")},
    )
)
SolverRegistry.register(
    type(
        "QueimadaComAfinidadeSolver",
        (QueimadaSolver,),
        {"nome": "queimada_com_afinidade", "__init__": lambda self: QueimadaSolver.__init__(self, afinidade_termica=True, nome="queimada_com_afinidade")},
    )
)
SolverRegistry.register(
    type(
        "QueimadaSemAfinidadeSolver",
        (QueimadaSolver,),
        {"nome": "queimada_sem_afinidade", "__init__": lambda self: QueimadaSolver.__init__(self, afinidade_termica=False, nome="queimada_sem_afinidade")},
    )
)
SolverRegistry.register(
    type(
        "QueimadaRegimeRSolver",
        (QueimadaSolver,),
        {"nome": "queimada_regime_r", "__init__": lambda self: QueimadaSolver.__init__(self, adaptar_regime_r=True, afinidade_termica=True, nome="queimada_regime_r")},
    )
)
SolverRegistry.register(
    type(
        "QueimadaTTP2Solver",
        (QueimadaSolver,),
        {"nome": "queimada_ttp2", "__init__": lambda self: QueimadaSolver.__init__(self, modelo="TTP2", nome="queimada_ttp2")},
    )
)
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
        {"nome": "queimada_kruskal", "__init__": lambda self: QueimadaSolver.__init__(self, metodo_construcao="kruskal", afinidade_termica=False, refinamento_rota=False, nome="queimada_kruskal")},
    )
)
SolverRegistry.register(
    type(
        "QueimadaReversaSolver",
        (QueimadaSolver,),
        {"nome": "queimada_reversa", "__init__": lambda self: QueimadaSolver.__init__(self, ordem_selecao="reversa", nome="queimada_reversa")},
    )
)
SolverRegistry.register(
    type(
        "QueimadaRefinadaSolver",
        (QueimadaSolver,),
        {"nome": "queimada_refinada", "__init__": lambda self: QueimadaSolver.__init__(self, refinamento=True, nome="queimada_refinada")},
    )
)
SolverRegistry.register(
    type(
        "QueimadaReversaRefinadaSolver",
        (QueimadaSolver,),
        {"nome": "queimada_reversa_refinada", "__init__": lambda self: QueimadaSolver.__init__(self, ordem_selecao="reversa", refinamento=True, nome="queimada_reversa_refinada")},
    )
)
SolverRegistry.register(
    type(
        "QueimadaLegacySolver",
        (QueimadaSolver,),
        {
            "nome": "queimada_legacy",
            "__init__": lambda self: QueimadaSolver.__init__(
                self,
                temperatura="knapsack_greedy",
                dissipacao="dijkstra",
                direcao="calor_final",
                selecao="greedy_relative",
                metodo_construcao="walk",
                ordem_selecao="densidade",
                refinamento=False,
                refinamento_rota=False,
                afinidade_termica=False,
                nome="queimada_legacy",
            ),
        },
    )
)
SolverRegistry.register(
    type(
        "QueimadaBackwardArestasSolver",
        (QueimadaSolver,),
        {
            "nome": "queimada_backward_arestas",
            "__init__": lambda self: QueimadaSolver.__init__(
                self,
                metodo_construcao="backward_edges",
                nome="queimada_backward_arestas",
            ),
        },
    )
)
SolverRegistry.register(
    type(
        "QueimadaBackwardSem2optSolver",
        (QueimadaSolver,),
        {
            "nome": "queimada_backward_sem_2opt",
            "__init__": lambda self: QueimadaSolver.__init__(
                self,
                metodo_construcao="backward_edges",
                refinamento_rota=False,
                nome="queimada_backward_sem_2opt",
            ),
        },
    )
)
SolverRegistry.register(
    type(
        "QueimadaComBuracosSolver",
        (QueimadaSolver,),
        {
            "nome": "queimada_com_buracos",
            "__init__": lambda self: QueimadaSolver.__init__(
                self,
                otimizar_buracos=True,
                nome="queimada_com_buracos",
            ),
        },
    )
)
SolverRegistry.register(
    type(
        "QueimadaSem2OptBuracosSolver",
        (QueimadaSolver,),
        {
            "nome": "queimada_sem_2opt_buracos",
            "__init__": lambda self: QueimadaSolver.__init__(
                self,
                refinamento_rota=False,
                otimizar_buracos=True,
                nome="queimada_sem_2opt_buracos",
            ),
        },
    )
)
SolverRegistry.register(
    type(
        "QueimadaBackwardBuracosSolver",
        (QueimadaSolver,),
        {
            "nome": "queimada_backward_buracos",
            "__init__": lambda self: QueimadaSolver.__init__(
                self,
                metodo_construcao="backward_edges",
                otimizar_buracos=True,
                nome="queimada_backward_buracos",
            ),
        },
    )
)
SolverRegistry.register(
    type(
        "QueimadaBackwardSem2optBuracosSolver",
        (QueimadaSolver,),
        {
            "nome": "queimada_backward_sem_2opt_buracos",
            "__init__": lambda self: QueimadaSolver.__init__(
                self,
                metodo_construcao="backward_edges",
                refinamento_rota=False,
                otimizar_buracos=True,
                nome="queimada_backward_sem_2opt_buracos",
            ),
        },
    )
)

SolverRegistry.register(
    type(
        "QueimadaKruskalPadraoSolver",
        (QueimadaSolver,),
        {
            "nome": "queimada_kruskal_padrao",
            "__init__": lambda self: QueimadaSolver.__init__(
                self,
                metodo_construcao="kruskal",
                dissipacao="bfs",
                otimizar_buracos=False,
                nome="queimada_kruskal_padrao",
            ),
        },
    )
)

# Variações com Grafo Direcionado (elimina reflexão de auto-calor)
SolverRegistry.register(
    type(
        "QueimadaBackwardDirecionadaSolver",
        (QueimadaSolver,),
        {
            "nome": "queimada_backward_direcionada",
            "__init__": lambda self: QueimadaSolver.__init__(
                self,
                metodo_construcao="backward_directed",
                grafo_direcionado=True,
                refinamento_rota=True,
                otimizar_buracos=False,
                nome="queimada_backward_direcionada",
            ),
        },
    )
)
SolverRegistry.register(
    type(
        "QueimadaBackwardDirecionadaBuracosSolver",
        (QueimadaSolver,),
        {
            "nome": "queimada_backward_direcionada_buracos",
            "__init__": lambda self: QueimadaSolver.__init__(
                self,
                metodo_construcao="backward_directed",
                grafo_direcionado=True,
                refinamento_rota=True,
                otimizar_buracos=True,
                nome="queimada_backward_direcionada_buracos",
            ),
        },
    )
)
SolverRegistry.register(
    type(
        "QueimadaBackwardDirecionadaSem2optBuracosSolver",
        (QueimadaSolver,),
        {
            "nome": "queimada_backward_direcionada_sem_2opt_buracos",
            "__init__": lambda self: QueimadaSolver.__init__(
                self,
                metodo_construcao="backward_directed",
                grafo_direcionado=True,
                refinamento_rota=False,
                otimizar_buracos=True,
                nome="queimada_backward_direcionada_sem_2opt_buracos",
            ),
        },
    )
)
SolverRegistry.register(
    type(
        "QueimadaForwardDirecionadaSolver",
        (QueimadaSolver,),
        {
            "nome": "queimada_forward_direcionada",
            "__init__": lambda self: QueimadaSolver.__init__(
                self,
                metodo_construcao="forward_directed",
                grafo_direcionado=True,
                refinamento_rota=True,
                otimizar_buracos=False,
                nome="queimada_forward_direcionada",
            ),
        },
    )
)
SolverRegistry.register(
    type(
        "QueimadaForwardDirecionadaBuracosSolver",
        (QueimadaSolver,),
        {
            "nome": "queimada_forward_direcionada_buracos",
            "__init__": lambda self: QueimadaSolver.__init__(
                self,
                metodo_construcao="forward_directed",
                grafo_direcionado=True,
                refinamento_rota=True,
                otimizar_buracos=True,
                nome="queimada_forward_direcionada_buracos",
            ),
        },
    )
)
SolverRegistry.register(
    type(
        "QueimadaForwardDirecionadaSem2optBuracosSolver",
        (QueimadaSolver,),
        {
            "nome": "queimada_forward_direcionada_sem_2opt_buracos",
            "__init__": lambda self: QueimadaSolver.__init__(
                self,
                metodo_construcao="forward_directed",
                grafo_direcionado=True,
                refinamento_rota=False,
                otimizar_buracos=True,
                nome="queimada_forward_direcionada_sem_2opt_buracos",
            ),
        },
    )
)
