"""
Avaliador acelerado para soluções do Traveling Thief Problem (TTP1 e TTP2).
Evita alocações redundantes de dicionários e objetos durante buscas locais
e rotinas intensivas de avaliação de empacotamento.
"""

import math
from typing import Dict, List, Optional, Set
from src.core.models import Item, TTPInstance


class FastTTPEvaluator:
    """
    Avaliador otimizado de soluções TTP1 e TTP2 para uma rota fixa.
    Pré-computa distâncias entre pernas e índices de cidades para alcançar
    alta vazão em operações de Drop & Add, 2-opt e refinamento.
    """

    def __init__(self, instance: TTPInstance, rota: List[int], modelo: str = "TTP1"):
        self.instance = instance
        self.rota = list(rota)
        self.n = len(self.rota)
        self.modelo = modelo
        self.cidade_idx: Dict[int, int] = {self.rota[i]: i for i in range(self.n)}
        self.itens_by_id: Dict[int, Item] = {it.id: it for it in instance.itens}

        # Cache direto de propriedades dos itens para evitar overhead de dataclass
        self.item_cidade_idx: Dict[int, int] = {
            it.id: self.cidade_idx[it.cidade_origem]
            for it in instance.itens
            if it.cidade_origem in self.cidade_idx
        }
        self.item_peso: Dict[int, float] = {it.id: it.peso for it in instance.itens}
        self.item_valor: Dict[int, float] = {it.id: it.valor for it in instance.itens}

        # Distâncias fixas das pernas da rota
        self.distancias: List[float] = [
            instance.grafo.distancia(self.rota[i], self.rota[(i + 1) % self.n])
            for i in range(self.n)
        ]

        self.cap_mochila = max(instance.capacidade_mochila, 1e-6)
        self.v_min = instance.v_min
        self.v_max = instance.v_max
        self.taxa_aluguel = instance.taxa_aluguel
        self.constante_degradacao = max(instance.constante_degradacao, 1e-6)
        self.fator_velocidade = (self.v_max - self.v_min) / self.cap_mochila

        # Estado incremental para busca local ultrarrápida
        self.peso_por_cidade: List[float] = [0.0] * self.n
        self.itens_coletados: Set[int] = set()
        self.valor_bruto_atual: float = 0.0
        self.peso_total_atual: float = 0.0

    def inicializar_estado(self, itens_ids: Optional[Set[int]] = None) -> None:
        """Inicializa o estado interno para avaliações incrementais."""
        self.peso_por_cidade = [0.0] * self.n
        self.itens_coletados = set()
        self.valor_bruto_atual = 0.0
        self.peso_total_atual = 0.0
        if itens_ids:
            for it_id in itens_ids:
                self.adicionar_item(it_id)

    def adicionar_item(self, it_id: int) -> None:
        """Adiciona um item ao estado interno em O(1)."""
        idx = self.item_cidade_idx.get(it_id)
        if idx is not None:
            p = self.item_peso[it_id]
            v = self.item_valor[it_id]
            self.peso_por_cidade[idx] += p
            self.valor_bruto_atual += v
            self.peso_total_atual += p
            self.itens_coletados.add(it_id)

    def remover_item(self, it_id: int) -> None:
        """Remove um item do estado interno em O(1)."""
        idx = self.item_cidade_idx.get(it_id)
        if idx is not None and it_id in self.itens_coletados:
            p = self.item_peso[it_id]
            v = self.item_valor[it_id]
            self.peso_por_cidade[idx] -= p
            self.valor_bruto_atual -= v
            self.peso_total_atual -= p
            self.itens_coletados.remove(it_id)

    def calcular_lucro_atual(self, modelo: Optional[str] = None) -> float:
        """Calcula o lucro líquido G para o estado incremental atual em O(N)."""
        mod = modelo or self.modelo
        if mod == "TTP2":
            tempo_coleta_cidade = [0.0] * self.n
            tempo_viagem = 0.0
            peso_acc = 0.0

            for i in range(self.n):
                tempo_coleta_cidade[i] = tempo_viagem
                peso_acc += self.peso_por_cidade[i]
                vel = self.v_max - peso_acc * self.fator_velocidade
                if vel < self.v_min:
                    vel = self.v_min
                tempo_viagem += self.distancias[i] / vel

            tempo_fim = tempo_viagem
            valor_degradado_total = 0.0

            for it_id in self.itens_coletados:
                v = self.item_valor[it_id]
                idx = self.item_cidade_idx.get(it_id)
                if idx is not None:
                    t_coleta = tempo_coleta_cidade[idx]
                    tempo_decorrido = tempo_fim - t_coleta
                    degradacao = math.ceil(tempo_decorrido / self.constante_degradacao)
                    valor_degradado_total += (v - degradacao)

            return valor_degradado_total
        else:
            peso_acc = 0.0
            tempo_viagem = 0.0

            for i in range(self.n):
                peso_acc += self.peso_por_cidade[i]
                vel = self.v_max - peso_acc * self.fator_velocidade
                if vel < self.v_min:
                    vel = self.v_min
                tempo_viagem += self.distancias[i] / vel

            return self.valor_bruto_atual - self.taxa_aluguel * tempo_viagem

    def calcular_lucro(
        self, itens_coletados_ids: Set[int], modelo: Optional[str] = None
    ) -> float:
        """
        Calcula o lucro líquido G para o conjunto de itens selecionados.
        - TTP1: G = valor_bruto - taxa_aluguel * tempo_viagem
        - TTP2: G = soma(valor_item - ceil((tempo_fim - tempo_coleta) / C))
        """
        mod = modelo or self.modelo
        peso_por_cidade = [0.0] * self.n

        for it_id in itens_coletados_ids:
            idx = self.item_cidade_idx.get(it_id)
            if idx is not None:
                peso_por_cidade[idx] += self.item_peso[it_id]

        if mod == "TTP2":
            tempo_coleta_cidade = [0.0] * self.n
            tempo_viagem = 0.0
            peso_acc = 0.0

            for i in range(self.n):
                tempo_coleta_cidade[i] = tempo_viagem
                peso_acc += peso_por_cidade[i]
                vel = self.v_max - peso_acc * self.fator_velocidade
                if vel < self.v_min:
                    vel = self.v_min
                tempo_viagem += self.distancias[i] / vel

            tempo_fim = tempo_viagem
            valor_degradado_total = 0.0

            for it_id in itens_coletados_ids:
                v = self.item_valor[it_id]
                idx = self.item_cidade_idx.get(it_id)
                if idx is not None:
                    t_coleta = tempo_coleta_cidade[idx]
                    tempo_decorrido = tempo_fim - t_coleta
                    degradacao = math.ceil(tempo_decorrido / self.constante_degradacao)
                    valor_degradado_total += (v - degradacao)

            return valor_degradado_total

        else:
            valor_bruto = sum(self.item_valor[it_id] for it_id in itens_coletados_ids)
            peso_acc = 0.0
            tempo_viagem = 0.0

            for i in range(self.n):
                peso_acc += peso_por_cidade[i]
                vel = self.v_max - peso_acc * self.fator_velocidade
                if vel < self.v_min:
                    vel = self.v_min
                tempo_viagem += self.distancias[i] / vel

            return valor_bruto - self.taxa_aluguel * tempo_viagem
