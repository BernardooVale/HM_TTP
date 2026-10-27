"""
Pacote avaliacao_r: Avaliação do impacto do parâmetro R no Travelling Thief Problem (TTP).
"""

from .config import AvaliacaoRConfig
from .evaluator import avaliar_instancia_ttp, InstanciaAvaliacaoResultado
from .runner import executar_bateria

__all__ = [
    "AvaliacaoRConfig",
    "avaliar_instancia_ttp",
    "InstanciaAvaliacaoResultado",
    "executar_bateria"
]
