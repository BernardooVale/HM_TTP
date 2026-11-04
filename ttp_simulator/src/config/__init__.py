"""
Módulo de Configurações do TTP Simulator.
Gerencia hiperparâmetros calibrados e configurações persistidas para heurísticas e solucionadores.
"""

from .config_manager import (
    FALLBACK_CONFIGURACOES,
    atualizar_campeao_se_melhor,
    obter_melhor_configuracao,
    salvar_melhor_configuracao,
)

__all__ = [
    "FALLBACK_CONFIGURACOES",
    "obter_melhor_configuracao",
    "salvar_melhor_configuracao",
    "atualizar_campeao_se_melhor",
]
