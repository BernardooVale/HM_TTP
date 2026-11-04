"""
Módulo de Alta Performance do TTP Simulator.
Fornece kernels compilados JIT (Numba) e gerenciamento adaptativo de memória RAM para aceleração.
"""

from .numba_kernels import (
    AdaptiveRAMCache,
    NUMBA_AVAILABLE,
    detectar_memoria_sistema_bytes,
    estimar_tamanho_bytes,
    numba_calcular_menores_caminhos,
    numba_caminho_fixo_2opt,
    numba_caminho_fixo_bb,
    numba_construir_rota_gulosa,
    numba_construir_rota_kruskal,
    numba_dissipar_calor_matriz,
    numba_knapsack_01_bb,
    numba_otimizar_2opt,
    numba_refinar_ttp1,
    numba_refinar_ttp2,
    obter_cache_ram_global,
)

__all__ = [
    "AdaptiveRAMCache",
    "NUMBA_AVAILABLE",
    "detectar_memoria_sistema_bytes",
    "estimar_tamanho_bytes",
    "numba_calcular_menores_caminhos",
    "numba_caminho_fixo_2opt",
    "numba_caminho_fixo_bb",
    "numba_construir_rota_gulosa",
    "numba_construir_rota_kruskal",
    "numba_dissipar_calor_matriz",
    "numba_knapsack_01_bb",
    "numba_otimizar_2opt",
    "numba_refinar_ttp1",
    "numba_refinar_ttp2",
    "obter_cache_ram_global",
]
