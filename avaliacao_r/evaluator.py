"""
Mecanismo Central de Avaliação do Impacto de R no TTP.

Compara:
- Solução Ótima do TTP (via B&B Exato ou Cota Incumbente)
- Solução 1: KP Ótimo + TSP Ótimo Condicionado
- Solução 2: KP Ótimo + TSP Guloso
- Solução 3: TSP Ótimo + Packing Ótimo Condicionado
- Solução 4: TSP Ótimo + Packing Guloso

Calcula:
- Relação percentual em relação ao ótimo (rho)
- Preço do Desacoplamento (Interdependence Gap)
- Ponto de Cruzamento (Crossover Point)
- Intervalos de equilíbrio sob o limiar especificado (ex: 0.9x)
"""

import math
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

import sys
from pathlib import Path
_sim_path = str(Path(__file__).resolve().parent.parent / "ttp_simulator")
if _sim_path not in sys.path:
    sys.path.insert(0, _sim_path)

from src.core.models import TTPInstance, SolucaoTTP, Item
from .config import AvaliacaoRConfig
from .exact_solvers import (
    resolver_kp_exato,
    resolver_tsp_exato,
    resolver_rota_exata_itens_fixos,
    resolver_packing_exato_rota_fixa,
    resolver_ttp_exato_bb
)
from .heuristic_solvers import (
    resolver_kp_guloso,
    resolver_tsp_guloso,
    resolver_rota_gulosa_itens_fixos,
    resolver_packing_guloso_rota_fixa
)


@dataclass
class PontoRResultado:
    fator_r: float
    taxa_r: float
    
    # Pontuações Z (Lucro Líquido)
    z_otimo_ttp: float
    z_kp_otimo_tsp_otimo: float
    z_kp_otimo_tsp_guloso: float
    z_tsp_otimo_kp_otimo: float
    z_tsp_otimo_kp_guloso: float
    
    # Razões relativas em relação ao ótimo do TTP (rho = Z / Z_opt)
    rho_kp_otimo: float
    rho_kp_guloso: float
    rho_tsp_otimo: float
    rho_tsp_guloso: float
    
    # Gap de interdependência (1 - max(rho_kp_otimo, rho_tsp_otimo))
    gap_interdependencia: float
    
    # Verificação de limiar (ex: >= 0.9)
    atinge_limiar_kp: bool
    atinge_limiar_tsp: bool
    ambos_atingem_limiar: bool
    nenhum_atinge_limiar: bool
    
    otimo_comprovado: bool


@dataclass
class InstanciaAvaliacaoResultado:
    instancia_nome: str
    dimension: int
    num_items: int
    capacidade: float
    r_nominal: float
    
    # Métricas síntese
    r_crossover: float                     # R onde Z(KP) se aproxima de Z(TSP)
    r_max_gap: float                       # R onde o desacoplamento mais perde qualidade
    max_gap_valor: float                   # Valor máximo de gap observado
    
    # Faixas de R onde cada abordagem atinge o limiar (ex: 0.9)
    faixa_kp_limiar_min: Optional[float]
    faixa_kp_limiar_max: Optional[float]
    faixa_tsp_limiar_min: Optional[float]
    faixa_tsp_limiar_max: Optional[float]
    
    # Zona de forte acoplamento (onde nenhum atinge 0.9)
    zona_forte_acoplamento_min: Optional[float]
    zona_forte_acoplamento_max: Optional[float]
    
    status_otimo: str                      # 'EXATO' ou 'TIMEOUT_HEURISTICO'
    tempo_execucao_s: float
    
    pontos_r: List[PontoRResultado]


def _calcular_rho(z_solucao: float, z_otimo: float) -> float:
    """Calcula a razão rho = z_solucao / z_otimo tratando divisões por zero e valores negativos."""
    if abs(z_otimo) < 1e-6:
        return 1.0 if abs(z_solucao) < 1e-6 else (1.0 if z_solucao > 0 else 0.0)
    
    if z_otimo > 0:
        return max(0.0, z_solucao / z_otimo)
    else:
        # Se z_otimo for negativo (custo de viagem superou ganho), a solução menos negativa é melhor
        return 1.0 - (z_otimo - z_solucao) / abs(z_otimo)


def avaliar_ponto_r(
    instance: TTPInstance,
    taxa_r: float,
    fator_r: float,
    kp_opt_valor: float,
    t_kp_opt_tsp_opt: float,
    t_kp_opt_tsp_gul: float,
    tsp_opt_rota: List[int],
    config: AvaliacaoRConfig,
    plano_kp_dict: Dict[int, List[int]],
    rota_kp_opt_tsp_opt: List[int]
) -> PontoRResultado:
    """Avalia um único ponto de R."""
    # 1. KP Ótimo + TSP Ótimo Condicionado
    z_kp_ot_tsp_ot = kp_opt_valor - taxa_r * t_kp_opt_tsp_opt
    
    # 2. KP Ótimo + TSP Guloso
    z_kp_ot_tsp_gul = kp_opt_valor - taxa_r * t_kp_opt_tsp_gul
    
    # 3. TSP Ótimo + Packing Ótimo Condicionado
    z_tsp_ot_kp_ot, plano_tsp_ot_kp_ot = resolver_packing_exato_rota_fixa(instance, tsp_opt_rota, taxa_r)
    
    # 4. TSP Ótimo + Packing Guloso
    z_tsp_ot_kp_gul, _ = resolver_packing_guloso_rota_fixa(instance, tsp_opt_rota, taxa_r)
    
    # Melhor incumbente inicial
    solucao_kp_first = SolucaoTTP(rota=rota_kp_opt_tsp_opt, plano_coleta=plano_kp_dict)
    solucao_tsp_first = SolucaoTTP(rota=tsp_opt_rota, plano_coleta=plano_tsp_ot_kp_ot)
    
    melhor_incumbente = solucao_kp_first if z_kp_ot_tsp_ot >= z_tsp_ot_kp_ot else solucao_tsp_first
    
    # 5. TTP Integrado Ótimo (Branch & Bound Global)
    num_cidades = instance.dimension or len(instance.grafo.cidades)
    if num_cidades <= config.max_cidades_exato_ttp:
        timeout_ponto = min(5.0, max(0.5, config.timeout_segundos_por_instancia / max(len(config.fatores_r), 1)))
        z_otimo_ttp, sol_opt, otimo_comprovado = resolver_ttp_exato_bb(
            instance, 
            taxa_r, 
            timeout_s=timeout_ponto,
            solucao_incumbente=melhor_incumbente
        )
    else:
        # Para instâncias maiores, usa a melhor incumbente condicionada encontrada
        z_otimo_ttp = max(z_kp_ot_tsp_ot, z_tsp_ot_kp_ot)
        otimo_comprovado = False

    # Garante que z_otimo_ttp seja no mínimo tão bom quanto as soluções conhecidas
    z_otimo_ttp = max(z_otimo_ttp, z_kp_ot_tsp_ot, z_tsp_ot_kp_ot)

    # 6. Razões relativas (rho)
    rho_kp_ot = _calcular_rho(z_kp_ot_tsp_ot, z_otimo_ttp)
    rho_kp_gul = _calcular_rho(z_kp_ot_tsp_gul, z_otimo_ttp)
    rho_tsp_ot = _calcular_rho(z_tsp_ot_kp_ot, z_otimo_ttp)
    rho_tsp_gul = _calcular_rho(z_tsp_ot_kp_gul, z_otimo_ttp)
    
    # Gap de interdependência
    melhor_desacoplado = max(rho_kp_ot, rho_tsp_ot)
    gap = max(0.0, 1.0 - melhor_desacoplado)
    
    limiar = config.limiar_qualidade
    atinge_kp = (rho_kp_ot >= limiar)
    atinge_tsp = (rho_tsp_ot >= limiar)
    
    return PontoRResultado(
        fator_r=fator_r,
        taxa_r=taxa_r,
        z_otimo_ttp=z_otimo_ttp,
        z_kp_otimo_tsp_otimo=z_kp_ot_tsp_ot,
        z_kp_otimo_tsp_guloso=z_kp_ot_tsp_gul,
        z_tsp_otimo_kp_otimo=z_tsp_ot_kp_ot,
        z_tsp_otimo_kp_guloso=z_tsp_ot_kp_gul,
        rho_kp_otimo=rho_kp_ot,
        rho_kp_guloso=rho_kp_gul,
        rho_tsp_otimo=rho_tsp_ot,
        rho_tsp_guloso=rho_tsp_gul,
        gap_interdependencia=gap,
        atinge_limiar_kp=atinge_kp,
        atinge_limiar_tsp=atinge_tsp,
        ambos_atingem_limiar=(atinge_kp and atinge_tsp),
        nenhum_atinge_limiar=(not atinge_kp and not atinge_tsp),
        otimo_comprovado=otimo_comprovado
    )


def avaliar_instancia_ttp(
    instance: TTPInstance, 
    config: Optional[AvaliacaoRConfig] = None
) -> InstanciaAvaliacaoResultado:
    """
    Executa a avaliação completa de uma instância TTP ao longo da variação de R.
    """
    import time
    start_time = time.time()
    
    if config is None:
        config = AvaliacaoRConfig()

    num_cidades = instance.dimension or len(instance.grafo.cidades)
    num_itens = instance.num_items or len(instance.itens)
    capacidade = instance.capacidade_mochila
    r_nominal = instance.taxa_aluguel if instance.taxa_aluguel > 0 else 1.0

    # 1. Resolve os subproblemas independentes (calculados UMA ÚNICA VEZ)
    kp_opt_valor, kp_opt_peso, kp_opt_ids = resolver_kp_exato(instance.itens, capacidade)
    itens_kp_opt = [it for it in instance.itens if it.id in kp_opt_ids]
    
    # Plano de coleta para a solução KP
    plano_kp_dict: Dict[int, List[int]] = {}
    for it in itens_kp_opt:
        plano_kp_dict.setdefault(it.cidade_origem, []).append(it.id)
        
    # TSP ótimo e guloso
    tsp_opt_dist, tsp_opt_rota = resolver_tsp_exato(instance.grafo)
    
    # Rotas para o conjunto de itens do KP ótimo (ótima vs gulosa)
    t_kp_opt_tsp_opt, rota_kp_opt_tsp_opt = resolver_rota_exata_itens_fixos(
        instance.grafo, itens_kp_opt, capacidade, instance.v_min, instance.v_max
    )
    t_kp_opt_tsp_gul, _ = resolver_rota_gulosa_itens_fixos(
        instance.grafo, itens_kp_opt, capacidade, instance.v_min, instance.v_max
    )

    # 2. Avalia ao longo da grade de R
    pontos_resultados: List[PontoRResultado] = []
    
    for fator in config.fatores_r:
        taxa_r_atual = fator * r_nominal
        ponto = avaliar_ponto_r(
            instance=instance,
            taxa_r=taxa_r_atual,
            fator_r=fator,
            kp_opt_valor=kp_opt_valor,
            t_kp_opt_tsp_opt=t_kp_opt_tsp_opt,
            t_kp_opt_tsp_gul=t_kp_opt_tsp_gul,
            tsp_opt_rota=tsp_opt_rota,
            config=config,
            plano_kp_dict=plano_kp_dict,
            rota_kp_opt_tsp_opt=rota_kp_opt_tsp_opt
        )
        pontos_resultados.append(ponto)

    # 3. Análise de Crossover, Gaps e Intervalos
    # Ponto de crossover: onde a diferença absoluta |Z_kp - Z_tsp| é mínima
    ponto_crossover = min(
        pontos_resultados, 
        key=lambda p: abs(p.z_kp_otimo_tsp_otimo - p.z_tsp_otimo_kp_otimo)
    )
    r_crossover = ponto_crossover.taxa_r

    # Ponto de gap máximo
    ponto_max_gap = max(pontos_resultados, key=lambda p: p.gap_interdependencia)
    r_max_gap = ponto_max_gap.taxa_r
    max_gap_valor = ponto_max_gap.gap_interdependencia

    # Faixas de limiar (ex: 0.9)
    kp_validos = [p.taxa_r for p in pontos_resultados if p.atinge_limiar_kp]
    faixa_kp_min = min(kp_validos) if kp_validos else None
    faixa_kp_max = max(kp_validos) if kp_validos else None

    tsp_validos = [p.taxa_r for p in pontos_resultados if p.atinge_limiar_tsp]
    faixa_tsp_min = min(tsp_validos) if tsp_validos else None
    faixa_tsp_max = max(tsp_validos) if tsp_validos else None

    # Zona forte de acoplamento (onde nenhum atinge o limiar de 0.9)
    nenhum_validos = [p.taxa_r for p in pontos_resultados if p.nenhum_atinge_limiar]
    zona_forte_min = min(nenhum_validos) if nenhum_validos else None
    zona_forte_max = max(nenhum_validos) if nenhum_validos else None

    total_time = time.time() - start_time
    status = "EXATO" if num_cidades <= config.max_cidades_exato_ttp else "TIMEOUT_HEURISTICO"

    return InstanciaAvaliacaoResultado(
        instancia_nome=instance.nome,
        dimension=num_cidades,
        num_items=num_itens,
        capacidade=capacidade,
        r_nominal=r_nominal,
        r_crossover=r_crossover,
        r_max_gap=r_max_gap,
        max_gap_valor=max_gap_valor,
        faixa_kp_limiar_min=faixa_kp_min,
        faixa_kp_limiar_max=faixa_kp_max,
        faixa_tsp_limiar_min=faixa_tsp_min,
        faixa_tsp_limiar_max=faixa_tsp_max,
        zona_forte_acoplamento_min=zona_forte_min,
        zona_forte_acoplamento_max=zona_forte_max,
        status_otimo=status,
        tempo_execucao_s=total_time,
        pontos_r=pontos_resultados
    )
