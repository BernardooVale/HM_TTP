from typing import Dict, Any
import math
from .models import TTPInstance, SolucaoTTP

def evaluate(instance: TTPInstance, solucao: SolucaoTTP) -> Dict[str, Any]:
    peso_acumulado = 0.0
    tempo_viagem_f = 0.0
    qtd_itens_coletados = 0
    
    rota = solucao.rota
    if not rota:
        return {
            'lucro_liquido_G': 0.0,
            'tempo_viagem_f': 0.0,
            'valor_bruto_itens_g': 0.0,
            'peso_acumulado': 0.0,
            'qtd_itens_coletados': 0
        }
        
    num_cidades = len(rota)
    
    # Precisamos rastrear quando cada item foi coletado para calcular a degradação no final
    # List de tuplas: (item, tempo_quando_coletado)
    itens_coletados = []
    
    for i in range(num_cidades):
        cidade_atual_id = rota[i]
        proxima_cidade_id = rota[(i + 1) % num_cidades]
        
        if cidade_atual_id in solucao.plano_coleta:
            itens_a_coletar_ids = set(solucao.plano_coleta[cidade_atual_id])
            cidade_atual = instance.grafo.cidades[cidade_atual_id]
            
            for item in cidade_atual.itens:
                if item.id in itens_a_coletar_ids:
                    if peso_acumulado + item.peso <= instance.capacidade_mochila:
                        peso_acumulado += item.peso
                        qtd_itens_coletados += 1
                        itens_coletados.append((item, tempo_viagem_f))
                        
        velocidade = instance.v_max - peso_acumulado * ((instance.v_max - instance.v_min) / instance.capacidade_mochila)
        distancia = instance.grafo.distancia(cidade_atual_id, proxima_cidade_id)
        tempo_viagem_f += distancia / velocidade
        
    # Calcular valor degradado dos itens
    valor_bruto_itens_g = 0.0
    for item, tempo_coleta in itens_coletados:
        tempo_decorrido = tempo_viagem_f - tempo_coleta
        valor_degradado = item.valor - math.ceil(tempo_decorrido / instance.constante_degradacao)
        # O valor pode ficar negativo, dependendo da interpretação, mas a fórmula diz p_k - ceil(T_k / C)
        valor_bruto_itens_g += valor_degradado
    
    lucro_liquido_G = valor_bruto_itens_g # Para TTP2, G(z) = g(z) com valores degradados, sem custo de aluguel
    
    return {
        'lucro_liquido_G': lucro_liquido_G,
        'tempo_viagem_f': tempo_viagem_f,
        'valor_bruto_itens_g': valor_bruto_itens_g,
        'peso_acumulado': peso_acumulado,
        'qtd_itens_coletados': qtd_itens_coletados
    }

avaliar_ttp2 = evaluate

