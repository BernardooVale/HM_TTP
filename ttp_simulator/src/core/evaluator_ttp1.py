from typing import Dict, Any
from .models import TTPInstance, SolucaoTTP

def evaluate(instance: TTPInstance, solucao: SolucaoTTP) -> Dict[str, Any]:
    peso_acumulado = 0.0
    valor_bruto_itens_g = 0.0
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
    
    for i in range(num_cidades):
        cidade_atual_id = rota[i]
        proxima_cidade_id = rota[(i + 1) % num_cidades]
        
        # Coletar itens na cidade atual
        if cidade_atual_id in solucao.plano_coleta:
            itens_a_coletar_ids = set(solucao.plano_coleta[cidade_atual_id])
            cidade_atual = instance.grafo.cidades[cidade_atual_id]
            
            for item in cidade_atual.itens:
                if item.id in itens_a_coletar_ids:
                    if peso_acumulado + item.peso <= instance.capacidade_mochila:
                        peso_acumulado += item.peso
                        valor_bruto_itens_g += item.valor
                        qtd_itens_coletados += 1
                        
        # Calcular velocidade para a próxima perna
        velocidade = instance.v_max - peso_acumulado * ((instance.v_max - instance.v_min) / instance.capacidade_mochila)
        
        # Calcular tempo da perna
        distancia = instance.grafo.distancia(cidade_atual_id, proxima_cidade_id)
        tempo_viagem_f += distancia / velocidade
        
    lucro_liquido_G = valor_bruto_itens_g - instance.taxa_aluguel * tempo_viagem_f
    
    return {
        'lucro_liquido_G': lucro_liquido_G,
        'tempo_viagem_f': tempo_viagem_f,
        'valor_bruto_itens_g': valor_bruto_itens_g,
        'peso_acumulado': peso_acumulado,
        'qtd_itens_coletados': qtd_itens_coletados
    }

avaliar_ttp1 = evaluate

