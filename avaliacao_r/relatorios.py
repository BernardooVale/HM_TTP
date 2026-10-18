"""
Gerador de Relatórios, Exportação de Dados e Persistência Segura.

Garante:
- Escrita atômica em disco (sem risco de corrupção de arquivos em caso de interrupção).
- Identificação de instâncias já calculadas para retomada automática (resume).
- Exportação em CSV consolidado, curvas detalhadas de R e JSON individual.
"""

import json
import csv
import os
from pathlib import Path
from typing import Set, List, Dict, Any
from dataclasses import asdict

from .config import AvaliacaoRConfig
from .evaluator import InstanciaAvaliacaoResultado


def obter_instancias_ja_calculadas(config: AvaliacaoRConfig) -> Set[str]:
    """Retorna o conjunto de nomes de instâncias que já foram processadas e salvas com sucesso."""
    calculadas = set()
    pasta_detalhes = config.diretorio_resultados / "detalhes"
    if pasta_detalhes.exists():
        for arq in pasta_detalhes.glob("*.json"):
            # Adiciona o stem do arquivo
            calculadas.add(arq.stem)
            calculadas.add(arq.stem.lower())
            calculadas.add(arq.stem.lower().replace("_", "").replace("-", ""))
            try:
                with open(arq, 'r', encoding='utf-8') as f:
                    dados = json.load(f)
                    nome = dados.get("instancia_nome")
                    if nome:
                        calculadas.add(nome)
                        calculadas.add(nome.lower())
                        calculadas.add(nome.lower().replace("_", "").replace("-", ""))
            except Exception:
                # Se o arquivo estiver corrompido, ignora para reprocessar
                continue
    return calculadas


def _escrita_atomica(caminho_alvo: Path, conteudo_str: str) -> None:
    """Escreve uma string em um arquivo temporário e substitui atomicamente."""
    caminho_alvo.parent.mkdir(parents=True, exist_ok=True)
    temp_path = caminho_alvo.with_suffix(caminho_alvo.suffix + ".tmp")
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(conteudo_str)
    os.replace(temp_path, caminho_alvo)


def salvar_resultado_instancia_json(resultado: InstanciaAvaliacaoResultado, config: AvaliacaoRConfig) -> None:
    """Salva os dados detalhados da instância em JSON de forma atômica."""
    pasta_detalhes = config.diretorio_resultados / "detalhes"
    pasta_detalhes.mkdir(parents=True, exist_ok=True)
    caminho_arquivo = pasta_detalhes / f"{resultado.instancia_nome}.json"
    
    dados_dict = asdict(resultado)
    conteudo_json = json.dumps(dados_dict, indent=2, ensure_ascii=False)
    _escrita_atomica(caminho_arquivo, conteudo_json)


def atualizar_relatorio_consolidado_csv(resultado: InstanciaAvaliacaoResultado, config: AvaliacaoRConfig) -> None:
    """Atualiza o arquivo CSV consolidado com as métricas síntese da instância."""
    caminho_csv = config.diretorio_resultados / "relatorio_geral_r.csv"
    
    colunas = [
        "instancia", "dimension", "num_items", "capacidade", "r_nominal",
        "r_crossover", "r_max_gap", "max_gap_valor",
        "faixa_kp_09_min", "faixa_kp_09_max",
        "faixa_tsp_09_min", "faixa_tsp_09_max",
        "zona_forte_acoplamento_min", "zona_forte_acoplamento_max",
        "status_otimo", "tempo_execucao_s"
    ]
    
    nova_linha = {
        "instancia": resultado.instancia_nome,
        "dimension": resultado.dimension,
        "num_items": resultado.num_items,
        "capacidade": f"{resultado.capacidade:.4f}",
        "r_nominal": f"{resultado.r_nominal:.4f}",
        "r_crossover": f"{resultado.r_crossover:.4f}",
        "r_max_gap": f"{resultado.r_max_gap:.4f}",
        "max_gap_valor": f"{resultado.max_gap_valor:.4f}",
        "faixa_kp_09_min": f"{resultado.faixa_kp_limiar_min:.4f}" if resultado.faixa_kp_limiar_min is not None else "N/A",
        "faixa_kp_09_max": f"{resultado.faixa_kp_limiar_max:.4f}" if resultado.faixa_kp_limiar_max is not None else "N/A",
        "faixa_tsp_09_min": f"{resultado.faixa_tsp_limiar_min:.4f}" if resultado.faixa_tsp_limiar_min is not None else "N/A",
        "faixa_tsp_09_max": f"{resultado.faixa_tsp_limiar_max:.4f}" if resultado.faixa_tsp_limiar_max is not None else "N/A",
        "zona_forte_acoplamento_min": f"{resultado.zona_forte_acoplamento_min:.4f}" if resultado.zona_forte_acoplamento_min is not None else "N/A",
        "zona_forte_acoplamento_max": f"{resultado.zona_forte_acoplamento_max:.4f}" if resultado.zona_forte_acoplamento_max is not None else "N/A",
        "status_otimo": resultado.status_otimo,
        "tempo_execucao_s": f"{resultado.tempo_execucao_s:.2f}"
    }

    linhas_existentes = []
    if caminho_csv.exists():
        with open(caminho_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("instancia") != resultado.instancia_nome:
                    linhas_existentes.append(row)
                    
    linhas_existentes.append(nova_linha)
    
    # Ordena por dimensão e nome
    linhas_existentes.sort(key=lambda r: (int(r.get("dimension", 0)), r.get("instancia", "")))
    
    import io
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=colunas)
    writer.writeheader()
    writer.writerows(linhas_existentes)
    
    _escrita_atomica(caminho_csv, output.getvalue())


def atualizar_curvas_detalhadas_csv(resultado: InstanciaAvaliacaoResultado, config: AvaliacaoRConfig) -> None:
    """Atualiza o arquivo CSV com todos os pontos da curva de R para a instância."""
    caminho_csv = config.diretorio_resultados / "curvas_r_detalhadas.csv"
    
    colunas = [
        "instancia", "fator_r", "taxa_r",
        "z_otimo_ttp", "z_kp_otimo_tsp_otimo", "z_kp_otimo_tsp_guloso",
        "z_tsp_otimo_kp_otimo", "z_tsp_otimo_kp_guloso",
        "rho_kp_otimo", "rho_kp_guloso", "rho_tsp_otimo", "rho_tsp_guloso",
        "gap_interdependencia", "atinge_limiar_kp", "atinge_limiar_tsp",
        "ambos_atingem", "nenhum_atinge", "otimo_comprovado"
    ]
    
    novas_linhas = []
    for p in resultado.pontos_r:
        novas_linhas.append({
            "instancia": resultado.instancia_nome,
            "fator_r": f"{p.fator_r:.4f}",
            "taxa_r": f"{p.taxa_r:.4f}",
            "z_otimo_ttp": f"{p.z_otimo_ttp:.4f}",
            "z_kp_otimo_tsp_otimo": f"{p.z_kp_otimo_tsp_otimo:.4f}",
            "z_kp_otimo_tsp_guloso": f"{p.z_kp_otimo_tsp_guloso:.4f}",
            "z_tsp_otimo_kp_otimo": f"{p.z_tsp_otimo_kp_otimo:.4f}",
            "z_tsp_otimo_kp_guloso": f"{p.z_tsp_otimo_kp_guloso:.4f}",
            "rho_kp_otimo": f"{p.rho_kp_otimo:.4f}",
            "rho_kp_guloso": f"{p.rho_kp_guloso:.4f}",
            "rho_tsp_otimo": f"{p.rho_tsp_otimo:.4f}",
            "rho_tsp_guloso": f"{p.rho_tsp_guloso:.4f}",
            "gap_interdependencia": f"{p.gap_interdependencia:.4f}",
            "atinge_limiar_kp": p.atinge_limiar_kp,
            "atinge_limiar_tsp": p.atinge_limiar_tsp,
            "ambos_atingem": p.ambos_atingem_limiar,
            "nenhum_atinge": p.nenhum_atinge_limiar,
            "otimo_comprovado": p.otimo_comprovado
        })

    linhas_existentes = []
    if caminho_csv.exists():
        with open(caminho_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("instancia") != resultado.instancia_nome:
                    linhas_existentes.append(row)
                    
    linhas_existentes.extend(novas_linhas)
    
    import io
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=colunas)
    writer.writeheader()
    writer.writerows(linhas_existentes)
    
    _escrita_atomica(caminho_csv, output.getvalue())


def salvar_todos_resultados(resultado: InstanciaAvaliacaoResultado, config: AvaliacaoRConfig) -> None:
    """Função unificada de salvamento seguro."""
    salvar_resultado_instancia_json(resultado, config)
    atualizar_relatorio_consolidado_csv(resultado, config)
    atualizar_curvas_detalhadas_csv(resultado, config)


def imprimir_resumo_instancia(res: InstanciaAvaliacaoResultado) -> None:
    """Imprime na saída padrão um resumo legível e estruturado da avaliação."""
    print("\n" + "=" * 78)
    print(f"📊 RESULTADO DA AVALIAÇÃO DE R: {res.instancia_nome}")
    print("=" * 78)
    print(f" • Dimensão: {res.dimension} cidades | {res.num_items} itens | Capacidade W: {res.capacidade:.1f}")
    print(f" • R Nominal: {res.r_nominal:.4f} | Status Ótimo: {res.status_otimo} | Tempo: {res.tempo_execucao_s:.2f}s")
    print("-" * 78)
    print(f" • 🎯 Ponto de Cruzamento (Crossover Rc): {res.r_crossover:.4f}")
    print(f" • ⚡ Ponto de Máxima Interdependência: {res.r_max_gap:.4f} (Gap: {res.max_gap_valor * 100:.1f}%)")
    
    kp_str = f"[{res.faixa_kp_limiar_min:.3f}, {res.faixa_kp_limiar_max:.3f}]" if res.faixa_kp_limiar_min is not None else "Nunca"
    tsp_str = f"[{res.faixa_tsp_limiar_min:.3f}, {res.faixa_tsp_limiar_max:.3f}]" if res.faixa_tsp_limiar_min is not None else "Nunca"
    forte_str = f"[{res.zona_forte_acoplamento_min:.3f}, {res.zona_forte_acoplamento_max:.3f}]" if res.zona_forte_acoplamento_min is not None else "Nenhuma"
    
    print(f" • Faixa R onde KP-First >= 90% do Ótimo:  {kp_str}")
    print(f" • Faixa R onde TSP-First >= 90% do Ótimo: {tsp_str}")
    print(f" • Zona de Acoplamento Forte (ambos < 90%): {forte_str}")
    print("-" * 78)
    print(" Detalhamento dos pontos de R:")
    print(f" {'R':>8} | {'Z* TTP':>10} | {'Z KP_ot':>10} | {'Z TSP_ot':>10} | {'rho_KP':>7} | {'rho_TSP':>7} | {'Gap %':>6}")
    print("-" * 78)
    for p in res.pontos_r:
        print(f" {p.taxa_r:8.3f} | {p.z_otimo_ttp:10.1f} | {p.z_kp_otimo_tsp_otimo:10.1f} | {p.z_tsp_otimo_kp_otimo:10.1f} | {p.rho_kp_otimo*100:6.1f}% | {p.rho_tsp_otimo*100:6.1f}% | {p.gap_interdependencia*100:5.1f}%")
    print("=" * 78 + "\n")
