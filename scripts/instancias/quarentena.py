"""
Script de Quarentena de Instâncias do Traveling Thief Problem (TTP).

Identifica as instâncias da pasta 'bateria_testes' que não tiveram todas as suas
execuções concluídas no CSV de resultados detalhados e as move para uma subpasta
'quarentena' dentro de 'ttp_simulator/bateria_testes'.

Comandos úteis:
    python quarentena.py              # Identifica e move instâncias incompletas para quarentena
    python quarentena.py --listar     # Apenas lista instâncias incompletas (dry-run, sem mover)
    python quarentena.py --restaurar  # Restaura instâncias da quarentena de volta para bateria_testes
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Configura stdout para UTF-8 de forma segura no Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configura paths
_raiz = Path(__file__).resolve().parents[2]
_sim_path = _raiz / "ttp_simulator"
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))
if str(_sim_path) not in sys.path:
    sys.path.insert(0, str(_sim_path))

import src.solvers  # Trigger registration de solvers
from src.runners.batch_runner import _obter_metadados_tamanho
from src.solvers.base import SolverRegistry


def obter_chaves_concluidas(csv_path: Path) -> Set[Tuple[str, str, str, int]]:
    """Lê o CSV detalhado e retorna o conjunto de tuplas (arquivo, modelo, solver, seed)."""
    concluidas = set()
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return concluidas

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inst = row.get("arquivo_instancia", "").strip()
            mod = row.get("modelo_ttp", "").strip()
            solv = row.get("solucionador", "").strip()
            seed_str = row.get("seed", "42").strip()
            try:
                seed = int(seed_str)
            except ValueError:
                seed = 42
            if inst and mod and solv:
                concluidas.add((inst, mod, solv, seed))
    return concluidas


def verificar_instancias(
    bateria_dir: Path,
    csv_path: Path,
    modelos: List[str],
    solvers: List[str],
    seed_base: int = 42,
    runs: int = 1,
) -> Tuple[List[dict], List[dict]]:
    """
    Classifica as instâncias em completas e incompletas com base nas execuções esperadas.
    """
    concluidas = obter_chaves_concluidas(csv_path)

    # Coleta arquivos de instância apenas no nível raiz da pasta de testes
    extensoes = ["*.ttp", "*.txt"]
    arquivos: List[Path] = []
    for ext in extensoes:
        for f in bateria_dir.glob(ext):
            if f.is_file() and "quarentena" not in f.parts:
                arquivos.append(f)

    # Remove duplicados preservando ordem estável
    vistos = set()
    arquivos_unicos = []
    for arq in arquivos:
        if arq.name not in vistos:
            vistos.add(arq.name)
            arquivos_unicos.append(arq)

    arquivos_unicos.sort(key=lambda a: (_obter_metadados_tamanho(a), a.name))

    esperado_total = len(modelos) * len(solvers) * runs
    completas = []
    incompletas = []

    for arq in arquivos_unicos:
        executadas = 0
        faltantes = []
        for s in solvers:
            for m in modelos:
                for r_idx in range(runs):
                    s_val = seed_base + r_idx
                    chave = (arq.name, m, s, s_val)
                    if chave in concluidas:
                        executadas += 1
                    else:
                        faltantes.append((m, s, s_val))

        meta = _obter_metadados_tamanho(arq)
        info = {
            "caminho": arq,
            "nome": arq.name,
            "dimensao": meta[0],
            "itens": meta[1],
            "tamanho_bytes": meta[2],
            "executadas": executadas,
            "esperadas": esperado_total,
            "faltantes": faltantes,
            "pct_concluida": (executadas / esperado_total * 100.0) if esperado_total > 0 else 0.0,
        }

        if executadas >= esperado_total:
            completas.append(info)
        else:
            incompletas.append(info)

    return completas, incompletas


def mover_para_quarentena(incompletas: List[dict], pasta_quarentena: Path) -> int:
    """Move os arquivos incompletos para a pasta de quarentena."""
    pasta_quarentena.mkdir(parents=True, exist_ok=True)
    movidos = 0
    for info in incompletas:
        origem: Path = info["caminho"]
        destino = pasta_quarentena / origem.name
        try:
            shutil.move(str(origem), str(destino))
            movidos += 1
            print(f"  📦 [QUARENTENA] {origem.name:<45} -> {destino.parent.name}/{destino.name}")
        except Exception as e:
            print(f"  ❌ Erro ao mover {origem.name}: {e}")
    return movidos


def restaurar_quarentena(pasta_quarentena: Path, bateria_dir: Path) -> int:
    """Restaura arquivos da pasta quarentena de volta para bateria_testes."""
    if not pasta_quarentena.exists():
        print(f"Pasta de quarentena não encontrada em: {pasta_quarentena}")
        return 0

    arquivos = [f for f in pasta_quarentena.iterdir() if f.is_file() and f.suffix in (".ttp", ".txt")]
    if not arquivos:
        print(f"Nenhum arquivo encontrado em {pasta_quarentena} para restaurar.")
        return 0

    restaurados = 0
    for f in arquivos:
        destino = bateria_dir / f.name
        try:
            shutil.move(str(f), str(destino))
            restaurados += 1
            print(f"  🔄 [RESTAURADO] {f.name} -> {bateria_dir.name}/{f.name}")
        except Exception as e:
            print(f"  ❌ Erro ao restaurar {f.name}: {e}")
    return restaurados


def main():
    parser = argparse.ArgumentParser(
        description="Gerenciador de Quarentena de Instâncias do Traveling Thief Problem (TTP)."
    )
    parser.add_argument(
        "--listar",
        action="store_true",
        help="Apenas lista as instâncias que estão incompletas sem movê-las (dry-run).",
    )
    parser.add_argument(
        "--restaurar",
        action="store_true",
        help="Restaura os arquivos da pasta quarentena de volta para bateria_testes.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(_raiz / "results" / "todos" / "completo.csv" if (_raiz / "results" / "todos" / "completo.csv").exists() else _raiz / "results" / "bateria_solucoes_detalhada.csv"),
        help="Caminho do CSV detalhado de execuções.",
    )
    parser.add_argument(
        "--bateria",
        type=str,
        default=str(_sim_path / "bateria_testes"),
        help="Diretório de instâncias de teste.",
    )

    args = parser.parse_args()

    bateria_dir = Path(args.bateria).resolve()
    pasta_quarentena = bateria_dir / "quarentena"
    csv_path = Path(args.csv).resolve()

    print("\n" + "=" * 90)
    print(f"{'GERENCIADOR DE QUARENTENA DE INSTÂNCIAS TTP':^90}")
    print(f" • Diretório de Testes: {bateria_dir}")
    print(f" • Pasta de Quarentena: {pasta_quarentena}")
    print(f" • CSV de Resultados:   {csv_path}")
    print("=" * 90 + "\n")

    if args.restaurar:
        print("Iniciando restauração de arquivos em quarentena...")
        total_rest = restaurar_quarentena(pasta_quarentena, bateria_dir)
        print(f"\n✅ Total de instâncias restauradas com sucesso: {total_rest}\n")
        return

    solvers = SolverRegistry.listar()
    modelos = ["TTP1", "TTP2"]

    print(f"Verificando status de execuções para {len(solvers)} solucionadores e {len(modelos)} modelos...")
    completas, incompletas = verificar_instancias(
        bateria_dir=bateria_dir,
        csv_path=csv_path,
        modelos=modelos,
        solvers=solvers,
        seed_base=42,
        runs=1,
    )

    total_inst = len(completas) + len(incompletas)
    print(f"\nStatus geral das instâncias encontradas em {bateria_dir.name}:")
    print(f" • Total avaliadas:           {total_inst}")
    print(f" • 100% Concluídas:          {len(completas)} instâncias")
    print(f" • Incompletas / Pendentes:   {len(incompletas)} instâncias\n")

    if not incompletas:
        print("🎉 Parabéns! Todas as instâncias em bateria_testes já foram executadas por todas as soluções.")
        return

    print("-" * 90)
    print(f"{'Instância':<42} | {'Cidades':>7} | {'Itens':>7} | {'Feitas':>7} | {'Total':>7} | {'Status':>8}")
    print("-" * 90)
    for info in incompletas:
        st = f"{info['pct_concluida']:>6.1f}%"
        print(
            f"{info['nome']:<42} | {info['dimensao']:>7} | {info['itens']:>7} | "
            f"{info['executadas']:>7} | {info['esperadas']:>7} | {st:>8}"
        )
    print("-" * 90 + "\n")

    if args.listar:
        print("Modo --listar ativado. Nenhum arquivo foi movido.")
        print(f"Para mover estas {len(incompletas)} instâncias para a quarentena, execute: python quarentena.py")
    else:
        print(f"Movendo {len(incompletas)} instâncias incompletas para '{pasta_quarentena}'...")
        total_mov = mover_para_quarentena(incompletas, pasta_quarentena)
        print(f"\n✅ Concluído! {total_mov} instâncias foram movidas com segurança para a quarentena.")
        print("Dessa forma, sua próxima execução do orquestrador ou bateria não processará esses arquivos.")
        print("Para devolvê-los à bateria_testes no futuro, use: python quarentena.py --restaurar\n")


if __name__ == "__main__":
    main()
