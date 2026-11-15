"""
Benchmark Comparativo Completo de Solvers TTP1.

Avalia todos os solvers (construtivos, iterativos, literatura, Queimada construtiva/hibrida,
ReinforceNS e ReinforceNS com hot-start Queimada) — excluindo heuristic_37* — nas instancias
'equilibradas' e 'bateria_testes/reinforcens', com time_limit padronizado em 5.0s e
multiplas seeds para solvers estocasticos.

Calcula RDI (Relative Deviation Index), Lucro Medio (G), Desvio Padrao, Melhor Lucro,
Taxa de Vitorias e Tempo Medio, separando por suite (Equilibradas, ReinforceNS e Geral).
"""

import argparse
import concurrent.futures
import csv
import importlib
import os
import pkgutil
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_raiz = Path(__file__).resolve().parents[2]
_sim_path = _raiz / "ttp_simulator"
sys.path.insert(0, str(_raiz))
sys.path.insert(0, str(_sim_path))

from src.core.evaluator_ttp1 import avaliar_ttp1
from src.scenarios.parser import InstanceParser
from src.solvers.base import SolverRegistry
from src.utils.results_manager import obter_instancias_equilibradas

# Importa todos os modulos de solvers para popular o SolverRegistry
import src.solvers

for _mod in pkgutil.walk_packages(src.solvers.__path__, "src.solvers."):
    importlib.import_module(_mod.name)

SOLVERS_CAMPEOES = [
    # Construtivos classicos & Faulkner (2015) - campeoes empiricos em <=5s
    "s5_refinada",
    "faulkner_c3",
    # Literatura TTP
    "pack_iterative",
    "local_search",
    "cosolver",
    "cttp_2opt",
    "coco_solver",
    "wsm_solver",
    # Queimada (campea construtiva e campea hibrida/iterativa)
    "queimada_construtiva",
    "queimada",
    # ReinforceNS (base + PM + Hot-Start Queimada)
    "reinforcens",
    "reinforcens_pm",
    "reinforcens_queimada",
    "reinforcens_pm_queimada",
]

SOLVERS_RL = [
    "reinforcens",
    "reinforcens_pm",
    "reinforcens_queimada",
    "reinforcens_pm_queimada",
]

SOLVERS_COMPLETO = [
    # Baselines
    "random",
    "greedy",
    # Construtivos classicos & Faulkner (2015)
    "s2_heuristic",
    "s2_refinada",
    "s5_heuristic",
    "s5_refinada",
    "faulkner_s1",
    "faulkner_s2",
    "faulkner_s3",
    "faulkner_s4",
    "faulkner_s5",
    "faulkner_c1",
    "faulkner_c2",
    "faulkner_c3",
    "faulkner_c4",
    "faulkner_c5",
    "faulkner_c6",
    # Literatura TTP
    "pack_iterative",
    "cosolver",
    "local_search",
    "cosolver2b_first_fit",
    "cosolver2b_best_fit",
    "cttp_2opt",
    "cttp_pgch",
    "coco_solver",
    "wsm_solver",
    # Queimada (campeas calibradas + variantes literatura)
    "queimada_construtiva",
    "queimada",
    "queimada_pgch_coco",
    "queimada_insertion_bitflip",
    "queimada_boundary_bitflip",
    "queimada_pack_iterative",
    # ReinforceNS (RL + Hot-Start Queimada)
    "reinforcens",
    "reinforcens_pm",
    "reinforcens_queimada",
    "reinforcens_pm_queimada",
]

STOCHASTIC_SOLVERS = {
    "random",
    "local_search",
    "reinforcens",
    "reinforcens_pm",
    "reinforcens_queimada",
    "reinforcens_pm_queimada",
}

SEEDS = [42, 123, 456]
TIME_LIMIT_STD = 5.0

_INST_CACHE: Dict[str, Any] = {}


def _load_instance(path_str: str):
    if path_str not in _INST_CACHE:
        _INST_CACHE[path_str] = InstanceParser().parse(path_str)
    return _INST_CACHE[path_str]


def _run_task(args: Tuple[str, str, int, float]) -> Dict[str, Any]:
    solver_name, path_str, seed, time_limit = args
    inst = _load_instance(path_str)
    inst_name = Path(path_str).name

    random.seed(seed)
    np.random.seed(seed)

    solver = SolverRegistry.get(solver_name)
    if hasattr(solver, "time_limit"):
        solver.time_limit = time_limit
    if hasattr(solver, "time_limit_pos"):
        solver.time_limit_pos = time_limit
    if hasattr(solver, "seed"):
        solver.seed = seed

    t0 = time.perf_counter()
    try:
        sol = solver.solve(inst)
        dt = time.perf_counter() - t0
        res = avaliar_ttp1(inst, sol)
        return {
            "solver": solver_name,
            "instancia": inst_name,
            "seed": seed,
            "lucro_G": float(res["lucro_liquido_G"]),
            "tempo_s": float(dt),
            "itens": int(res["qtd_itens_coletados"]),
            "status": "ok",
            "erro": "",
        }
    except Exception as exc:
        dt = time.perf_counter() - t0
        return {
            "solver": solver_name,
            "instancia": inst_name,
            "seed": seed,
            "lucro_G": float("nan"),
            "tempo_s": float(dt),
            "itens": 0,
            "status": "erro",
            "erro": f"{type(exc).__name__}: {exc}",
        }


def coletar_instancias() -> Tuple[Dict[str, str], Dict[str, Dict[str, bool]]]:
    eq_names = obter_instancias_equilibradas()
    bateria_dir = _sim_path / "bateria_testes"
    reinforcens_dir = bateria_dir / "reinforcens"
    aleatorias_dir = bateria_dir / "aleatorias_equilibradas"

    inst_paths: Dict[str, str] = {}
    meta: Dict[str, Dict[str, bool]] = defaultdict(
        lambda: {"equilibradas": False, "reinforcens": False, "aleatorias_equilibradas": False}
    )

    # 1. Equilibradas (prioriza arquivos fora de subpastas ou primeiro encontrado)
    for p in sorted(bateria_dir.rglob("*"), key=lambda x: (len(x.parts), str(x))):
        if p.is_file() and p.name in eq_names:
            if p.name not in inst_paths:
                inst_paths[p.name] = str(p)
            meta[p.name]["equilibradas"] = True

    # 2. ReinforceNS
    for p in sorted(reinforcens_dir.glob("*.ttp")):
        if p.name not in inst_paths:
            inst_paths[p.name] = str(p)
        meta[p.name]["reinforcens"] = True

    # 3. Aleatorias Equilibradas
    if aleatorias_dir.is_dir():
        for ext in ("*.txt", "*.ttp"):
            for p in sorted(aleatorias_dir.glob(ext)):
                if p.name not in inst_paths:
                    inst_paths[p.name] = str(p)
                meta[p.name]["aleatorias_equilibradas"] = True

    return inst_paths, meta


def calcular_resumo(
    rows: List[Dict[str, Any]],
    instancias_alvo: List[str],
    solvers_ativos: List[str],
    titulo: str,
    out_csv: Path,
) -> List[Dict[str, Any]]:
    alvo_set = set(instancias_alvo)
    sub = [r for r in rows if r["instancia"] in alvo_set and r["status"] == "ok"]

    # Agrupa por (solver, instancia)
    by_si: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    time_si: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in sub:
        by_si[(r["solver"], r["instancia"])].append(r["lucro_G"])
        time_si[(r["solver"], r["instancia"])].append(r["tempo_s"])

    # Melhor e pior por instancia (usando a media por solver na instancia para RDI justo)
    best_inst: Dict[str, float] = {}
    worst_inst: Dict[str, float] = {}
    for inst in instancias_alvo:
        means = [np.mean(by_si[(s, inst)]) for s in solvers_ativos if (s, inst) in by_si]
        if means:
            best_inst[inst] = float(max(means))
            worst_inst[inst] = float(min(means))

    summary_rows = []
    for s in solvers_ativos:
        inst_means = []
        inst_bests = []
        inst_stds = []
        inst_rdis = []
        inst_times = []
        wins = 0

        for inst in instancias_alvo:
            vals = by_si.get((s, inst))
            if not vals:
                continue
            m = float(np.mean(vals))
            b = float(np.max(vals))
            sd = float(np.std(vals))
            t_m = float(np.mean(time_si[(s, inst)]))
            b_i = best_inst[inst]
            w_i = worst_inst[inst]
            if b_i > w_i:
                rdi = ((b_i - m) / (b_i - w_i)) * 100.0
            else:
                rdi = 0.0
            if abs(m - b_i) <= 1e-6:
                wins += 1

            inst_means.append(m)
            inst_bests.append(b)
            inst_stds.append(sd)
            inst_rdis.append(rdi)
            inst_times.append(t_m)

        if not inst_means:
            continue

        summary_rows.append(
            {
                "solver": s,
                "n_instancias": len(inst_means),
                "rdi_medio_pct": float(np.mean(inst_rdis)),
                "lucro_medio_G": float(np.mean(inst_means)),
                "melhor_lucro_medio": float(np.mean(inst_bests)),
                "std_medio_seeds": float(np.mean(inst_stds)),
                "vitorias": wins,
                "tempo_medio_s": float(np.mean(inst_times)),
            }
        )

    # Ordena por menor RDI medio (e maior lucro medio em caso de empate)
    summary_rows.sort(key=lambda x: (x["rdi_medio_pct"], -x["lucro_medio_G"]))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "solver",
                "n_instancias",
                "rdi_medio_pct",
                "lucro_medio_G",
                "melhor_lucro_medio",
                "std_medio_seeds",
                "vitorias",
                "tempo_medio_s",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n{'='*105}")
    print(f"RANKING: {titulo} ({len(instancias_alvo)} instancias)")
    print(f"{'='*105}")
    print(
        f"{'Rank':<4} | {'Solver':<26} | {'RDI (%)':>8} | {'Lucro Medio (G)':>16} | "
        f"{'Std Seeds':>10} | {'Vitorias':>8} | {'Tempo (s)':>9}"
    )
    print("-" * 105)
    for rk, row in enumerate(summary_rows, 1):
        print(
            f"{rk:<4} | {row['solver']:<26} | {row['rdi_medio_pct']:8.2f} | "
            f"{row['lucro_medio_G']:16.2f} | {row['std_medio_seeds']:10.2f} | "
            f"{row['vitorias']:>8} | {row['tempo_medio_s']:9.3f}"
        )
    return summary_rows


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark comparativo de solvers TTP1 com RDI e Lucro Medio."
    )
    parser.add_argument(
        "--modo",
        choices=["campeoes", "completo", "rl"],
        default="campeoes",
        help="Conjunto de solvers: 'campeoes' (padrao, 14 representantes), 'completo' (36) ou 'rl' (4 ReinforceNS).",
    )
    parser.add_argument(
        "--instancias",
        choices=["todas", "equilibradas", "reinforcens", "aleatorias_equilibradas"],
        default="todas",
        help="Subconjunto de instancias: 'todas' (equilibradas + reinforcens), 'equilibradas', 'reinforcens' ou 'aleatorias_equilibradas'.",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=None,
        help="Lista explicita de solvers (sobrescreve --modo).",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=TIME_LIMIT_STD,
        help=f"Limite de tempo padrao em segundos (padrao: {TIME_LIMIT_STD}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(12, os.cpu_count() or 8),
        help="Numero de processos paralelos.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmark_completo",
        help="Diretorio de saida para os CSVs.",
    )
    args = parser.parse_args()

    if args.solvers:
        solvers_ativos = args.solvers
    elif args.modo == "completo":
        solvers_ativos = SOLVERS_COMPLETO
    elif args.modo == "rl":
        solvers_ativos = SOLVERS_RL
    else:
        solvers_ativos = SOLVERS_CAMPEOES

    out_dir = (_raiz / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = out_dir / "execucoes_detalhadas.csv"

    inst_paths, meta = coletar_instancias()
    eq_list = sorted([k for k, v in meta.items() if v["equilibradas"]])
    rf_list = sorted([k for k, v in meta.items() if v["reinforcens"]])
    ale_list = sorted([k for k, v in meta.items() if v["aleatorias_equilibradas"]])
    if args.instancias == "equilibradas":
        target_list = eq_list
    elif args.instancias == "reinforcens":
        target_list = rf_list
    elif args.instancias == "aleatorias_equilibradas":
        target_list = ale_list
    else:
        target_list = sorted(set(eq_list) | set(rf_list))

    print(
        f"Instancias alvo: {len(target_list)} ({args.instancias}) | "
        f"Modo: {args.modo} ({len(solvers_ativos)} solvers) | "
        f"Time limit: {args.time_limit}s | Seeds estocasticos: {SEEDS}"
    )

    tasks = []
    for s in solvers_ativos:
        seeds = SEEDS if s in STOCHASTIC_SOLVERS else [42]
        for inst_name in target_list:
            path_str = inst_paths[inst_name]
            for seed in seeds:
                tasks.append((s, path_str, seed, args.time_limit))

    print(f"Total de execucoes programadas: {len(tasks)} (workers={args.workers})")
    rows: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_run_task, t) for t in tasks]
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            res["in_equilibradas"] = meta[res["instancia"]]["equilibradas"]
            res["in_reinforcens"] = meta[res["instancia"]]["reinforcens"]
            res["in_aleatorias_equilibradas"] = meta[res["instancia"]]["aleatorias_equilibradas"]
            rows.append(res)
            done += 1
            if done % 25 == 0 or done == len(tasks):
                elapsed = time.perf_counter() - t_start
                print(f"Progresso: {done}/{len(tasks)} ({100.0*done/len(tasks):.1f}%) | {elapsed:.1f}s", flush=True)

    rows.sort(key=lambda r: (r["solver"], r["instancia"], r["seed"]))

    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "solver",
                "instancia",
                "seed",
                "lucro_G",
                "tempo_s",
                "itens",
                "in_equilibradas",
                "in_reinforcens",
                "in_aleatorias_equilibradas",
                "status",
                "erro",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    erros = [r for r in rows if r["status"] != "ok"]
    if erros:
        print(f"\nAVISO: {len(erros)} execucoes falharam:")
        for e in erros[:10]:
            print(f"  - {e['solver']} @ {e['instancia']} (seed={e['seed']}): {e['erro']}")

    if args.instancias in ("todas", "equilibradas"):
        calcular_resumo(rows, eq_list, solvers_ativos, "EQUILIBRADAS", out_dir / "resumo_equilibradas.csv")
    if args.instancias in ("todas", "reinforcens"):
        calcular_resumo(rows, rf_list, solvers_ativos, "REINFORCENS", out_dir / "resumo_reinforcens.csv")
    if args.instancias == "aleatorias_equilibradas":
        calcular_resumo(
            rows,
            ale_list,
            solvers_ativos,
            "ALEATORIAS EQUILIBRADAS",
            out_dir / "resumo_aleatorias_equilibradas.csv",
        )
    if args.instancias == "todas":
        calcular_resumo(rows, target_list, solvers_ativos, "CONSOLIDADO GERAL", out_dir / "resumo_geral.csv")
    print(f"\nCSV detalhado salvo em: {raw_csv}")


if __name__ == "__main__":
    main()
