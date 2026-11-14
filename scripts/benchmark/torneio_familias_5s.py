"""
Torneio de Familias (com Timeout Rigido de 5.0s por Execucao).

Avalia todas as variacoes das familias S2/S5, Faulkner (S1-S5, C1-C6), CoSolver e CTTP
com limite rigido de 5.0s (via subprocesso/Process.join(5.0)), identificando o campeao
empirico de cada familia por RDI e Lucro Medio (G).
"""

import concurrent.futures
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_raiz = Path(__file__).resolve().parents[2]
_sim_path = _raiz / "ttp_simulator"
sys.path.insert(0, str(_raiz))
sys.path.insert(0, str(_sim_path))

CANDIDATOS = [
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
    "cosolver",
    "cosolver2b_first_fit",
    "cosolver2b_best_fit",
    "cttp_2opt",
    "cttp_pgch",
]

FAMILIAS = {
    "S2/S5": ["s2_heuristic", "s2_refinada", "s5_heuristic", "s5_refinada"],
    "Faulkner_S": ["faulkner_s1", "faulkner_s2", "faulkner_s3", "faulkner_s4", "faulkner_s5"],
    "Faulkner_C": ["faulkner_c1", "faulkner_c2", "faulkner_c3", "faulkner_c4", "faulkner_c5", "faulkner_c6"],
    "CoSolver": ["cosolver", "cosolver2b_first_fit", "cosolver2b_best_fit"],
    "CTTP": ["cttp_2opt", "cttp_pgch"],
}

TIME_LIMIT = 5.0


def _worker_proc(solver_name: str, path_str: str, time_limit: float, q: mp.Queue):
    import importlib
    import pkgutil
    sys.path.insert(0, str(_raiz))
    sys.path.insert(0, str(_sim_path))
    import src.solvers
    for _mod in pkgutil.walk_packages(src.solvers.__path__, "src.solvers."):
        importlib.import_module(_mod.name)
    from src.core.evaluator_ttp1 import avaliar_ttp1
    from src.scenarios.parser import InstanceParser
    from src.solvers.base import SolverRegistry

    try:
        inst = InstanceParser().parse(path_str)
        solver = SolverRegistry.get(solver_name)
        if hasattr(solver, "time_limit"):
            solver.time_limit = time_limit
        t0 = time.perf_counter()
        sol = solver.solve(inst)
        dt = time.perf_counter() - t0
        res = avaliar_ttp1(inst, sol)
        q.put(("ok", float(res["lucro_liquido_G"]), float(dt)))
    except Exception as exc:
        q.put(("erro", float("nan"), 0.0))


def _run_with_timeout(args):
    solver_name, path_str = args
    inst_name = Path(path_str).name
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker_proc, args=(solver_name, path_str, TIME_LIMIT, q))
    t0 = time.perf_counter()
    p.start()
    p.join(TIME_LIMIT)
    if p.is_alive():
        p.terminate()
        p.join(1.0)
        return solver_name, inst_name, "timeout", float("nan"), TIME_LIMIT
    if not q.empty():
        status, g, dt = q.get_nowait()
        return solver_name, inst_name, status, g, dt
    return solver_name, inst_name, "erro", float("nan"), time.perf_counter() - t0


def main():
    from scripts.benchmark.avaliar_todos_solvers import coletar_instancias

    inst_paths, meta = coletar_instancias()
    eq_list = sorted([k for k, v in meta.items() if v["equilibradas"]])
    tasks = [(s, inst_paths[i]) for s in CANDIDATOS for i in eq_list]
    print(f"Torneio 5s: {len(CANDIDATOS)} solvers x {len(eq_list)} equilibradas = {len(tasks)} execucoes")

    by_inst = defaultdict(dict)
    times = defaultdict(list)
    timeouts = defaultdict(int)
    t0 = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        done = 0
        for s, inst, status, g, dt in ex.map(_run_with_timeout, tasks):
            if status == "ok":
                by_inst[inst][s] = g
                times[s].append(dt)
            elif status == "timeout":
                timeouts[s] += 1
            done += 1
            if done % 68 == 0 or done == len(tasks):
                print(f"Progresso: {done}/{len(tasks)} ({100.0*done/len(tasks):.1f}%) | {time.perf_counter()-t0:.1f}s", flush=True)

    best_i = {inst: max(d.values()) for inst, d in by_inst.items() if d}
    worst_i = {inst: min(d.values()) for inst, d in by_inst.items() if d}

    print(f"\n{'Familia':<12} | {'Solver':<22} | {'OK/34':>5} | {'TO':>2} | {'RDI (%)':>8} | {'Lucro Medio (G)':>15} | {'Tempo (s)':>9}")
    print("-" * 88)
    for fam, members in FAMILIAS.items():
        fam_rows = []
        for s in members:
            gs = []
            rdis = []
            for inst in eq_list:
                if s in by_inst.get(inst, {}):
                    g = by_inst[inst][s]
                    gs.append(g)
                    bi, wi = best_i[inst], worst_i[inst]
                    rdis.append(((bi - g) / (bi - wi) * 100.0) if bi > wi else 0.0)
            mean_g = float(np.mean(gs)) if gs else float("-inf")
            mean_rdi = float(np.mean(rdis)) if rdis else 100.0
            mean_t = float(np.mean(times[s])) if times[s] else 0.0
            fam_rows.append((mean_rdi, -mean_g, s, len(gs), timeouts[s], mean_g, mean_t))
        fam_rows.sort()
        for rdi, _, s, ok_n, to_n, mg, mt in fam_rows:
            print(f"{fam:<12} | {s:<22} | {ok_n:>5} | {to_n:>2} | {rdi:8.2f} | {mg:15.2f} | {mt:9.3f}")
        print("-" * 88)


if __name__ == "__main__":
    main()
