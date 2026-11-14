import concurrent.futures
import itertools
import sys
import time
from pathlib import Path

_raiz = Path(__file__).resolve().parents[2]
_sim_path = _raiz / "ttp_simulator"
sys.path.insert(0, str(_raiz))
sys.path.insert(0, str(_sim_path))

from src.config.config_manager import obter_melhor_configuracao, salvar_melhor_configuracao
from src.core.evaluator_ttp1 import avaliar_ttp1
from src.scenarios.parser import InstanceParser
from src.solvers.constructive.queimada import QueimadaSolver
from src.utils.results_manager import obter_instancias_equilibradas

_INST_CACHE = {}


def _load_inst(path_str: str):
    if path_str not in _INST_CACHE:
        _INST_CACHE[path_str] = InstanceParser().parse(path_str)
    return _INST_CACHE[path_str]


def _eval_one(args):
    idx, params, path_str = args
    inst = _load_inst(path_str)
    solver = QueimadaSolver(**params)
    t0 = time.perf_counter()
    sol = solver.solve(inst)
    dt = time.perf_counter() - t0
    res = avaliar_ttp1(inst, sol)
    return idx, res["lucro_liquido_G"], dt


def main():
    base_cfg = obter_melhor_configuracao("queimada")
    base_params = dict(base_cfg)
    import json
    cfg_json = json.loads((_sim_path / "src" / "config" / "melhores_configuracoes.json").read_text(encoding="utf-8"))
    base_score = float(cfg_json.get("queimada", {}).get("score_medio", float("-inf")))
    eq_names = obter_instancias_equilibradas()

    bateria_dir = _sim_path / "bateria_testes"
    seen_names = set()
    inst_paths = []
    for p in sorted(bateria_dir.rglob("*"), key=lambda x: (len(x.parts), str(x))):
        if p.is_file() and p.name in eq_names and p.name not in seen_names:
            seen_names.add(p.name)
            inst_paths.append(str(p))

    print(f"Instancias equilibradas encontradas: {len(inst_paths)}")
    combos = list(
        itertools.product(
            [False, True],  # usar_pack_iterative
            [False, True],  # usar_delaunay_2opt
            ["nenhuma", "pgch_coco", "insertion_bitflip", "boundary_bitflip"],  # pos_otimizacao
        )
    )

    configs = []
    for pack, del2, pos in combos:
        p = dict(base_params)
        p["usar_pack_iterative"] = pack
        p["usar_delaunay_2opt"] = del2
        p["pos_otimizacao"] = pos
        p["time_limit_pos"] = 5.0
        configs.append(((pack, del2, pos), p))

    tasks = []
    for idx, (_, p) in enumerate(configs):
        for path_str in inst_paths:
            tasks.append((idx, p, path_str))

    scores = {i: [] for i in range(len(configs))}
    times = {i: [] for i in range(len(configs))}
    t_start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as ex:
        for idx, lucro, dt in ex.map(_eval_one, tasks, chunksize=4):
            scores[idx].append(lucro)
            times[idx].append(dt)

    elapsed = time.perf_counter() - t_start
    print(f"Concluido em {elapsed:.2f}s\n")

    results = []
    for idx, ((pack, del2, pos), p) in enumerate(configs):
        mean_g = sum(scores[idx]) / len(scores[idx])
        mean_t = sum(times[idx]) / len(times[idx])
        results.append((mean_g, mean_t, pack, del2, pos, p))

    results.sort(key=lambda x: x[0], reverse=True)
    print(f"{'Rank':<4} | {'PackIter':<8} | {'Delaunay':<8} | {'PosOpt':<18} | {'Lucro Medio (G)':>16} | {'Tempo (s)':>9}")
    print("-" * 80)
    for r, (mg, mt, pack, del2, pos, _) in enumerate(results, 1):
        print(f"{r:<4} | {str(pack):<8} | {str(del2):<8} | {pos:<18} | {mg:16.2f} | {mt:9.3f}")

    best_g, best_t, best_pack, best_del2, best_pos, best_p = results[0]
    print(f"\nScore base anterior: {base_score:.2f} | Melhor hibrido: {best_g:.2f}")
    if best_g > base_score:
        salvar_melhor_configuracao(
            "queimada",
            best_p,
            best_g,
            origem="grid_search_hibrido_campeao",
            detalhes={
                "total_instancias": len(inst_paths),
                "espaco": "hibrido_sobre_campeao",
                "tempo_medio_s": best_t,
            },
        )
        print("Nova configuracao campea 'queimada' salva com sucesso!")


if __name__ == "__main__":
    main()
