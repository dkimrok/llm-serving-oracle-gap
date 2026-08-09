"""
Slice batch runner (D4, local edition) — resumable, memory-safe, parameterized.

Notebook:  import run_slices; run_slices.main(pilot=True, time_limit=900)
CLI:       python run_slices.py --pilot --tl 900
           python run_slices.py --main  --tl 900     (full 50-slice grid)

Results append to results/results.csv after EVERY slice -> safe to interrupt;
already-done slice_ids are skipped on re-run. Memory: the full trace parquet
(~0.7GB RAM) is released before each MILP solve (B&B tree needs the headroom).
"""
import gc, os, time, csv, argparse
from pathlib import Path
import loader

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = str(ROOT / "results" / "results.csv")
from loader import SliceSpec, pick_window_start, load_slice, _load
from simulator import run_policy, replay_schedule
from formulation import build_and_solve

FIELDS = ["slice_id", "service", "start_ms", "n", "delta", "prefill_chunk",
          "cache_cap", "bucket_budget", "admit_window",
          "fcfs_lat", "fcfs_ttft", "sjf_lat", "sjf_ttft",
          "milp_incumbent", "milp_bound", "milp_gap", "milp_status",
          "solve_s", "binvars", "warm_applied"]


def make_specs(pilot=True, n_req=None):
    n_req = n_req or (200 if pilot else 400)
    services = ["conv"] if pilot else ["conv", "code"]
    ranks = [0.3, 0.6, 0.9] if pilot else [0.1, 0.3, 0.5, 0.7, 0.9]
    offsets = [0, 20] if pilot else [0, 12, 24, 36, 48]
    specs = []
    for service in services:
        df = _load(service)
        for hour_rank in ranks:
            hour_start = pick_window_start(df, hour_rank)
            for offset_min in offsets:
                sid = f"{service}_r{hour_rank}_o{offset_min}_n{n_req}"
                specs.append((sid, SliceSpec(
                    service=service, start_ms=hour_start + offset_min * 60_000,
                    n_requests=n_req)))
    return specs


def done_ids(results_csv):
    if not os.path.exists(results_csv):
        return set()
    with open(results_csv) as f:
        return {row["slice_id"] for row in csv.DictReader(f)}


def done_ids_glob(pattern):
    """샤드 실행 대비: 패턴에 걸리는 모든 원장의 완료 slice_id 합집합."""
    import glob
    done = set()
    for p in glob.glob(pattern):
        done |= done_ids(p)
    return done


def parse_shard(s):
    """'0/2' -> (0, 2). None -> (0, 1)."""
    if not s:
        return 0, 1
    i, n = s.split("/")
    i, n = int(i), int(n)
    assert 0 <= i < n, "shard는 0/N .. (N-1)/N"
    return i, n


def append_row(results_csv, row):
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    new = not os.path.exists(results_csv)
    with open(results_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def run_one(sid, spec, time_limit, mip_gap, results_csv):
    t0 = time.time()
    inst = load_slice(spec)
    loader._cache.clear(); gc.collect()

    heur = {}
    for pol in ("fcfs", "sjf"):
        a = run_policy(inst, pol)
        lat, ttft, v = replay_schedule(inst, a)
        assert not v, (sid, pol, v[:3])
        heur[pol] = (lat, ttft, a)

    warm = min(heur.values(), key=lambda x: x[0])[2]
    by_rid = {r.rid: r for r in inst.requests}
    max_wait = max(warm[i] - by_rid[i].arrival for i in warm)
    inst.admit_window = max(150, max_wait + 10)

    res = build_and_solve(inst, warm_admit=warm,
                          time_limit=time_limit, mip_gap=mip_gap)
    _, _, viol = replay_schedule(inst, res["admit"])
    assert not viol, (sid, "milp", viol[:3])

    import json
    plans_dir = ROOT / "results" / "plans"
    os.makedirs(plans_dir, exist_ok=True)
    with open(plans_dir / f"{sid}.json", "w") as f:
        json.dump({"admit": res["admit"], "incumbent": res["objective"],
                   "bound": res["dual_bound"], "admit_window": inst.admit_window,
                   "heur_fcfs": heur["fcfs"][0], "heur_sjf": heur["sjf"][0]}, f)

    append_row(results_csv, {
        "slice_id": sid, "service": spec.service, "start_ms": spec.start_ms,
        "n": spec.n_requests, "delta": spec.delta,
        "prefill_chunk": spec.prefill_chunk, "cache_cap": spec.cache_cap,
        "bucket_budget": spec.bucket_budget, "admit_window": inst.admit_window,
        "fcfs_lat": heur["fcfs"][0], "fcfs_ttft": heur["fcfs"][1],
        "sjf_lat": heur["sjf"][0], "sjf_ttft": heur["sjf"][1],
        "milp_incumbent": res["objective"], "milp_bound": res["dual_bound"],
        "milp_gap": res["mip_gap"], "milp_status": res["status"],
        "solve_s": round(time.time() - t0, 1), "binvars": res["num_bin_vars"],
        "warm_applied": res["warm_applied"],
    })
    inc, bnd = res["objective"], res["dual_bound"]
    print(f"[done] {sid}: fcfs_gap=[{(heur['fcfs'][0]-inc)/inc:+.1%},"
          f"{(heur['fcfs'][0]-bnd)/bnd:+.1%}] milp_gap={res['mip_gap']:.1%} "
          f"t={time.time()-t0:.0f}s", flush=True)


def main(pilot=True, time_limit=None, mip_gap=0.01, n_req=None,
         results_csv=None, shard=None):
    shard_i, shard_n = parse_shard(shard)
    if results_csv is None:
        results_csv = DEFAULT_RESULTS if shard_n == 1 else \
            str(ROOT / "results" / f"results_s{shard_i}.csv")
    time_limit = time_limit or (120 if pilot else 900)
    specs = make_specs(pilot=pilot, n_req=n_req)
    specs = [sp for k, sp in enumerate(specs) if k % shard_n == shard_i]
    skip = done_ids_glob(str(ROOT / "results" / "results*.csv"))
    print(f"grid: {len(specs)} slices ({'PILOT' if pilot else 'MAIN'}"
          f"{'' if shard_n == 1 else f', shard {shard_i}/{shard_n}'}), "
          f"done(전 원장): {len(skip)}, TL={time_limit}s", flush=True)
    for sid, spec in specs:
        if sid in skip:
            print(f"[skip] {sid}", flush=True)
            continue
        run_one(sid, spec, time_limit, mip_gap, results_csv)


def polish(gap_threshold=0.10, time_limit=1800, mip_gap=0.01,
           results_csv=None, polish_csv=None, shard=None):
    """milp_gap > threshold 슬라이스만 저장된 발견해에서 이어서 재풀이.
    결과는 results_polish.csv에 append (분석 시 slice_id별 최선값 병합)."""
    import json
    import pandas as pd
    shard_i, shard_n = parse_shard(shard)
    results_csv = results_csv or DEFAULT_RESULTS
    if polish_csv is None:
        polish_csv = str(ROOT / "results" / ("results_polish.csv" if shard_n == 1
                         else f"results_polish_s{shard_i}.csv"))
    import glob as _g
    frames = [pd.read_csv(p) for p in _g.glob(str(ROOT / "results" / "results.csv"))
              + _g.glob(str(ROOT / "results" / "results_s*.csv"))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("slice_id", keep="last")
    done = done_ids_glob(str(ROOT / "results" / "results_polish*.csv"))
    targets = df[df.milp_gap > gap_threshold].reset_index(drop=True)
    targets = targets[targets.index % shard_n == shard_i]
    print(f"polish: {len(targets)} targets (gap>{gap_threshold:.0%}"
          f"{'' if shard_n == 1 else f', shard {shard_i}/{shard_n}'}), "
          f"done {len(done)}, TL={time_limit}s", flush=True)
    for _, row in targets.iterrows():
        sid = row.slice_id
        if sid in done:
            print(f"[skip] {sid}", flush=True); continue
        spec = SliceSpec(service=row.service, start_ms=int(row.start_ms),
                         n_requests=int(row.n), delta=int(row.delta),
                         prefill_chunk=int(row.prefill_chunk),
                         cache_cap=int(row.cache_cap),
                         bucket_budget=int(row.bucket_budget))
        t0 = time.time()
        inst = load_slice(spec)
        loader._cache.clear(); gc.collect()
        plans_dir = ROOT / "results" / "plans"
        os.makedirs(plans_dir, exist_ok=True)
        plan_file = plans_dir / f"{sid}.json"
        if plan_file.exists():
            plan = json.load(open(plan_file))
            warm = {int(k): v for k, v in plan["admit"].items()}
            inst.admit_window = int(plan.get("admit_window", 150))
        else:   # 계획 파일 없으면 휴리스틱으로 재시드
            a = run_policy(inst, "sjf")
            warm = a
            by_rid = {r.rid: r for r in inst.requests}
            inst.admit_window = max(150, max(a[i] - by_rid[i].arrival for i in a) + 10)
        res = build_and_solve(inst, warm_admit=warm,
                              time_limit=time_limit, mip_gap=mip_gap)
        _, _, viol = replay_schedule(inst, res["admit"])
        assert not viol, (sid, "polish", viol[:3])
        with open(plan_file, "w") as f:
            json.dump({"admit": res["admit"], "incumbent": res["objective"],
                       "bound": res["dual_bound"], "admit_window": inst.admit_window,
                       "heur_fcfs": row.fcfs_lat, "heur_sjf": row.sjf_lat}, f)
        append_row(polish_csv, {
            "slice_id": sid, "service": row.service, "start_ms": row.start_ms,
            "n": row.n, "delta": row.delta, "prefill_chunk": row.prefill_chunk,
            "cache_cap": row.cache_cap, "bucket_budget": row.bucket_budget,
            "admit_window": inst.admit_window,
            "fcfs_lat": row.fcfs_lat, "fcfs_ttft": row.fcfs_ttft,
            "sjf_lat": row.sjf_lat, "sjf_ttft": row.sjf_ttft,
            "milp_incumbent": res["objective"], "milp_bound": res["dual_bound"],
            "milp_gap": res["mip_gap"], "milp_status": res["status"],
            "solve_s": round(time.time() - t0, 1), "binvars": res["num_bin_vars"],
            "warm_applied": res["warm_applied"],
        })
        print(f"[polish] {sid}: gap {row.milp_gap:.1%} -> {res['mip_gap']:.1%} "
              f"t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--pilot", action="store_true", default=True)
    g.add_argument("--main", dest="pilot", action="store_false")
    ap.add_argument("--tl", type=int, default=None, help="per-slice MILP seconds")
    ap.add_argument("--n", type=int, default=None, help="requests per slice")
    ap.add_argument("--polish", action="store_true", help="느슨한 슬라이스 재풀이 모드")
    ap.add_argument("--gap-th", type=float, default=0.10)
    ap.add_argument("--shard", type=str, default=None,
                    help="병렬 분담: 예) 터미널1 --shard 0/2, 터미널2 --shard 1/2")
    a = ap.parse_args()
    if a.polish:
        polish(gap_threshold=a.gap_th, time_limit=a.tl or 1800, shard=a.shard)
    else:
        main(pilot=a.pilot, time_limit=a.tl, n_req=a.n, shard=a.shard)
