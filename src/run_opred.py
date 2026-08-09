"""
O-pred layer batch runner (design doc §4, layer B + C_pred).

For each slice of the SAME grid as run_slices:
  1. fit causal predictor on trace history strictly before the window
  2. plan: MILP on the PREDICTED instance (warm start = SJF-pred plan)
  3. execute: keep the plan's admission order under TRUE lengths
     (earliest-feasible list scheduling = repair rule)
  4. also execute SJF-pred (heuristic with the same information) -> C_pred

Ledger: results/results_opred.csv (append per slice, resumable).
Does NOT need results/results.csv — the A-layer merge happens at analysis.

Notebook:  import run_opred; run_opred.main(pilot=False, time_limit=900)
CLI:       python local_script/run_opred.py --main --tl 900
"""
import gc, os, time, csv, argparse
from pathlib import Path
import numpy as np
import loader
from loader import load_slice
from simulator import run_policy, replay_schedule, execute_plan_order
from formulation import Request, Instance, build_and_solve
from predictor import fit_causal
from run_slices import make_specs, done_ids

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OPRED = str(ROOT / "results" / "results_opred.csv")

FIELDS = ["slice_id", "service", "start_ms", "n",
          "opred_lat", "opred_ttft", "sjfpred_lat", "sjfpred_ttft",
          "plan_obj", "plan_bound", "plan_status",
          "pred_mae", "pred_bias", "solve_s", "warm_applied"]


def append_row(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def run_one(sid, spec, time_limit, mip_gap, results_csv):
    t0 = time.time()
    inst_true = load_slice(spec)
    pred = fit_causal(spec.service, before_ms=spec.start_ms)
    loader._cache.clear(); gc.collect()

    pred_map = {r.rid: max(1, pred.predict(r.prompt)) for r in inst_true.requests}
    err = np.array([pred_map[r.rid] - r.output for r in inst_true.requests])
    inst_pred = Instance(
        requests=[Request(r.rid, r.arrival, r.prompt, pred_map[r.rid])
                  for r in inst_true.requests],
        cache_cap=spec.cache_cap, bucket_budget=spec.bucket_budget,
        delta=spec.delta, prefill_chunk=spec.prefill_chunk)

    # C_pred: SJF ordered by predicted lengths, executed under truth
    a_sjfp = run_policy(inst_true, "sjf", pred=pred_map)
    c_lat, c_ttft, v = replay_schedule(inst_true, a_sjfp)
    assert not v, (sid, "sjf_pred", v[:3])

    # B: plan on predicted instance (warm start = SJF-pred plan applied to it),
    #    then execute the plan's order under truth
    by_rid = {r.rid: r for r in inst_pred.requests}
    max_wait = max(a_sjfp[i] - by_rid[i].arrival for i in a_sjfp)
    inst_pred.admit_window = max(150, max_wait + 10)
    res = build_and_solve(inst_pred, warm_admit=a_sjfp,
                          time_limit=time_limit, mip_gap=mip_gap)
    order = sorted(res["admit"], key=lambda i: (res["admit"][i], i))
    admit_b = execute_plan_order(inst_true, order)
    b_lat, b_ttft, v = replay_schedule(inst_true, admit_b)
    assert not v, (sid, "opred_exec", v[:3])

    append_row(results_csv, {
        "slice_id": sid, "service": spec.service, "start_ms": spec.start_ms,
        "n": spec.n_requests,
        "opred_lat": b_lat, "opred_ttft": b_ttft,
        "sjfpred_lat": c_lat, "sjfpred_ttft": c_ttft,
        "plan_obj": res["objective"], "plan_bound": res["dual_bound"],
        "plan_status": res["status"],
        "pred_mae": round(float(np.abs(err).mean()), 1),
        "pred_bias": round(float(err.mean()), 1),
        "solve_s": round(time.time() - t0, 1),
        "warm_applied": res["warm_applied"],
    })
    print(f"[done] {sid}: B(opred)={b_lat} C(sjf_pred)={c_lat} "
          f"mae={np.abs(err).mean():.0f} t={time.time()-t0:.0f}s", flush=True)


def main(pilot=True, time_limit=None, mip_gap=0.01, n_req=None,
         results_csv=None):
    results_csv = results_csv or DEFAULT_OPRED
    time_limit = time_limit or (120 if pilot else 900)
    specs = make_specs(pilot=pilot, n_req=n_req)
    skip = done_ids(results_csv)
    print(f"O-pred grid: {len(specs)} slices ({'PILOT' if pilot else 'MAIN'}), "
          f"done: {len(skip)}, TL={time_limit}s", flush=True)
    for sid, spec in specs:
        if sid in skip:
            print(f"[skip] {sid}", flush=True)
            continue
        run_one(sid, spec, time_limit, mip_gap, results_csv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--pilot", action="store_true", default=True)
    g.add_argument("--main", dest="pilot", action="store_false")
    ap.add_argument("--tl", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    main(pilot=a.pilot, time_limit=a.tl, n_req=a.n)
