"""
Discretization (Delta) sensitivity for the appendix (Option 2).

Re-runs the 6 pilot slices at Delta in {4, 16} (Delta=8 = existing pilot),
holding PER-ITERATION capacity fixed: bucket_budget = 500*Delta tokens,
prefill_chunk = 256*Delta tokens, cache unchanged (time-independent).
Comparison metric: proven FCFS/SJF gaps (unit-free ratios).

Run:      python local_script/run_delta_sens.py            (resumable)
Analyze:  python local_script/run_delta_sens.py --analyze  (needs pilot rows
          in results/results.csv; writes appendix_delta_table.tex)

Expected runtime: Delta=16 fast (~5-10 min/slice), Delta=4 slow
(model ~2x pilot; ~30-45 min/slice). Total ~4-6 h. Shardable: --shard 0/2, 1/2.
"""
import argparse
from pathlib import Path
import pandas as pd

from run_slices import make_specs, run_one, done_ids_glob, parse_shard
from loader import SliceSpec

ROOT = Path(__file__).resolve().parent.parent
DELTA_CSV = str(ROOT / "results" / "results_delta.csv")

PER_ITER_BUDGET = 500     # tokens per iteration (= 4000/8, pilot calibration)
PER_ITER_PREFILL = 256    # prefill tokens per iteration (= 2048/8)
DELTAS = [4, 16]


def delta_specs():
    out = []
    for sid, spec in make_specs(pilot=True):
        for d in DELTAS:
            out.append((f"{sid}_d{d}", SliceSpec(
                service=spec.service, start_ms=spec.start_ms,
                n_requests=spec.n_requests, delta=d,
                prefill_chunk=PER_ITER_PREFILL * d,
                bucket_budget=PER_ITER_BUDGET * d,
                cache_cap=spec.cache_cap, iter_ms=spec.iter_ms)))
    return out


def main(time_limit=900, mip_gap=0.01, shard=None):
    shard_i, shard_n = parse_shard(shard)
    csv = DELTA_CSV if shard_n == 1 else str(ROOT / "results" / f"results_delta_s{shard_i}.csv")
    specs = [sp for k, sp in enumerate(delta_specs()) if k % shard_n == shard_i]
    skip = done_ids_glob(str(ROOT / "results" / "results_delta*.csv"))
    print(f"delta grid: {len(specs)} runs, done {len(skip)}, TL={time_limit}s", flush=True)
    for sid, spec in specs:
        if sid in skip:
            print(f"[skip] {sid}", flush=True)
            continue
        run_one(sid, spec, time_limit, mip_gap, csv)


def analyze():
    import glob
    base = pd.read_csv(ROOT / "results" / "results.csv")
    base = base[base.n == 200].copy()
    base["delta_run"] = 8
    frames = [pd.read_csv(p) for p in glob.glob(str(ROOT / "results" / "results_delta*.csv"))]
    assert frames, "results_delta*.csv 없음 — 먼저 실행하세요"
    dd = pd.concat(frames, ignore_index=True)
    dd["delta_run"] = dd.slice_id.str.extract(r"_d(\d+)$").astype(int)
    dd["slice_id"] = dd.slice_id.str.replace(r"_d\d+$", "", regex=True)
    df = pd.concat([base, dd], ignore_index=True)
    df["fcfs_gap_lo"] = (df.fcfs_lat - df.milp_incumbent) / df.milp_incumbent
    piv = df.pivot_table(index="slice_id", columns="delta_run",
                         values="fcfs_gap_lo").sort_index()
    piv.columns = [f"d{c}" for c in piv.columns]
    print("\n=== proven FCFS gap by Delta ===")
    print((piv * 100).round(1).to_string())
    sp = piv.corr(method="spearman")
    print("\nSpearman rank corr:\n", sp.round(3).to_string())

    rows = []
    for sid, r in piv.iterrows():
        short = sid.replace("conv_", "").replace("_n200", "").replace("_", "\\_")
        rows.append(f"{short} & {r.get('d4', float('nan'))*100:.1f} & "
                    f"{r.get('d8', float('nan'))*100:.1f} & {r.get('d16', float('nan'))*100:.1f} \\\\")
    tex = ("\\begin{tabular}{lccc}\n\\toprule\n"
           "slice & $\\Delta{=}4$ & $\\Delta{=}8$ & $\\Delta{=}16$ \\\\\n\\midrule\n"
           + "\n".join(rows) +
           "\n\\bottomrule\n\\end{tabular}\n")
    out = ROOT / "results" / "appendix_delta_table.tex"
    out.write_text(tex)
    print("\nLaTeX 표 저장:", out)
    return piv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--tl", type=int, default=900)
    ap.add_argument("--shard", type=str, default=None)
    a = ap.parse_args()
    if a.analyze:
        analyze()
    else:
        main(time_limit=a.tl, shard=a.shard)
