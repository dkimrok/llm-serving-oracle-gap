"""
Azure LLM 2024 trace loader (design doc §6, D2).

Trace: Azure/AzurePublicDataset, dataset-llm-2024 (CC-BY-4.0; cite DynamoLLM, HPCA'25).
Parquet schema: t_ms (int64, ms since trace start), p (context tokens), L (generated tokens).

Bucketization: one MILP bucket = `delta` iterations; wall duration = delta * iter_ms.
iter_ms default 25ms is a placeholder — CALIBRATE against a serving stack or cite
measured values before any paper-grade run (design doc §5 sensitivity axis).
"""
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from formulation import Request, Instance

ROOT = Path(__file__).resolve().parent.parent   # gap_bench/ (스크립트 폴더의 부모)
TRACES = ROOT / "traces"


@dataclass
class SliceSpec:
    service: str = "conv"          # "conv" | "code"
    start_ms: int = None           # window start; None -> use hour_rank instead
    hour_rank: float = 0.9         # pick window from this load quantile hour (0=idle, 1=peak)
    n_requests: int = 400
    delta: int = 8                 # tokens per bucket per active request
    prefill_chunk: int = 2048      # B_pf: prefill tokens per bucket per request (v0.2)
    iter_ms: float = 25.0          # wall ms per iteration (PLACEHOLDER — calibrate)
    cache_cap: int = 60_000        # M tokens (e.g., ~A100-80GB class KV budget; scenario knob)
    bucket_budget: int = 4_000     # B*Delta tokens per bucket (scenario knob)
    lambda_scale: float = 1.0      # arrival-interval scaling (RQ3 load knob)


_cache = {}

def _load(service: str) -> pd.DataFrame:
    if service not in _cache:
        _cache[service] = pd.read_parquet(TRACES / f"{service}.parquet")
    return _cache[service]


def pick_window_start(df: pd.DataFrame, hour_rank: float) -> int:
    """Return t_ms of the start of the hour whose request count sits at given quantile."""
    hours = (df.t_ms // 3_600_000).value_counts().sort_values()
    idx = min(int(hour_rank * (len(hours) - 1)), len(hours) - 1)
    hour = hours.index[idx]
    return int(hour * 3_600_000)


def load_slice(spec: SliceSpec) -> Instance:
    df = _load(spec.service)
    start = spec.start_ms if spec.start_ms is not None else pick_window_start(df, spec.hour_rank)
    win = df[df.t_ms >= start].head(spec.n_requests)
    assert len(win) == spec.n_requests, "trace exhausted for this window"

    bucket_ms = spec.delta * spec.iter_ms
    t_rel = (win.t_ms - int(win.t_ms.iloc[0])) * spec.lambda_scale
    reqs = [
        Request(rid=i,
                arrival=int(t // bucket_ms),
                prompt=int(p),
                output=int(L))
        for i, (t, p, L) in enumerate(zip(t_rel, win.p, win.L))
    ]
    return Instance(requests=reqs, cache_cap=spec.cache_cap,
                    bucket_budget=spec.bucket_budget, delta=spec.delta,
                    prefill_chunk=spec.prefill_chunk)


def slice_summary(inst: Instance) -> str:
    import numpy as np
    arr = [r.arrival for r in inst.requests]
    L = np.array([r.output for r in inst.requests])
    p = np.array([r.prompt for r in inst.requests])
    span = max(arr) - min(arr) + 1
    return (f"n={len(inst.requests)} arrival_span={span} buckets "
            f"(~{span * inst.delta} iters), p: mean={p.mean():.0f}, "
            f"L: mean={L.mean():.0f} p90={np.percentile(L, 90):.0f} CV={L.std()/L.mean():.2f}")
