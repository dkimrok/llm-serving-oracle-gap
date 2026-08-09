"""
Output-length predictor (design doc §4, O-pred layer).

Causal binned-median: for each evaluation slice, fit ONLY on trace rows with
t_ms < window start (no future leakage). Prompt-length quantile bins ->
median generated length per bin. Deliberately simple: this is the *realistic
information* baseline, not a modeling contribution.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT / "traces"


class BinnedMedianPredictor:
    def __init__(self, edges, medians, global_median):
        self.edges = edges              # bin edges over prompt length
        self.medians = medians          # median L per bin
        self.global_median = global_median

    def predict(self, prompt: int) -> int:
        j = int(np.searchsorted(self.edges, prompt, side="right")) - 1
        j = min(max(j, 0), len(self.medians) - 1)
        m = self.medians[j]
        return int(m if np.isfinite(m) else self.global_median)

    def stats(self):
        return {"n_bins": len(self.medians), "global_median": self.global_median}


def fit_causal(service: str, before_ms: int, n_bins: int = 10,
               max_rows: int = 2_000_000) -> BinnedMedianPredictor:
    """Fit on the last `max_rows` trace rows strictly before `before_ms`."""
    df = pd.read_parquet(TRACES / f"{service}.parquet")
    past = df[df.t_ms < before_ms]
    assert len(past) > 1000, "not enough causal history before this window"
    past = past.tail(max_rows)
    edges = np.unique(np.quantile(past.p, np.linspace(0, 1, n_bins + 1)))
    bins = np.searchsorted(edges, past.p, side="right") - 1
    bins = np.clip(bins, 0, len(edges) - 2)
    med = past.groupby(bins).L.median()
    medians = np.full(len(edges) - 1, np.nan)
    medians[med.index.values] = med.values
    return BinnedMedianPredictor(edges[:-1], medians, float(past.L.median()))
