"""
One-time trace preparation (D2). Re-run after any environment reset.

Downloads Azure LLM Inference 2024 traces (CC-BY-4.0) and converts to sorted
parquet (t_ms, p, L). Cite: Stojkovic et al., "DynamoLLM", HPCA 2025.
Usage: python3 prep_trace.py   (creates ./traces/{conv,code}.parquet, ~2GB temp CSV)
"""
import os, subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT / "traces"

URL = "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_{svc}_1week.csv"
os.makedirs(TRACES, exist_ok=True)

for svc in ["conv", "code"]:
    csv = str(TRACES / f"azure_llm_2024_{svc}.csv")
    if not os.path.exists(csv):
        print(f"downloading {svc} ...")
        subprocess.run(["curl", "-sL", "-o", csv, URL.format(svc=svc)], check=True)
    parts = []
    for chunk in pd.read_csv(csv, chunksize=3_000_000,
                             dtype={"TIMESTAMP": "str", "ContextTokens": "int32",
                                    "GeneratedTokens": "int32"}):
        ts = pd.to_datetime(chunk.TIMESTAMP, utc=True, format="mixed", errors="coerce")
        parts.append(pd.DataFrame({"ts": ts, "p": chunk.ContextTokens,
                                   "L": chunk.GeneratedTokens}).dropna())
    df = pd.concat(parts, ignore_index=True).sort_values("ts").reset_index(drop=True)
    out = pd.DataFrame({
        "t_ms": ((df.ts - df.ts.iloc[0]).dt.total_seconds() * 1000).astype("int64"),
        "p": df.p.astype("int32"),
        "L": df.L.clip(lower=1).astype("int32"),
    })
    out.to_parquet(TRACES / f"{svc}.parquet", index=False)
    print(svc, len(out), "rows ->", str(TRACES / f"{svc}.parquet"))
