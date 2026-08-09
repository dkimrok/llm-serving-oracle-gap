"""
Paper figures (D7). Inputs: results/merged_with_srpt.csv (main, polish 병합),
results.csv의 파일럿 행, results_opred*.csv. Outputs: figures/fig{1,2,3}.png+pdf.
"""
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
R, F = ROOT/"results", ROOT/"figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

m = pd.read_csv(R/"merged_with_srpt.csv")
m["rank"] = m.slice_id.str.extract(r"_r([\d.]+)_").astype(float)
m["gap_lo"] = (m.fcfs_lat - m.milp_incumbent)/m.milp_incumbent
m["gap_hi"] = (m.fcfs_lat - m.milp_bound)/m.milp_bound

# Fig1: 서비스별 FCFS 증명 갭 분포 (구간 표시)
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for k, (svc, c) in enumerate([("conv", "#d1495b"), ("code", "#00798c")]):
    g = m[m.service == svc].sort_values("gap_lo").reset_index(drop=True)
    x = np.arange(len(g)) + k*27
    ax.vlines(x, g.gap_lo*100, g.gap_hi*100, color=c, alpha=.35, lw=1.2)
    ax.scatter(x, g.gap_lo*100, s=10, color=c, label=f"{svc} (proven)", zorder=3)
ax.axhline(0, color="gray", lw=.6)
ax.set_ylabel("FCFS optimality gap (%)"); ax.set_xlabel("slices (sorted within service)")
ax.set_xticks([]); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(F/"fig1_gap_distribution.png", dpi=300); fig.savefig(F/"fig1_gap_distribution.pdf")

# Fig2: 파일럿 4단 사다리 분해 (A 대비 %)
pilot = pd.read_csv(R/"results.csv") if (R/"results.csv").exists() else None
op_files = sorted(R.glob("results_opred*.csv"))
op = pd.concat([pd.read_csv(p) for p in op_files]).drop_duplicates("slice_id", keep="last")
base = pd.read_csv("/mnt/user-data/uploads/results.csv") if pilot is None else pilot
base = base[base.n == 200].set_index("slice_id")
op = op.set_index("slice_id").join(base[["milp_incumbent","sjf_lat","fcfs_lat"]])
op["info"] = (op.opred_lat - op.milp_incumbent)/op.milp_incumbent*100
op["algo"] = (op.sjfpred_lat - op.opred_lat)/op.milp_incumbent*100
op = op.sort_index()
fig, ax = plt.subplots(figsize=(3.4, 2.6))
x = np.arange(len(op))
ax.bar(x, op["info"], .6, label="information gap (A→B)", color="#edae49")
ax.bar(x, op["algo"], .6, bottom=op["info"], label="algorithmic gap (B→C)", color="#00798c")
ax.scatter(x, (op.fcfs_lat - op.milp_incumbent)/op.milp_incumbent*100, marker="_",
           s=200, color="#d1495b", label="FCFS total gap")
ax.set_xticks(x); ax.set_xticklabels([s.replace("conv_","").replace("_n200","") for s in op.index],
                                     rotation=45, ha="right", fontsize=7)
ax.set_ylabel("% above full-information oracle"); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(F/"fig2_ladder.png", dpi=300); fig.savefig(F/"fig2_ladder.pdf")

# Fig3: 부하-갭 (서비스별, 클러스터 평균 ± 슬라이스 산점)
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for svc, c in [("conv", "#d1495b"), ("code", "#00798c")]:
    g = m[m.service == svc]
    ax.scatter(g["rank"] + np.random.uniform(-.012, .012, len(g)), g.gap_lo*100,
               s=9, alpha=.45, color=c)
    mu = g.groupby("rank").gap_lo.mean()*100
    ax.plot(mu.index, mu.values, "-o", color=c, ms=4, label=svc)
ax.set_xlabel("load rank of trace hour"); ax.set_ylabel("proven FCFS gap (%)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(F/"fig3_load.png", dpi=300); fig.savefig(F/"fig3_load.pdf")
print("figures saved:", sorted(p.name for p in F.glob("*.png")))
