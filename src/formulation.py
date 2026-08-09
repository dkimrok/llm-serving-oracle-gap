"""
Oracle MILP for LLM serving scheduling — v0.2 (chunked prefill, design doc §3+§3.1).

Time-bucketed, no-preemption, fixed-chunk-prefill oracle.
Conservative on two counts (no preemption; fixed chunking restricts flexible systems),
so measured heuristic gaps are LOWER BOUNDS on true gaps.

Model (admission bucket s, elapsed phi = t - s):
  prefill phase phi in [0, pi_i):   tokens min(B_pf, p_i - phi*B_pf),
                                    cache  min((phi+1)*B_pf, p_i)
  decode  phase phi in [pi_i, w_i): tokens min(Delta, L_i - (phi-pi_i)*Delta),
                                    cache  p_i + min((phi-pi_i+1)*Delta, L_i)
  pi_i = ceil(p_i / B_pf), d_i = ceil(L_i / Delta), w_i = pi_i + d_i
  completion C_i = s + w_i,  TTFT_i = s + pi_i - a_i  (first token at end of prefill)
All coefficients are precomputable -> constraints stay linear in the single
admission variable z[i,s]; model size identical in shape to v0.1.
"""
from dataclasses import dataclass
from math import ceil
import pulp


@dataclass(frozen=True)
class Request:
    rid: int
    arrival: int      # arrival bucket a_i
    prompt: int       # p_i (tokens)
    output: int       # L_i (tokens); oracle uses truth, O-pred passes prediction

    def d(self, delta: int) -> int:
        return ceil(self.output / delta)

    def pi(self, prefill_chunk: int) -> int:
        return ceil(self.prompt / prefill_chunk)

    def w(self, delta: int, prefill_chunk: int) -> int:
        return self.pi(prefill_chunk) + self.d(delta)


@dataclass
class Instance:
    requests: list          # list[Request]
    cache_cap: int          # M (tokens)
    bucket_budget: int      # B*Delta (tokens processable per bucket, shared pf+decode)
    delta: int              # decode tokens per bucket per active request
    prefill_chunk: int = 2048   # B_pf: prefill tokens per bucket per request
    horizon: int = None
    admit_window: int = None

    def T(self) -> int:
        if self.horizon is not None:
            return self.horizon
        last_arr = max(r.arrival for r in self.requests)
        total_work = sum(r.w(self.delta, self.prefill_chunk) for r in self.requests)
        return last_arr + total_work + 2


def cache_coeff(r: Request, s: int, t: int, delta: int, prefill_chunk: int) -> int:
    """KV-cache tokens occupied by r during bucket t if admitted at s (0 if inactive)."""
    phi = t - s
    pi_i = r.pi(prefill_chunk)
    if phi < 0 or phi >= pi_i + r.d(delta):
        return 0
    if phi < pi_i:                                  # prefill phase
        return min((phi + 1) * prefill_chunk, r.prompt)
    return r.prompt + min((phi - pi_i + 1) * delta, r.output)   # decode phase


def tok_coeff(r: Request, s: int, t: int, delta: int, prefill_chunk: int) -> int:
    """Budget tokens consumed by r during bucket t if admitted at s (prefill or decode)."""
    phi = t - s
    pi_i = r.pi(prefill_chunk)
    if phi < 0 or phi >= pi_i + r.d(delta):
        return 0
    if phi < pi_i:
        return min(prefill_chunk, r.prompt - phi * prefill_chunk)
    return min(delta, r.output - (phi - pi_i) * delta)


def build_and_solve(inst: Instance, msg: bool = False, time_limit: int = 300,
                    mip_gap: float = 0.0, warm_admit: dict = None):
    """warm_admit: optional rid->admit_bucket feasible plan used as MIP start."""
    T = inst.T()
    reqs = inst.requests
    W = inst.admit_window
    D, PF = inst.delta, inst.prefill_chunk

    prob = pulp.LpProblem("oracle_gap", pulp.LpMinimize)
    z = {}
    for r in reqs:
        t_hi = T - r.w(D, PF)
        if W is not None:
            t_hi = min(t_hi, r.arrival + W)
        for t in range(r.arrival, t_hi + 1):
            z[r.rid, t] = pulp.LpVariable(f"z_{r.rid}_{t}", cat="Binary")

    for r in reqs:
        prob += pulp.lpSum(z[i, t] for (i, t) in z if i == r.rid) == 1, f"admit_{r.rid}"

    # bucket-wise accumulators, O(|z| * avg w)
    by_rid = {r.rid: r for r in reqs}
    cache_acc = [[] for _ in range(T)]
    tok_acc = [[] for _ in range(T)]
    for (i, s), var in z.items():
        r = by_rid[i]
        for t in range(s, min(s + r.w(D, PF), T)):
            cache_acc[t].append(cache_coeff(r, s, t, D, PF) * var)
            tok_acc[t].append(tok_coeff(r, s, t, D, PF) * var)
    for t in range(T):
        if cache_acc[t]:
            prob += pulp.lpSum(cache_acc[t]) <= inst.cache_cap, f"cache_{t}"
        if tok_acc[t]:
            prob += pulp.lpSum(tok_acc[t]) <= inst.bucket_budget, f"budget_{t}"

    # objective F1: sum of latencies
    prob += pulp.lpSum(
        (s + by_rid[i].w(D, PF) - by_rid[i].arrival) * var for (i, s), var in z.items()
    )

    # solve via highspy directly (reliable status / dual bound / warm start)
    import highspy, tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".mps", delete=False)
    tmp.close()
    prob.writeMPS(tmp.name)
    h = highspy.Highs()
    h.setOptionValue("output_flag", msg)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", float(mip_gap))
    h.readModel(tmp.name)

    if warm_admit is not None:
        lp = h.getLp()
        name_to_col = {lp.col_names_[j]: j for j in range(lp.num_col_)}
        vals = [0.0] * lp.num_col_
        ok = True
        for (i, s), var in z.items():
            if warm_admit.get(i) == s:
                j = name_to_col.get(var.name)
                if j is None:
                    ok = False
                    break
                vals[j] = 1.0
        for i in warm_admit:
            if not any(k == i and warm_admit[i] == s for (k, s) in z):
                ok = False   # warm plan admits outside variable domain (e.g., window)
        if ok:
            sol = highspy.HighsSolution()
            sol.col_value = vals
            h.setSolution(sol)
        warm_applied = ok
    else:
        warm_applied = None

    h.run()
    info = h.getInfo()
    status = h.modelStatusToString(h.getModelStatus())
    sol = h.getSolution()
    lp = h.getLp()
    vals = dict(zip([lp.col_names_[j] for j in range(lp.num_col_)], sol.col_value))
    os.unlink(tmp.name)

    admit = {}
    for (i, s), var in z.items():
        if vals.get(var.name, 0.0) > 0.5:
            admit[i] = s
    return {
        "status": status,
        "objective": info.objective_function_value,
        "dual_bound": info.mip_dual_bound,
        "mip_gap": info.mip_gap,
        "admit": admit,
        "num_bin_vars": len(z),
        "T": T,
        "warm_applied": warm_applied,
    }
