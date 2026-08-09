"""
O-pred layer verification.
Invariant 1 (perfect prediction): O-pred plan executed under truth reproduces
the O-true optimum exactly (same total latency, zero violations).
Invariant 2 (degraded prediction): executed latency is feasible and >= O-true.
"""
from formulation import Request, Instance, build_and_solve
from simulator import run_policy, replay_schedule, execute_plan_order

reqs = [
    Request(0, 0, 5, 8), Request(1, 0, 3, 2), Request(2, 0, 3, 2),
    Request(3, 1, 3, 2), Request(4, 1, 3, 2),
]
inst_true = Instance(requests=reqs, cache_cap=14, bucket_budget=8, delta=1, prefill_chunk=3)

res_true = build_and_solve(inst_true)
A = res_true["objective"]
print(f"[A: O-true] optimum = {A}")

def opred_value(pred_map):
    inst_pred = Instance(
        requests=[Request(r.rid, r.arrival, r.prompt, pred_map[r.rid]) for r in reqs],
        cache_cap=14, bucket_budget=8, delta=1, prefill_chunk=3)
    res = build_and_solve(inst_pred)
    order = sorted(res["admit"], key=lambda i: (res["admit"][i], i))
    admit = execute_plan_order(inst_true, order)
    lat, ttft, viol = replay_schedule(inst_true, admit)
    assert not viol, viol
    return lat

# Invariant 1: perfect predictions
B_perfect = opred_value({r.rid: r.output for r in reqs})
assert abs(B_perfect - A) < 1e-6, (B_perfect, A)
print(f"[B: O-pred, perfect] = {B_perfect}  == A  (invariant 1 OK)")

# Invariant 2: everyone predicted as median length (long job disguised as short)
B_median = opred_value({r.rid: 2 for r in reqs})
assert B_median >= A - 1e-6
print(f"[B: O-pred, all-median] = {B_median}  >= A  (invariant 2 OK, info gap = {B_median - A})")

# ladder sanity: C_pred (SJF with same wrong predictions) vs B
a_sjf = run_policy(inst_true, "sjf", pred={r.rid: 2 for r in reqs})
c_lat, _, v = replay_schedule(inst_true, a_sjf); assert not v
print(f"[C: SJF-pred, all-median] = {c_lat}")
print("ladder:", f"A={A} <= B={B_median} ; C_pred={c_lat}")
