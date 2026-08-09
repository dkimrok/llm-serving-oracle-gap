"""
v0.2 verification: chunked prefill forces multi-bucket prefill in the toy.
1) brute force == MILP  2) oracle replay 0 violations  3) heuristic gaps
4) warm start sanity (incumbent <= warm plan objective)
"""
from itertools import product
from formulation import Request, Instance, build_and_solve
from simulator import run_policy, replay_schedule

# prompts=4~5, prefill_chunk=3 -> pi_i = 2 for r0 (multi-bucket prefill exercised)
reqs = [
    Request(0, arrival=0, prompt=5, output=8),   # long: pi=2, d=8
    Request(1, arrival=0, prompt=3, output=2),
    Request(2, arrival=0, prompt=3, output=2),
    Request(3, arrival=1, prompt=3, output=2),
    Request(4, arrival=1, prompt=3, output=2),
]
inst = Instance(requests=reqs, cache_cap=14, bucket_budget=8, delta=1, prefill_chunk=3)

W = 10
best, best_admit = None, None
for combo in product(*[range(r.arrival, r.arrival + W + 1) for r in reqs]):
    admit = {r.rid: s for r, s in zip(reqs, combo)}
    lat, _, viol = replay_schedule(inst, admit)
    if not viol and (best is None or lat < best):
        best, best_admit = lat, admit
print(f"[brute force] optimal total latency = {best}, admit = {best_admit}")

res = build_and_solve(inst)
print(f"[MILP] status={res['status']} obj={res['objective']} bound={res['dual_bound']} admit={res['admit']}")
lat_m, _, viol = replay_schedule(inst, res["admit"])
assert not viol, viol
assert abs(res["objective"] - best) < 1e-6, "MILP != brute force!"
print("[check] MILP == brute force, replay violations = 0 ")

for pol in ("fcfs", "sjf"):
    a = run_policy(inst, pol)
    lat, ttft, v = replay_schedule(inst, a)
    assert not v
    print(f"[{pol}] latency={lat} ttft={ttft} gap={(lat-best)/best:+.1%}")

# warm start: feed FCFS plan; incumbent must be <= FCFS objective and == optimum here
a_fcfs = run_policy(inst, "fcfs")
lat_fcfs, _, _ = replay_schedule(inst, a_fcfs)
res_w = build_and_solve(inst, warm_admit=a_fcfs, time_limit=60)
assert res_w["objective"] <= lat_fcfs + 1e-6
assert abs(res_w["objective"] - best) < 1e-6
print(f"[warm start] incumbent={res_w['objective']} (<= fcfs {lat_fcfs}) ")
