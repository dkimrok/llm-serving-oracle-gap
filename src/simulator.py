"""
Bucket-based replay simulator — v0.2 (chunked prefill).
Coefficient functions imported from formulation.py = single source of truth,
so simulator feasibility == MILP feasibility by construction.

Policies: fcfs / sjf (predicted-length order via `pred`), clairvoyant-feasible
greedy admission, no preemption. TODO v0.3: preemptive vLLM-like variant.
replay_schedule() validates any admission plan and computes latency/TTFT.
"""
from formulation import Request, Instance, cache_coeff, tok_coeff


def _feasible_to_admit(cand, s, active, inst):
    D, PF = inst.delta, inst.prefill_chunk
    end = s + cand.w(D, PF)
    horizon = max([end] + [ss + r.w(D, PF) for r, ss in active]) if active else end
    for t in range(s, horizon):
        cache = cache_coeff(cand, s, t, D, PF) + sum(
            cache_coeff(r, ss, t, D, PF) for r, ss in active)
        if cache > inst.cache_cap:
            return False
        tok = tok_coeff(cand, s, t, D, PF) + sum(
            tok_coeff(r, ss, t, D, PF) for r, ss in active)
        if tok > inst.bucket_budget:
            return False
    return True


def run_policy(inst: Instance, policy: str = "fcfs", pred=None):
    D, PF = inst.delta, inst.prefill_chunk
    reqs = sorted(inst.requests, key=lambda r: (r.arrival, r.rid))
    T = inst.T() * 4
    queue, active, admit = [], [], {}
    ptr = 0
    for t in range(T):
        while ptr < len(reqs) and reqs[ptr].arrival <= t:
            queue.append(reqs[ptr]); ptr += 1
        active = [(r, s) for r, s in active if t < s + r.w(D, PF)]
        if policy == "sjf":
            key = (lambda r: (pred or {}).get(r.rid, r.output))
            queue.sort(key=lambda r: (key(r), r.arrival, r.rid))
        admitted = True
        while admitted:
            admitted = False
            for q in list(queue):
                if _feasible_to_admit(q, t, active, inst):
                    active.append((q, t)); queue.remove(q)
                    admit[q.rid] = t; admitted = True
                    break
        if ptr >= len(reqs) and not queue and not active:
            break
    missing = [r.rid for r in reqs if r.rid not in admit]
    assert not missing, f"requests never admitted: {missing[:5]} (model/params issue)"
    return admit


def replay_schedule(inst: Instance, admit: dict):
    """Validate plan; return (total_latency, total_ttft, violations)."""
    D, PF = inst.delta, inst.prefill_chunk
    T = max(admit[r.rid] + r.w(D, PF) for r in inst.requests)
    violations = []
    for t in range(T):
        cache = sum(cache_coeff(r, admit[r.rid], t, D, PF) for r in inst.requests)
        tok = sum(tok_coeff(r, admit[r.rid], t, D, PF) for r in inst.requests)
        if cache > inst.cache_cap:
            violations.append(("cache", t, cache))
        if tok > inst.bucket_budget:
            violations.append(("budget", t, tok))
    for r in inst.requests:
        if admit[r.rid] < r.arrival:
            violations.append(("arrival", r.rid, admit[r.rid]))
    total_lat = sum(admit[r.rid] + r.w(D, PF) - r.arrival for r in inst.requests)
    total_ttft = sum(admit[r.rid] + r.pi(PF) - r.arrival for r in inst.requests)
    return total_lat, total_ttft, violations


def execute_plan_order(inst: Instance, order):
    """Admit requests in the given rid order (strict), each at the earliest
    feasible bucket >= its arrival, under TRUE lengths in `inst`.
    This is the O-pred repair rule: keep the plan's admission order, delay
    admissions as needed for feasibility under reality."""
    D, PF = inst.delta, inst.prefill_chunk
    by_rid = {r.rid: r for r in inst.requests}
    seq = [by_rid[i] for i in order]
    T = inst.T() * 4
    active, admit = [], {}
    k = 0                                    # next request in sequence to admit
    for t in range(T):
        active = [(r, s) for r, s in active if t < s + r.w(D, PF)]
        while k < len(seq):
            head = seq[k]
            if head.arrival > t:
                break
            if _feasible_to_admit(head, t, active, inst):
                active.append((head, t)); admit[head.rid] = t; k += 1
            else:
                break
        if k >= len(seq) and not active:
            break
    assert len(admit) == len(seq), f"plan execution incomplete: {len(admit)}/{len(seq)}"
    return admit
