"""
Combinatorial lower bounds independent of the MILP (D7).

SRPT fluid relaxation: drop cache, chunking, and per-request rate caps ->
single preemptive machine of speed B tokens/bucket, job work w_i = p_i + L_i,
release a_i. SRPT is exactly optimal for 1|r_j, pmtn|sum C_j, and the relaxed
optimum lower-bounds our bucketed objective. Runtime: O(n log n) events.
Final per-slice bound = max(milp_dual_bound, srpt_bound).
"""
import heapq


def srpt_bound(inst):
    """Return fluid SRPT lower bound on total latency (bucket units)."""
    B = float(inst.bucket_budget)
    jobs = sorted(((r.arrival, r.prompt + r.output, r.rid) for r in inst.requests))
    total_flow = 0.0
    t = 0.0
    i = 0
    heap = []          # (remaining_work, rid, arrival)
    n = len(jobs)
    while i < n or heap:
        if not heap:
            t = max(t, jobs[i][0])
        # next arrival time
        t_next = jobs[i][0] if i < n else float("inf")
        while i < n and jobs[i][0] <= t:
            a, w, rid = jobs[i]
            heapq.heappush(heap, (float(w), rid, float(a)))
            i += 1
            t_next = jobs[i][0] if i < n else float("inf")
        if not heap:
            continue
        rem, rid, a = heapq.heappop(heap)
        finish = t + rem / B
        if finish <= t_next:            # completes before next arrival
            t = finish
            total_flow += t - a
        else:                            # preempted at next arrival
            rem -= (t_next - t) * B
            t = t_next
            heapq.heappush(heap, (rem, rid, a))
    return total_flow
