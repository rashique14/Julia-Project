# ===========================================================================
#  Single-tour orienteering with wait-dependent, arrival-dependent reward.
#  Monolithic (cutting-plane) formulation with:
#    - tightened per-arc big-M              (LP-bound tightening)
#    - time-budget knapsack                 (caps nodes per tour; lifts to covers)
#    - earliest-arrival timing bound        (triangle inequality)
#    - 2-cycle elimination                  (kills fractional i<->j)
#    - seeded static reward tangents        (real reward surface at the root)
#    - fractional GSEC connectivity cuts    (max-flow/min-cut user cuts)
#    - lazy tangent refinement              (exact reward at integer nodes)
#    - greedy warm start                    (immediate incumbent to prune with)
#
#  CAVEATS (read before trusting the bound):
#   * big-M is now T_end + t[i,j] (the tightest valid value), not 1000.
#   * The reward h(w,z)=f1(w)*f2(z) is NOT globally concave: near w=0 the
#     Hessian is indefinite (concave region is 1000*w*f2 >= (T_end-z)^2).
#     Tangent cuts are therefore a *heuristic-strength* outer approximation,
#     not a guaranteed-valid one. Seeds are filtered to the concave region,
#     but a tangent taken there can still be violated in the far w~0 corner.
#     If you need a certified optimum, replace these with a true concave
#     envelope of h over the operating box. This applies to the lazy cuts too.
# ===========================================================================

using JuMP, Gurobi
import MathOptInterface as MOI
using Random

Random.seed!(90)

# ------------------------------------------------------------------- instance
const NNODES  = 15               # <- set to 8 for a quick sanity run
const nodes   = 1:NNODES
const nodes_0 = 0:NNODES
const T_end   = 30.0
const U_max   = 100.0

const coords = Dict(i => (10*rand(), 10*rand()) for i in nodes_0)   # RNG order:
const A = [(i,j) for i in nodes_0 for j in nodes_0 if i != j]       #  coords ...
const t = Dict((i,j) => hypot(coords[i][1]-coords[j][1],
                              coords[i][2]-coords[j][2]) for (i,j) in A)
const Q = Dict(i => 5.0 + 10.0*rand() for i in nodes)               #  ... then Q

# Tightest valid big-M for  Z[j] >= Z_bar[i] + t[i,j] - M*(1 - y[i,j]):
# when y=0 the constraint must be slack, i.e. M >= max(Z_bar[i]+t) - min(Z[j])
#        = T_end + t[i,j].
bigM(i,j) = T_end + t[(i,j)]

# =============================================================== build model
model = Model(Gurobi.Optimizer)

@variable(model, y[a in A], Bin)
@variable(model, Z[i in nodes_0]     >= 0)     # arrival
@variable(model, Z_bar[i in nodes_0] >= 0)     # departure
@variable(model, mu[i in nodes]      >= 0)     # collected reward (linearized h)

@objective(model, Max, sum(Q[i]*mu[i] for i in nodes))

# ---- routing --------------------------------------------------------------
@constraint(model, depot_out, sum(y[(0,j)] for j in nodes) == 1)
@constraint(model, depot_in,  sum(y[(i,0)] for i in nodes) == 1)
@constraint(model, flow[i in nodes],
    sum(y[(i,j)] for j in nodes_0 if i != j) ==
    sum(y[(j,i)] for j in nodes_0 if j != i))
@constraint(model, one_out[i in nodes],
    sum(y[(i,j)] for j in nodes_0 if i != j) <= 1)

# ---- timing (tightened big-M) --------------------------------------------
@constraint(model, Z_bar[0] == 0)
@constraint(model, Z[0] <= U_max)
@constraint(model, MTZ[(i,j) in A; j != 0],
    Z[j] >= Z_bar[i] + t[(i,j)] - bigM(i,j)*(1 - y[(i,j)]))
@constraint(model, MTZ_ret[i in nodes],
    Z[0] >= Z_bar[i] + t[(i,0)] - bigM(i,0)*(1 - y[(i,0)]))
@constraint(model, seq[i in nodes], Z[i] <= Z_bar[i])
@constraint(model, depart_bound[i in nodes], Z_bar[i] <= T_end)
@constraint(model, anchor_Z[i in nodes],
    Z[i] >= T_end*(1 - sum(y[(j,i)] for j in nodes_0 if j != i)))
@constraint(model, max_wait[i in nodes], Z_bar[i] - Z[i] <= 20)
@constraint(model, no_expiration[i in nodes], Z[i] <= T_end)

# ---- reward linking -------------------------------------------------------
# f1 = w/(1+w) with w <= 20  =>  f1 <= 20/21;  f2 <= 1  =>  h <= 20/21.
@constraint(model, mu_cap[i in nodes], mu[i] <= 20/21)
@constraint(model, mu_visit[i in nodes],
    mu[i] <= sum(y[(j,i)] for j in nodes_0 if j != i))

# ---- (A) time-budget knapsack --------------------------------------------
# Every visited arrival is >= accumulated non-return travel and <= T_end,
# so the total non-return travel of any tour is capped by the horizon.
@constraint(model, time_budget,
    sum(t[(i,j)]*y[(i,j)] for (i,j) in A if j != 0) <= T_end)

# ---- (B) earliest-arrival bound (triangle inequality) --------------------
@constraint(model, arr_lb[i in nodes],
    Z[i] >= t[(0,i)]*sum(y[(j,i)] for j in nodes_0 if j != i))

# ---- (C) 2-cycle elimination ---------------------------------------------
@constraint(model, twocycle[i in nodes, j in nodes; i < j],
    y[(i,j)] + y[(j,i)] <= 1)

# ---- (D) seeded reward tangents (static outer-approximation) --------------
# Give the root LP a real reward surface instead of mu <= 20/21. Only emit a
# tangent where h is locally concave: 1000*w0*f2 >= (T_end - z0)^2.
let zs = (8.0, 14.0, 20.0, 26.0, 29.0), ws = (2.0, 6.0, 12.0, 19.0)
    for i in nodes, z0 in zs, w0 in ws
        f2 = 1 - 0.001*(T_end - z0)^2
        1000*w0*f2 >= (T_end - z0)^2 || continue        # concave-region guard
        f1    = w0/(1 + w0)
        df_dw = f2/(1 + w0)^2
        df_dz = f1*0.002*(T_end - z0)
        @constraint(model,
            mu[i] <= f1*f2 + df_dw*((Z_bar[i] - Z[i]) - w0) + df_dz*(Z[i] - z0))
    end
end

# ============================================ fractional connectivity (GSEC)
# For S with 0 not in S and k in S:
#     sum_{arcs entering S} y  >=  x_k   (in-degree of k).
# The most violated S for a fixed k is the min 0->k cut of the support graph
# with capacities y*, found by one max-flow. Separated at fractional nodes.

const NV  = length(nodes_0)          # vertices 0..NNODES  ->  indices 1..NV
idx(i)    = i + 1
nodeid(v) = v - 1

# Edmonds-Karp. Returns (flow, source_side_indices) where the source side is
# the min-cut side containing s.
function maxflow_mincut(cap::Matrix{Float64}, s::Int, snk::Int)
    n   = size(cap, 1)
    res = copy(cap)
    flow = 0.0
    parent = zeros(Int, n)
    while true
        fill!(parent, 0); parent[s] = s
        q = Int[s]
        while !isempty(q)
            u = popfirst!(q)
            for v in 1:n
                if parent[v] == 0 && res[u, v] > 1e-12
                    parent[v] = u; push!(q, v)
                end
            end
        end
        parent[snk] == 0 && break                 # sink unreachable -> optimal
        b = Inf; v = snk
        while v != s; u = parent[v]; b = min(b, res[u, v]); v = u; end
        v = snk
        while v != s; u = parent[v]; res[u, v] -= b; res[v, u] += b; v = u; end
        flow += b
    end
    reachable = falses(n); reachable[s] = true
    stack = Int[s]
    while !isempty(stack)
        u = pop!(stack)
        for v in 1:n
            if !reachable[v] && res[u, v] > 1e-12
                reachable[v] = true; push!(stack, v)
            end
        end
    end
    return flow, Set(v for v in 1:n if reachable[v])
end

function connectivity_callback(cb_data)
    yval = Dict(a => callback_value(cb_data, y[a]) for a in A)

    cap = zeros(Float64, NV, NV)
    for (i, j) in A
        cap[idx(i), idx(j)] = max(0.0, yval[(i, j)])
    end

    seen = Set{Set{Int}}()
    for k in nodes
        xk = sum(yval[(j, k)] for j in nodes_0 if j != k)
        xk > 1e-4 || continue

        f, src = maxflow_mincut(cap, idx(0), idx(k))
        f < xk - 1e-4 || continue
        idx(0) in src || continue                 # keep depot on source side

        S = Set(nodeid(v) for v in 1:NV if !(v in src))   # sink side (node ids)
        (k in S && !(0 in S)) || continue
        S in seen && continue
        push!(seen, S)

        # Strengthen for free: attach the cut to the most-served node in S.
        kbest = argmax(i -> sum(yval[(j, i)] for j in nodes_0 if j != i), collect(S))

        con = @build_constraint(
            sum(y[(i, j)] for (i, j) in A if !(i in S) && (j in S)) >=
            sum(y[(j, kbest)] for j in nodes_0 if j != kbest))
        MOI.submit(model, MOI.UserCut(cb_data), con)
    end
end

# ================================================== lazy tangent refinement
# At integer candidates, add the exact tangent of h wherever mu overshoots.
function tangent_callback(cb_data)
    for i in nodes
        zi  = callback_value(cb_data, Z[i])
        zbi = callback_value(cb_data, Z_bar[i])
        mi  = callback_value(cb_data, mu[i])
        w0  = max(0.0, zbi - zi)
        f1  = w0/(1 + w0)
        f2  = 1 - 0.001*(T_end - zi)^2
        h   = f1*f2
        if mi > h + 1e-5
            df_dw = f2/(1 + w0)^2
            df_dz = f1*0.002*(T_end - zi)
            con = @build_constraint(
                mu[i] <= h + df_dw*((Z_bar[i] - Z[i]) - w0) + df_dz*(Z[i] - zi))
            MOI.submit(model, MOI.LazyConstraint(cb_data), con)
        end
    end
end

# MOI/Gurobi supports both channels at once. If your Gurobi.jl ever objects,
# merge them: one callback that branches on callback_node_status(cb_data,model)
# (INTEGER -> tangent lazy cut, otherwise -> connectivity user cut).
set_attribute(model, MOI.LazyConstraintCallback(), tangent_callback)
set_attribute(model, MOI.UserCutCallback(),        connectivity_callback)

# ===================================================== greedy warm start
# Feasible tour by nearest-feasible insertion with a small fixed wait.
# We set y only; Gurobi completes Z, Z_bar, mu by solving the sub-LP, which
# also keeps the start consistent with the tangent constraints.
function greedy_route()
    unvisited = Set(nodes)
    route  = Tuple{Int,Int}[]
    cur    = 0
    curdep = 0.0
    W      = 3.0
    while true
        best = 0; bestd = Inf
        for j in unvisited
            arr = curdep + t[(cur, j)]
            if arr <= T_end && t[(cur, j)] < bestd
                bestd = t[(cur, j)]; best = j
            end
        end
        best == 0 && break
        arr = curdep + t[(cur, best)]
        push!(route, (cur, best))
        delete!(unvisited, best)
        cur = best
        curdep = min(arr + W, T_end)
    end
    push!(route, (cur, 0))
    return Set(route)
end

let routeset = greedy_route()
    for a in A
        set_start_value(y[a], a in routeset ? 1.0 : 0.0)
    end
end

# ===================================================== solve
set_optimizer_attribute(model, "PreCrush", 1)        # REQUIRED for user cuts
set_optimizer_attribute(model, "TimeLimit", 3600.0)
# Do NOT cap Threads here (no shared subproblem); let Gurobi use all cores.
# Optional knobs if the gap stalls:
#   set_optimizer_attribute(model, "MIPFocus", 3)    # 2 if incumbents lag
#   set_optimizer_attribute(model, "Cuts", 2)

optimize!(model)

# ===================================================== report
println("\n=== Final Solution ===")
println("Status : ", termination_status(model))
if primal_status(model) == MOI.FEASIBLE_POINT
    println("Objective = ", round(objective_value(model), digits=6))
    println("Bound     = ", round(objective_bound(model),  digits=6))
    println("Gap       = ", round(100*relative_gap(model),  digits=4), " %")

    println("\nRoute:")
    for (i, j) in A
        value(y[(i, j)]) > 0.5 && println("  $i -> $j")
    end

    println("\nNode details (visited only):")
    for i in nodes
        if value(sum(y[(j, i)] for j in nodes_0 if j != i)) > 0.5
            w  = max(0.0, value(Z_bar[i]) - value(Z[i]))
            z  = value(Z[i])
            f1 = w/(1 + w)
            f2 = 1 - 0.001*(T_end - z)^2
            println("  Node $i: arrive=$(round(z, digits=2)) ",
                    "depart=$(round(value(Z_bar[i]), digits=2)) ",
                    "w=$(round(w, digits=2)) ",
                    "h=$(round(f1*f2, digits=4)) ",
                    "mu=$(round(value(mu[i]), digits=4))")
        end
    end
else
    println("No feasible solution. Bound = ",
            round(objective_bound(model), digits=6))
end
