# ============================================================================
# Same model as before, but the ROUTE IS NO LONGER A DECISION.
#
# Why: at 100 nodes the old stage-indexed x[(i,j),s] has ~101*100 arcs times
# ~101 stages — over a million binaries, rebuilt from scratch every outer-
# approximation iteration. That's not a "let it run longer" problem, it's
# structurally the wrong tool once the route is known.
#
# So: pick the route once (below, via a simple nearest-neighbor construction —
# swap in your own ordered node list if you have one), then only optimize what
# still varies — how long to wait at each stop. Every constraint that existed
# purely to let the solver pick arcs (one_arc_per_stage, must_depart,
# no_redeparture, chain, visit_once, real_return) disappears, because there's
# no longer an arc-selection decision to constrain. What's left is a pure LP
# refined by the same cutting-plane loop, since h(w,z) is still nonconvex.
# ============================================================================

using JuMP, Gurobi
import MathOptInterface as MOI
using Random

Random.seed!(90)

# ---------- instance ----------
N       = 100                        # candidate customer nodes (was 5)
nodes   = 1:N
nodes_0 = 0:N
T_end   = 30.0
U_max   = 100.0
coords  = Dict(i => (10*rand(), 10*rand()) for i in nodes_0)   # same 10x10 field as the original

α = Dict((i,j) => hypot(coords[i][1]-coords[j][1], coords[i][2]-coords[j][2])
          for i in nodes_0 for j in nodes_0 if i != j)
α[(0,0)] = 0.0   # only ever looked up if build_fixed_route returns an empty route

Q = Dict(i => rand(6:15) for i in nodes)   # no given values for 100 nodes — sampled in the original's range; edit if you have real weights

# ---------- FIXED ROUTE ----------
# Nearest-neighbor construction, stopping before any hop would land a customer
# arrival past T_end (checked assuming zero wait, which is the most permissive
# case — actual waiting only ever raises later arrivals, and the solver below
# will cap it accordingly). Replace the call below with `route = [...]` directly
# if the sequence should come from somewhere else.
function build_fixed_route(coords, nodes, α, T_end)
    unvisited = Set(nodes)
    route     = Int[]
    current   = 0
    z         = 0.0
    while true
        best_j, best_d = 0, Inf
        for j in unvisited
            d = α[(current, j)]
            if z + d <= T_end && d < best_d
                best_j, best_d = j, d
            end
        end
        best_j == 0 && break        # nothing reachable within T_end — stop here
        push!(route, best_j)
        delete!(unvisited, best_j)
        z      += best_d
        current = best_j
    end
    return route
end

route = build_fixed_route(coords, nodes, α, T_end)
M     = length(route)         # number of customer stages the fixed route actually uses
Tmax  = M + 1                 # + 1 for the fixed return-to-depot stage

full_path = vcat([0], route, [0])
stage_arc = Dict(s => (full_path[s], full_path[s+1]) for s in 1:Tmax)   # (i,j) used at each stage — now data
leg_time  = Dict(s => α[stage_arc[s]] for s in 1:Tmax)                  # fixed travel time per stage

println("Fixed route visits $M of $N candidate nodes (T_end = $T_end); Tmax = $Tmax.")

model = Model(Gurobi.Optimizer)

# ---------- timing: same stage-indexed Z / Zbar as before, driven by fixed leg_time instead of x ----------
@variable(model, Z[s in 1:Tmax]    >= 0)
@variable(model, Zbar[s in 0:Tmax] >= 0)
@constraint(model, Zbar[0] == 0)

@constraint(model, propagate[s in 1:Tmax], Z[s] == Zbar[s-1] + leg_time[s])
@constraint(model, seq[s in 1:Tmax],       Zbar[s] >= Z[s])
@constraint(model, max_wait[s in 1:Tmax],  Zbar[s] - Z[s] <= 20)

# customer stages (1..M) get the tight horizon T_end; the return stage (Tmax) gets U_max.
# The original picked this out with a weighted sum over x because it didn't know in
# advance which stage would be the return leg. Now the route is fixed, so it's just
# "which index is Tmax" — no indicator arithmetic needed.
@constraint(model, time_cap[s in 1:M],   Z[s]    <= T_end)
@constraint(model, depart_cap[s in 1:M], Zbar[s] <= T_end)
@constraint(model, time_cap_return,      Z[Tmax]    <= U_max)
@constraint(model, depart_cap_return,    Zbar[Tmax] <= U_max)

# ---------- reward: h(w,z) = f1(w) * f2(z) ----------
# One nu per customer stage. mu_is / cap_by_visit / cap_by_wait / total_reward are
# all gone too: those existed to arbitrate "which node does this stage's reward
# belong to" when the solver was still choosing arcs. With the route fixed, that
# mapping is just route[s] — nothing left to arbitrate.
@variable(model, nu[s in 1:M] >= 0)
@constraint(model, nu_ub[s in 1:M], nu[s] <= 1)

@objective(model, Max, sum(Q[route[s]] * nu[s] for s in 1:M))

# ---------- outer-approximation for h(w,z) = w/(1+w) * (1 - 0.001*(T_end-z)^2) ----------
function separate_plane(model, M, Z, Zbar, nu, route, T_end::Float64, tolerance::Float64)
    println("\n=== PLANE-CUT SEPARATION ===")
    Z_val    = value.(Z)
    Zbar_val = value.(Zbar)
    nu_val   = value.(nu)
    added    = false

    for s in 1:M
        w0 = max(0.0, Zbar_val[s] - Z_val[s])
        z0 = Z_val[s]

        f1    = w0 / (1.0 + w0)
        f2    = 1.0 - 0.001 * (T_end - z0)^2
        h_val = f1 * f2

        if h_val < nu_val[s] - tolerance
            df_dw = f2 / (1.0 + w0)^2
            df_dz = f1 * 0.002 * (T_end - z0)

            @constraint(model,
                nu[s] <= h_val
                       + df_dw * ((Zbar[s] - Z[s]) - w0)
                       + df_dz * (Z[s] - z0))

            println("  Stage $s (node $(route[s])) — cut added")
            println("    (w0, z0)  = ($(round(w0,digits=4)), $(round(z0,digits=4)))")
            println("    h(w0,z0) = $(round(h_val, digits=6))",
                    "  |  nu = $(round(nu_val[s],digits=6))",
                    "  |  viol = $(round(nu_val[s]-h_val,digits=6))")
            added = true
        end
    end
    return added
end

let iteration = 0
    while true
        optimize!(model)
        iteration += 1
        println("\n=== Iteration $iteration ===")

        status = termination_status(model)
        if status != MOI.OPTIMAL && status != MOI.FEASIBLE_POINT
            println("Solver status: $status — terminating.")
            break
        end

        println("Objective = $(round(objective_value(model), digits=6))")

        if !separate_plane(model, M, Z, Zbar, nu, route, T_end, 1e-4)
            println("\nNo violated cuts — optimal solution found.")
            break
        end
    end
end

if primal_status(model) == MOI.FEASIBLE_POINT
    println("\nFinal objective : ", round(objective_value(model), digits=6))

    println("\nRoute (fixed):")
    for s in 1:Tmax
        i, j = stage_arc[s]
        println("  $i -> $j   (stage $s)")
    end

    println("\nNode details (visited only):")
    for s in 1:M
        j  = route[s]
        w  = max(0.0, value(Zbar[s]) - value(Z[s]))
        z  = value(Z[s])
        f1 = w / (1.0 + w)
        f2 = 1.0 - 0.001 * (T_end - z)^2
        println("  Node $j:  arrive=$(round(z,digits=2))  ",
                "depart=$(round(value(Zbar[s]),digits=2))  ",
                "w=$(round(w,digits=2))  ",
                "f1=$(round(f1,digits=4))  ",
                "f2=$(round(f2,digits=4))  ",
                "h=$(round(f1*f2,digits=4))  ",
                "nu=$(round(value(nu[s]),digits=4))")
    end
end