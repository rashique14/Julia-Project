using JuMP, Gurobi
import MathOptInterface as MOI
using Random

Random.seed!(90)

nodes   = 1:5
nodes_0 = 0:5
Tmax    = length(nodes) + 1          # up to 5 real visits + 1 return leg
T_end   = 30.0
U_max   = 100.0
coords  = Dict(i => (10*rand(), 10*rand()) for i in nodes_0)

A     = [(i,j) for i in nodes_0 for j in nodes_0 if i != j]
A_ext = vcat(A, [(0,0)])             # zero-cost "parked at depot" arc

α = Dict((i,j) => hypot(coords[i][1]-coords[j][1], coords[i][2]-coords[j][2]) for (i,j) in A)
α[(0,0)] = 0.0

Q = Dict(1=>10, 2=>8, 3=>12, 4=>15, 5=>6)

model = Model(Gurobi.Optimizer)

# ---------- routing: which arc is stage s's move ----------
@variable(model, x[(i,j) in A_ext, s in 1:Tmax], Bin)

@constraint(model, one_arc_per_stage[s in 1:Tmax],
    sum(x[(i,j), s] for (i,j) in A_ext) == 1)

@constraint(model, must_depart,
    sum(x[(0,j), 1] for j in nodes) == 1)

# NEW: once you're back at the depot, you park — you don't leave again.
# Nothing in the stage structure gives you this for free.
@constraint(model, no_redeparture[j in nodes, s in 2:Tmax],
    x[(0,j), s] == 0)

@constraint(model, chain[k in nodes_0, s in 1:Tmax-1],
    sum(x[(i,k), s] for i in nodes_0 if (i,k) in A_ext) ==
    sum(x[(k,j), s+1] for j in nodes_0 if (k,j) in A_ext))

@constraint(model, visit_once[j in nodes],
    sum(x[(i,j), s] for s in 1:Tmax for i in nodes_0 if (i,j) in A_ext) <= 1)

@constraint(model, real_return,
    sum(x[(i,0), s] for s in 1:Tmax for i in nodes) == 1)

# ---------- timing: indexed by STAGE, not node — this is what kills MTZ ----------
@variable(model, Z[s in 1:Tmax]    >= 0)
@variable(model, Zbar[s in 0:Tmax] >= 0)
@constraint(model, Zbar[0] == 0)

@constraint(model, propagate[s in 1:Tmax],
    Z[s] == Zbar[s-1] + sum(α[(i,j)] * x[(i,j), s] for (i,j) in A_ext))

@constraint(model, seq[s in 1:Tmax], Zbar[s] >= Z[s])
@constraint(model, max_wait[s in 1:Tmax], Zbar[s] - Z[s] <= 20)

# customer-facing stages get the tight horizon T_end; whichever stage lands
# back on the depot (real return, or padding) gets the loose U_max —
# same asymmetry as your Z[i]<=T_end vs Z[0]<=U_max, picked out with a
# weighted sum instead of a node lookup
@constraint(model, time_cap[s in 1:Tmax],
    Z[s] <= T_end * sum(x[(i,j),s] for (i,j) in A_ext if j in nodes)
          + U_max * sum(x[(i,0),s] for i in nodes_0 if (i,0) in A_ext))
@constraint(model, depart_cap[s in 1:Tmax],
    Zbar[s] <= T_end * sum(x[(i,j),s] for (i,j) in A_ext if j in nodes)
             + U_max * sum(x[(i,0),s] for i in nodes_0 if (i,0) in A_ext))

# ---------- reward: h(w,z) = f1(w) * f2(z), linked back to customers ----------
@variable(model, nu[s in 1:Tmax] >= 0)          # stage-level h(w,z) proxy
@constraint(model, nu_ub[s in 1:Tmax], nu[s] <= 1)

@variable(model, mu_is[i in nodes, s in 1:Tmax] >= 0)
@variable(model, mu[i in nodes] >= 0)
@constraint(model, mu_ub[i in nodes], mu[i] <= 1)

@constraint(model, cap_by_visit[i in nodes, s in 1:Tmax],
    mu_is[i,s] <= sum(x[(k,i), s] for k in nodes_0 if (k,i) in A_ext))
@constraint(model, cap_by_wait[i in nodes, s in 1:Tmax],
    mu_is[i,s] <= nu[s])
@constraint(model, total_reward[i in nodes],
    mu[i] == sum(mu_is[i,s] for s in 1:Tmax))

@objective(model, Max, sum(Q[i] * mu[i] for i in nodes))

# ---------- outer-approximation for h(w,z) = w/(1+w) * (1 - 0.001*(T_end-z)^2) ----------
function separate_plane(model, Tmax, Z, Zbar, nu, T_end::Float64, tolerance::Float64)
    println("\n=== PLANE-CUT SEPARATION ===")
    Z_val    = value.(Z)
    Zbar_val = value.(Zbar)
    nu_val   = value.(nu)
    added    = false

    for s in 1:Tmax
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

            println("  Stage $s — cut added")
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

        if !separate_plane(model, Tmax, Z, Zbar, nu, T_end, 1e-4)
            println("\nNo violated cuts — optimal solution found.")
            break
        end
    end
end

if primal_status(model) == MOI.FEASIBLE_POINT
    println("\nFinal objective : ", round(objective_value(model), digits=6))

    active = sort(
        [(i,j,s) for (i,j) in A_ext for s in 1:Tmax
         if value(x[(i,j),s]) > 0.5 && (i,j) != (0,0)],
        by = e -> e[3])

    println("\nRoute:")
    for (i,j,s) in active
        println("  $i -> $j   (stage $s)")
    end

    println("\nNode details (visited only):")
    for (i,j,s) in active
        j == 0 && continue                 # skip the return-to-depot leg
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
                "mu=$(round(value(mu[j]),digits=4))")
    end
end