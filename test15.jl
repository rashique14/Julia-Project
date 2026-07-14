using JuMP, Gurobi

# ---------- Sets ----------
nodes   = 1:5
nodes_0 = 0:5
Tmax    = length(nodes) + 1     # up to 5 visits + 1 return trip

# real arcs, plus a zero-cost depot self-loop used only as "padding"
# once the vehicle has genuinely returned home
A     = [(i, j) for i in nodes_0 for j in nodes_0 if i != j]
A_ext = vcat(A, [(0, 0)])

coords = Dict(i => (Float64(i), 0.0) for i in nodes_0)
α = Dict((i,j) => hypot(coords[i][1]-coords[j][1], coords[i][2]-coords[j][2]) for (i,j) in A)
α[(0,0)] = 0.0

Q            = Dict(1=>10, 2=>8, 3=>12, 4=>15, 5=>6)
U_max        = 100.0
max_wait_cap = 20.0

model = Model(Gurobi.Optimizer)

# ---------- routing: which arc is stage s's move ----------
@variable(model, x[(i,j) in A_ext, s in 1:Tmax], Bin)

# ---------- timing: indexed by STAGE, not node ----------
@variable(model, Z[s in 1:Tmax] >= 0)
@variable(model, Zbar[s in 0:Tmax] >= 0)
@constraint(model, Zbar[0] == 0)

# ---------- reward ----------
@variable(model, nu[s in 1:Tmax] >= 0)                     # stage-level ≈ wait/(1+wait)
@constraint(model, nu_bound[s in 1:Tmax], nu[s] <= 1)
@variable(model, mu_is[i in nodes, s in 1:Tmax] >= 0)      # reward IF i is at stage s
@variable(model, mu[i in nodes] >= 0)                      # total reward for customer i

# ============ ROUTING — no MTZ anywhere ============

@constraint(model, one_arc_per_stage[s in 1:Tmax],
    sum(x[(i,j), s] for (i,j) in A_ext) == 1)

@constraint(model, must_depart,
    sum(x[(0,j), 1] for j in nodes) == 1)          # must actually visit ≥1 customer

@constraint(model, chain[k in nodes_0, s in 1:Tmax-1],
    sum(x[(i,k), s] for i in nodes_0 if (i,k) in A_ext) ==
    sum(x[(k,j), s+1] for j in nodes_0 if (k,j) in A_ext))

@constraint(model, visit_once[j in nodes],
    sum(x[(i,j), s] for s in 1:Tmax for i in nodes_0 if (i,j) in A_ext) <= 1)

@constraint(model, real_return,
    sum(x[(i,0), s] for s in 1:Tmax for i in nodes) == 1)

# ============ TIME — replaces MTZ entirely ============

@constraint(model, propagate[s in 1:Tmax],
    Z[s] == Zbar[s-1] + sum(α[(i,j)] * x[(i,j), s] for (i,j) in A_ext))

@constraint(model, seq[s in 1:Tmax], Zbar[s] >= Z[s])
@constraint(model, wait_cap[s in 1:Tmax], Zbar[s] - Z[s] <= max_wait_cap)
@constraint(model, budget, Z[Tmax] <= U_max)

# ============ link stage-time to customer reward (bound = 1, not big-M) ============

@constraint(model, cap_by_visit[i in nodes, s in 1:Tmax],
    mu_is[i,s] <= sum(x[(k,i), s] for k in nodes_0 if (k,i) in A_ext))
@constraint(model, cap_by_wait[i in nodes, s in 1:Tmax],
    mu_is[i,s] <= nu[s])
@constraint(model, total_reward[i in nodes],
    mu[i] == sum(mu_is[i,s] for s in 1:Tmax))

@objective(model, Max, sum(Q[i] * mu[i] for i in nodes))

# ============ lazy cuts: outer-linearize nu[s] ≈ w/(1+w) ============
function callback(cb_data)
    Z_val    = callback_value.(cb_data, Z)
    Zbar_val = callback_value.(cb_data, Zbar)
    nu_val   = callback_value.(cb_data, nu)

    tolerance = 1e-6
    for s in 1:Tmax
        w = Zbar_val[s] - Z_val[s]
        actual_val = w / (1 + w)
        if actual_val < nu_val[s] - tolerance
            slope = 1 / (1 + w)^2
            con = @build_constraint(nu[s] <= actual_val + slope * ((Zbar[s] - Z[s]) - w))
            MOI.submit(model, MOI.LazyConstraint(cb_data), con)
            println("   → Cut added for stage $s")
        end
    end
end
set_attribute(model, MOI.LazyConstraintCallback(), callback)

println("Start...\n")
optimize!(model)

if termination_status(model) in (MOI.OPTIMAL, MOI.TIME_LIMIT)
    active_edges = sort(
        [(i,j,s) for (i,j) in A_ext for s in 1:Tmax
         if value(x[(i,j),s]) > 0.5 && (i,j) != (0,0)],
        by = e -> e[3])

    using JSON
    open("route.json", "w") do f
        JSON.print(f, Dict("edges" => [(i,j) for (i,j,s) in active_edges]))
    end
    println("Route exported to route.json")

    println("\n=== ROUTE (visiting order — free from the stage index) ===")
    for (i,j,s) in active_edges
        w = value(Zbar[s]) - value(Z[s])
        rw = j in nodes ? " reward=$(round(value(mu_is[j,s]), digits=4))" : ""
        println("Stage $s: $i → $j | arrive=$(round(value(Z[s]),digits=2)) ",
                "depart=$(round(value(Zbar[s]),digits=2)) wait=$(round(w,digits=2))$rw")
    end

    println("\n=== PER-CUSTOMER SUMMARY ===")
    for i in nodes
        visited = value(sum(x[(k,i),s] for s in 1:Tmax for k in nodes_0 if (k,i) in A_ext)) > 0.5
        println(visited ?
            "Customer $i: visited | reward = $(round(value(mu[i]), digits=4))" :
            "Customer $i: not visited")
    end

    println("\nTotal reward: ", round(objective_value(model), digits=4))
else
    println("Status: ", termination_status(model))
end