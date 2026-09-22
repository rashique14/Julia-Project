using JuMP, Gurobi
import MathOptInterface as MOI
using Random

Random.seed!(90)
nodes   = 1:8
nodes_0 = 0:8
T_end   = 30.0
coords  = Dict(i => (10*rand(), 10*rand()) for i in nodes_0)

A = [(i,j) for i in nodes_0 for j in nodes_0 if i != j]
t = Dict((i,j) => hypot(coords[i][1]-coords[j][1], coords[i][2]-coords[j][2])
         for (i,j) in A)
U_max = 100.0

Q = Dict(i => 5.0 + 10.0 * rand() for i in nodes)
model = Model(Gurobi.Optimizer)
set_attribute(model, "NonConvex", 2)

@variable(model, y[(i,j) in A], Bin)
@variable(model, 0 <= Z[i in nodes_0] <= U_max)
@variable(model, 0 <= Z_bar[i in nodes_0] <= T_end)
@variable(model, mu[i in nodes] >= 0)

# New variables for the reward
@variable(model, 0 <= w[i in nodes] <= 20)   # waiting time
@variable(model, 0 <= u[i in nodes] <= 1)    # u ≤ f1(w) = w/(1+w)
@variable(model, 0 <= v[i in nodes] <= 1)    # v ≤ f2(z) = 1 - 0.001(30-z)^2

@objective(model, Max, sum(Q[i] * mu[i] for i in nodes))

Z_hat = T_end

# ---------------- Routing (same as before) ----------------
@constraint(model, depot_out, sum(y[(0,j)] for j in nodes) == 1)
@constraint(model, depot_in,  sum(y[(i,0)] for i in nodes) == 1)
@constraint(model, flow[i in nodes],
    sum(y[(i,j)] for j in nodes_0 if i != j) ==
    sum(y[(j,i)] for j in nodes_0 if j != i))
@constraint(model, one_out[i in nodes],
    sum(y[(i,j)] for j in nodes_0 if i != j) <= 1)

@constraint(model, Z_bar[0] == 0)

@constraint(model, MTZ[(i,j) in A; j != 0],
    Z[j] >= Z_bar[i] + t[(i,j)] - (T_end + t[(i,j)]) * (1 - y[(i,j)]))
@constraint(model, MTZ_return[i in nodes],
    Z[0] >= Z_bar[i] + t[(i,0)] - (T_end + t[(i,0)]) * (1 - y[(i,0)]))

@constraint(model, seq[i in nodes], Z[i] <= Z_bar[i])
@constraint(model, anchor_Z[i in nodes],
    Z[i] >= Z_hat * (1 - sum(y[(j,i)] for j in nodes_0 if j != i)))
@constraint(model, no_expiration[i in nodes], Z[i] <= T_end)
@constraint(model, mu_ub[i in nodes], mu[i] <= 1)
@constraint(model, mu_visit[i in nodes],
    mu[i] <= sum(y[(j,i)] for j in nodes_0 if j != i))

# ---------------- Reward: mu ≤ f1(w) * f2(z), written exactly ----------------
@constraint(model, w_def[i in nodes], w[i] == Z_bar[i] - Z[i])                # also enforces max_wait via w ≤ 20
@constraint(model, f1_def[i in nodes], u[i] * (1 + w[i]) <= w[i])             # u ≤ w/(1+w)
@constraint(model, f2_def[i in nodes], v[i] + 0.001 * (T_end - Z[i])^2 <= 1)  # v ≤ 1 - 0.001(30-z)^2
@constraint(model, h_def[i in nodes], mu[i] <= u[i] * v[i])                   # mu ≤ u·v

optimize!(model)

# ---------------- Results ----------------
println("\n=== Final Solution ===")
println("Status: $(termination_status(model))")
if primal_status(model) == MOI.FEASIBLE_POINT
    println("Objective = $(round(objective_value(model), digits=6))")

    println("\nRoute:")
    for (i,j) in A
        if value(y[(i,j)]) > 0.5
            println(" $i -> $j")
        end
    end

    println("\nNode details (visited only):")
    for i in nodes
        if value(sum(y[(j,i)] for j in nodes_0 if j != i)) > 0.5
            wi = max(0.0, value(Z_bar[i]) - value(Z[i]))
            zi = value(Z[i])
            f1 = wi / (1.0 + wi)
            f2 = 1.0 - 0.001 * (T_end - zi)^2
            println(" Node $i: arrive=$(round(zi,digits=2)) ",
                    "depart=$(round(value(Z_bar[i]),digits=2)) ",
                    "w=$(round(wi,digits=2)) ",
                    "f1=$(round(f1,digits=4)) ",
                    "f2=$(round(f2,digits=4)) ",
                    "h=$(round(f1*f2,digits=4)) ",
                    "mu=$(round(value(mu[i]),digits=4))")
        end
    end
end