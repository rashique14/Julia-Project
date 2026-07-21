using JuMP, Gurobi
import MathOptInterface as MOI
using Random

Random.seed!(90)
nodes   = 1:15
nodes_0 = 0:15
K       = 4
T_end   = 30.0
coords  = Dict(i => (10*rand(), 10*rand()) for i in nodes_0)

A = [(i,j) for i in nodes_0 for j in nodes_0 if i != j]
t = Dict((i,j) => hypot(coords[i][1]-coords[j][1], coords[i][2]-coords[j][2])
         for (i,j) in A)
M_time = 1000.0
U_max  = 100.0

Q = Dict(i => 5.0 + 10.0 * rand() for i in nodes)
model = Model(Gurobi.Optimizer)

@variable(model, y[(i,j) in A],      Bin)
@variable(model, Z[i in nodes_0]     >= 0)
@variable(model, Z_bar[i in nodes_0] >= 0)
@variable(model, mu[i in nodes]      >= 0)

@objective(model, Max, sum(Q[i] * mu[i] for i in nodes))

Z_hat = T_end     

@constraint(model, depot_out, sum(y[(0,j)] for j in nodes) == 1)
@constraint(model, depot_in,  sum(y[(i,0)] for i in nodes) == 1)
@constraint(model, flow[i in nodes],
    sum(y[(i,j)] for j in nodes_0 if i != j) ==
    sum(y[(j,i)] for j in nodes_0 if j != i))
@constraint(model, one_out[i in nodes],
    sum(y[(i,j)] for j in nodes_0 if i != j) <= 1)

@constraint(model, Z_bar[0] == 0)
@constraint(model, Z[0] <= U_max)
@constraint(model, MTZ[(i,j) in A; j != 0],
    Z[j] >= Z_bar[i] + t[(i,j)] - M_time * (1 - y[(i,j)]))
@constraint(model, MTZ_return[i in nodes],
    Z[0] >= Z_bar[i] + t[(i,0)] - M_time * (1 - y[(i,0)]))
@constraint(model, seq[i in nodes], Z[i] <= Z_bar[i])


@constraint(model, depart_bound[i in nodes], Z_bar[i] <= T_end)


@constraint(model, anchor_Z[i in nodes],
    Z[i] >= Z_hat * (1 - sum(y[(j,i)] for j in nodes_0 if j != i)))

@constraint(model, max_wait[i in nodes],      Z_bar[i] - Z[i] <= 20)
@constraint(model, no_expiration[i in nodes], Z[i] <= T_end)
@constraint(model, mu_ub[i in nodes],  mu[i] <= 1)
@constraint(model, mu_visit[i in nodes],
    mu[i] <= sum(y[(j,i)] for j in nodes_0 if j != i))


function separate_plane(model, nodes, Z, Z_bar, mu,
                        T_end::Float64, tolerance::Float64)
    println("\n=== PLANE-CUT SEPARATION ===")
    Z_bar_val = value.(Z_bar)
    Z_val     = value.(Z)
    mu_val    = value.(mu)
    added     = false

    for i in nodes
        w0    = max(0.0, Z_bar_val[i] - Z_val[i])
        z0    = Z_val[i]

        f1    = w0  / (1.0 + w0)
        f2    = 1.0 - 0.001 * (T_end - z0)^2   
        h_val = f1 * f2

        if h_val < mu_val[i] - tolerance
            df_dw = f2 / (1.0 + w0)^2           
            df_dz = f1 * 0.002 * (T_end - z0)   

        
            @constraint(model,
                mu[i] <= h_val
                       + df_dw * ((Z_bar[i] - Z[i]) - w0)
                       + df_dz * (Z[i] - z0))

            println("  Node $i — cut added")
            println("    (w0, z0)  = ($(round(w0,digits=4)), $(round(z0,digits=4)))")
            println("    h(w0,z0) = $(round(h_val, digits=6))",
                    "  |  mu = $(round(mu_val[i],digits=6))",
                    "  |  viol = $(round(mu_val[i]-h_val,digits=6))")
            println("    dh/dw = $(round(df_dw,digits=6))",
                    "  |  dh/dz = $(round(df_dz,digits=6))")
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

        if !separate_plane(model, nodes, Z, Z_bar, mu, T_end, 1e-4)
            println("\nNo violated cuts — optimal solution found.")
            break
        end
    end
end

if primal_status(model) == MOI.FEASIBLE_POINT
    println("\nFinal objective : ", round(objective_value(model), digits=6))

    println("\nRoute:")
    for (i,j) in A
        value(y[(i,j)]) > 0.5 && println("  $i -> $j")
    end

    println("\nNode details (visited only):")
    for i in nodes
        if sum(value(y[(j,i)]) for j in nodes_0 if j != i) > 0.5
            w  = max(0.0, value(Z_bar[i]) - value(Z[i]))
            z  = value(Z[i])
            f1 = w  / (1.0 + w)
            f2 = 1.0 - 0.001 * (T_end - z)^2
            println("  Node $i:  arrive=$(round(z,digits=2))  ",
                    "depart=$(round(value(Z_bar[i]),digits=2))  ",
                    "w=$(round(w,digits=2))  ",
                    "f1=$(round(f1,digits=4))  ",
                    "f2=$(round(f2,digits=4))  ",
                    "h=$(round(f1*f2,digits=4))  ",
                    "mu=$(round(value(mu[i]),digits=4))")
        end
    end
end