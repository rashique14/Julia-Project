using JuMP, Gurobi
import MathOptInterface as MOI
using Random

Random.seed!(90)
nodes   = 1:12
nodes_0 = 0:12
K       = 4
T_end   = 30.0
coords  = Dict(i => (10*rand(), 10*rand()) for i in nodes_0)

A = [(i,j) for i in nodes_0 for j in nodes_0 if i != j]
t = Dict((i,j) => hypot(coords[i][1]-coords[j][1], coords[i][2]-coords[j][2])
         for (i,j) in A)
U_max  = 100.0

Q = Dict(i => 5.0 + 10.0 * rand() for i in nodes)
model = Model(Gurobi.Optimizer)
set_optimizer_attribute(model, "NonConvex", 2)        # NEW: needed for mu <= g*(1-r)

@variable(model, y[(i,j) in A],      Bin)
@variable(model, Z[i in nodes_0]     >= 0)
@variable(model, Z_bar[i in nodes_0] >= 0)
@variable(model, mu[i in nodes]      >= 0)

# NEW: split the reward h = f1(w) * f2(z) into two on/off pieces
#   g = waiting part     g <= w/(1+w)
#   r = freshness loss   r >= 0.001*(T_end - Z)^2      so  f2 = 1 - r
@variable(model, 0 <= g[i in nodes] <= 20/21)          # 20/21 = f1(20), max wait 20
@variable(model, 0 <= r[i in nodes] <= 0.001 * T_end^2)

# NEW: on/off indicator of node i (1 if visited)
@expression(model, v[i in nodes], sum(y[(j,i)] for j in nodes_0 if j != i))

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
    Z[j] >= Z_bar[i] + t[(i,j)] - (T_end + t[(i,j)]) * (1 - y[(i,j)]))
@constraint(model, MTZ_return[i in nodes],
    Z[0] >= Z_bar[i] + t[(i,0)] - (T_end + t[(i,0)]) * (1 - y[(i,0)]))

@constraint(model, seq[i in nodes], Z[i] <= Z_bar[i])
@constraint(model, depart_bound[i in nodes], Z_bar[i] <= T_end)

@constraint(model, anchor_Z[i in nodes],
    Z[i] >= Z_hat * (1 - sum(y[(j,i)] for j in nodes_0 if j != i)))

@constraint(model, time_budget,
    sum(t[(i,j)] * y[(i,j)] for (i,j) in A if j != 0) +
    sum(Z_bar[i] - Z[i] for i in nodes) <= T_end)

@constraint(model, wait_link[i in nodes],
    Z_bar[i] - Z[i] <= 20 * sum(y[(j,i)] for j in nodes_0 if j != i))

# NEW: the reward. Exact: at the optimum g = f1(w), r = 0.001(T_end-Z)^2, so mu = h.
@constraint(model, reward[i in nodes], mu[i] <= g[i] - g[i] * r[i])

# CHANGED: static perspective cuts (replace the old static tangents on mu).
#   For an on/off function f with f(0)=0, the perspective cut at reference a is
#       f'(a) * x + (f(a) - a f'(a)) * v
#   g piece: f(w) = w/(1+w)  ->  g <= (w + a^2 v) / (1+a)^2
#   r piece: f(s) = 0.001 s^2 ->  r >= 0.001 (2 b s - b^2 v),   s = T_end - Z
for i in nodes, a in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    @constraint(model, g[i] <= ((Z_bar[i] - Z[i]) + a^2 * v[i]) / (1 + a)^2)
end
for i in nodes, b in (5.0, 10.0, 15.0, 20.0, 25.0)
    @constraint(model, r[i] >= 0.001 * (2b * (T_end - Z[i]) - b^2 * v[i]))
end

@constraint(model, max_wait[i in nodes],      Z_bar[i] - Z[i] <= 20)
@constraint(model, no_expiration[i in nodes], Z[i] <= T_end)
@constraint(model, mu_ub[i in nodes],  mu[i] <= 1)
@constraint(model, mu_visit[i in nodes],
    mu[i] <= sum(y[(j,i)] for j in nodes_0 if j != i))


cut_counts = Dict(i => 0 for i in nodes)
total_cuts = Ref(0)
const tol = 1e-5

# CHANGED: the callback now adds perspective cuts (slide 63)
function perspective_cut_callback(cb_data)

    if callback_node_status(cb_data, model) != MOI.CALLBACK_NODE_STATUS_INTEGER
        return
    end

    for i in nodes
        v0 = callback_value(cb_data, v[i])
        if v0 < 1e-6
            continue            # node not visited: w = 0, so nothing to cut (slide 64)
        end
        Z0 = callback_value(cb_data, Z[i])
        w0 = max(0.0, callback_value(cb_data, Z_bar[i]) - Z0)
        s0 = max(0.0, T_end - Z0)
        g0 = callback_value(cb_data, g[i])
        r0 = callback_value(cb_data, r[i])

        # g piece: f(w) = w/(1+w),  a = w/v
        a = min(w0 / v0, 20.0)
        if g0 - v0 * a / (1 + a) > tol
            con = @build_constraint(g[i] <= ((Z_bar[i] - Z[i]) + a^2 * v[i]) / (1 + a)^2)
            MOI.submit(model, MOI.LazyConstraint(cb_data), con)
            cut_counts[i] += 1
            total_cuts[]  += 1
        end

        # r piece: f(s) = 0.001 s^2,  b = s/v
        b = min(s0 / v0, T_end)
        if v0 * 0.001 * b^2 - r0 > tol
            con = @build_constraint(r[i] >= 0.001 * (2b * (T_end - Z[i]) - b^2 * v[i]))
            MOI.submit(model, MOI.LazyConstraint(cb_data), con)
            cut_counts[i] += 1
            total_cuts[]  += 1
        end
    end
end

MOI.set(model, MOI.LazyConstraintCallback(), perspective_cut_callback)

optimize!(model)

println("\n=== Lazy Cut Summary ===")
println("Total lazy cuts added: $(total_cuts[])")
for i in nodes
    if cut_counts[i] > 0
        println("   node $i: $(cut_counts[i]) cuts")
    end
end

println("\n=== Final Solution ===")
status = termination_status(model)
println("Status: $status")
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
            w = max(0.0, value(Z_bar[i]) - value(Z[i]))
            z = value(Z[i])
            f1 = w / (1.0 + w)
            f2 = 1.0 - 0.001 * (T_end - z)^2
            println(" Node $i: arrive=$(round(z,digits=2)) ",
                    "depart=$(round(value(Z_bar[i]),digits=2)) ",
                    "w=$(round(w,digits=2)) ",
                    "f1=$(round(f1,digits=4)) ",
                    "f2=$(round(f2,digits=4)) ",
                    "h=$(round(f1*f2,digits=4)) ",
                    "mu=$(round(value(mu[i]),digits=4))")
        end
    end
end