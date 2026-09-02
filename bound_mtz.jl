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

# Reward shape:  R(t,τ) = g(τ) h(t),  g(τ)=τ/(1+τ),  h(t)=1-α(T-t)^2.
const ALPHA = 0.001
sat(τ)   = τ / (1.0 + τ)                 # g(τ)
sat_p(τ) = 1.0 / (1.0 + τ)^2             # g'(τ)

# --- Eq. (30), triangle-lifted (note §9), packaged as the two numbers a tangent needs.
# Given the shared dwell s = τ_i + τ_j, return (Qbar, slope):
#   Qbar  = best the pair can do if they split s optimally   ← this max IS eq (30)
#   slope = derivative of that envelope at s (envelope theorem)
# The arrival time is already eliminated by the triangle-inequality lift, so the
# cut only involves the dwell sum. Requires triangle inequality (true: Euclidean).
function pair_cut_data(s, dij, dj0)
    cij = dij + dj0
    hi  = 1.0 - ALPHA*(cij + s)^2                     # node-i temporal factor (fixed by s)
    best, u, v = -Inf, 0.0, 0.0
    for k in 0:200                                    # best split of s between i and j
        uu = s*k/200; vv = s - uu
        val = sat(uu)*hi + sat(vv)*(1.0 - ALPHA*(dj0+vv)^2)
        val > best && ((best, u, v) = (val, uu, vv))
    end
    Bi = sat(u)*2.0*ALPHA*(cij + s)                   # B_i at the optimal split
    slope = v > 1e-9 ?
        sat_p(v)*(1.0 - ALPHA*(dj0+v)^2) - Bi - sat(v)*2.0*ALPHA*(dj0+v) :  # A_j - B_i - B_j
        sat_p(u)*hi - Bi                                                    # corner: A_i - B_i
    return best, slope
end

function plane_cut_callback(cb_data)
    Z_bar_val = callback_value.(cb_data, Z_bar)
    Z_val     = callback_value.(cb_data, Z)
    mu_val    = callback_value.(cb_data, mu)

    # -------- single-node objective tangents (unchanged) --------
    for i in nodes
        w0 = max(0.0, Z_bar_val[i] - Z_val[i]); z0 = Z_val[i]
        f1 = w0 / (1.0 + w0)
        f2 = 1.0 - ALPHA * (T_end - z0)^2
        h_val = f1 * f2
        if h_val < mu_val[i] - 1e-5
            df_dw = f2 / (1.0 + w0)^2
            df_dz = f1 * 2.0 * ALPHA * (T_end - z0)
            con = @build_constraint(
                mu[i] <= h_val + df_dw*((Z_bar[i]-Z[i]) - w0) + df_dz*(Z[i]-z0))
            MOI.submit(model, MOI.LazyConstraint(cb_data), con)
            println("   → Node cut for node $i")
        end
    end

    # -------- arc-pair cuts (eq. 30), reads exactly like the recipe --------
    for (i,j) in A
        (i in nodes && j in nodes) || continue                 # both reward-bearing
        callback_value(cb_data, y[(i,j)]) > 0.5 || continue     # arc actually used
        s = max(0.0, Z_bar_val[i]-Z_val[i]) + max(0.0, Z_bar_val[j]-Z_val[j])
        s <= 1e-9 && continue
        Qbar, slope = pair_cut_data(s, t[(i,j)], t[(j,0)])
        if mu_val[i] + mu_val[j] > Qbar + 1e-5
            M = 2.0 + abs(slope)*40.0 + 1.0                    # safe activation coeff.
            con = @build_constraint(
                mu[i] + mu[j] <= Qbar
                    + slope*((Z_bar[i]-Z[i]) + (Z_bar[j]-Z[j]) - s)
                    + M*(1 - y[(i,j)]))
            MOI.submit(model, MOI.LazyConstraint(cb_data), con)
            println("   → Pair cut for arc $i→$j")
        end
    end
end
MOI.set(model, MOI.LazyConstraintCallback(), plane_cut_callback)

optimize!(model)

println("\n=== Final Solution ===")
println("Status: $(termination_status(model))")
if primal_status(model) == MOI.FEASIBLE_POINT
    println("Objective = $(round(objective_value(model), digits=6))")
    println("\nRoute:")
    for (i,j) in A
        value(y[(i,j)]) > 0.5 && println(" $i -> $j")
    end
    println("\nNode details (visited only):")
    for i in nodes
        if value(sum(y[(j,i)] for j in nodes_0 if j != i)) > 0.5
            w = max(0.0, value(Z_bar[i]) - value(Z[i])); z = value(Z[i])
            f1 = w/(1.0+w); f2 = 1.0 - ALPHA*(T_end-z)^2
            println(" Node $i: arrive=$(round(z,digits=2)) depart=$(round(value(Z_bar[i]),digits=2)) ",
                    "w=$(round(w,digits=2)) h=$(round(f1*f2,digits=4)) mu=$(round(value(mu[i]),digits=4))")
        end
    end
end