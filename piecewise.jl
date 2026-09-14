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
U_max  = 100.0

Q = Dict(i => 5.0 + 10.0 * rand() for i in nodes)
model = Model(Gurobi.Optimizer)

@variable(model, y[(i,j) in A],      Bin)
@variable(model, Z[i in nodes_0]     >= 0)
@variable(model, Z_bar[i in nodes_0] >= 0)
@variable(model, mu[i in nodes]      >= 0)

@objective(model, Max, sum(Q[i] * mu[i] for i in nodes))

Z_hat = T_end

# --- Routing constraints ---------------------------------------------------
@constraint(model, depot_out, sum(y[(0,j)] for j in nodes) == 1)
@constraint(model, depot_in,  sum(y[(i,0)] for i in nodes) == 1)
@constraint(model, flow[i in nodes],
    sum(y[(i,j)] for j in nodes_0 if i != j) ==
    sum(y[(j,i)] for j in nodes_0 if j != i))
@constraint(model, one_out[i in nodes],
    sum(y[(i,j)] for j in nodes_0 if i != j) <= 1)

@constraint(model, Z_bar[0] == 0)
@constraint(model, Z[0] <= U_max)

# Tightened per-arc big-M (M_ij = T_end + t_ij) instead of a flat 1000.
@constraint(model, MTZ[(i,j) in A; j != 0],
    Z[j] >= Z_bar[i] + t[(i,j)] - (T_end + t[(i,j)]) * (1 - y[(i,j)]))
@constraint(model, MTZ_return[i in nodes],
    Z[0] >= Z_bar[i] + t[(i,0)] - (T_end + t[(i,0)]) * (1 - y[(i,0)]))

@constraint(model, seq[i in nodes], Z[i] <= Z_bar[i])
@constraint(model, depart_bound[i in nodes], Z_bar[i] <= T_end)

@constraint(model, anchor_Z[i in nodes],
    Z[i] >= Z_hat * (1 - sum(y[(j,i)] for j in nodes_0 if j != i)))

@constraint(model, max_wait[i in nodes],      Z_bar[i] - Z[i] <= 20)
@constraint(model, no_expiration[i in nodes], Z[i] <= T_end)
@constraint(model, mu_ub[i in nodes],  mu[i] <= 1)
@constraint(model, mu_visit[i in nodes],
    mu[i] <= sum(y[(j,i)] for j in nodes_0 if j != i))

# ===========================================================================
# 2-D piecewise-linear approximation of  mu[i] <= h(w,z) = f1(w) * f2(z)
#   w = Z_bar[i] - Z[i]  (wait, in [0, 20]),   z = Z[i]  (arrival, in [0, T_end])
#
# Disaggregated lambda formulation with SOS2 on both marginals:
#   - lam[i,p,q] are convex weights on grid vertices (Wg[p], Zg[q]).
#   - Marginals wm[i,p]=sum_q lam, zm[i,q]=sum_p lam are each SOS2, which
#     confines the weight to a single 2x2 grid cell.
#   - w and z are reconstructed from the marginals; the reward is bounded by
#     the interpolated surface value sum lam*H.
#
# This replaces the nonconvex product exactly at the grid vertices and
# linearly between them (error shrinks as the grid is refined). It needs no
# callback and terminates finitely. Refine Wg / Zg to trade accuracy vs size.
# ===========================================================================
f1(w) = w / (1.0 + w)
f2(z) = 1.0 - 0.001 * (T_end - z)^2

# Grid breakpoints. Denser near w=0 where f1 is most curved.
Wg = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 14.0, 20.0]   # w in [0, 20]
Zg = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]           # z in [0, T_end]
P  = length(Wg)
Q  = length(Zg)
H  = [f1(Wg[p]) * f2(Zg[q]) for p in 1:P, q in 1:Q]      # P x Q surface values

@variable(model, lam[i in nodes, p in 1:P, q in 1:Q] >= 0)
@variable(model, wm[i in nodes, p in 1:P] >= 0)          # z-marginals of lam
@variable(model, zm[i in nodes, q in 1:Q] >= 0)          # w-marginals of lam

# Convex-combination structure
@constraint(model, pwl_sum[i in nodes],
    sum(lam[i,p,q] for p in 1:P, q in 1:Q) == 1)
@constraint(model, pwl_wm[i in nodes, p in 1:P],
    wm[i,p] == sum(lam[i,p,q] for q in 1:Q))
@constraint(model, pwl_zm[i in nodes, q in 1:Q],
    zm[i,q] == sum(lam[i,p,q] for p in 1:P))

# Reconstruct (w, z) from the grid
@constraint(model, pwl_w[i in nodes],
    (Z_bar[i] - Z[i]) == sum(Wg[p] * wm[i,p] for p in 1:P))
@constraint(model, pwl_z[i in nodes],
    Z[i] == sum(Zg[q] * zm[i,q] for q in 1:Q))

# Reward bounded by the piecewise-linear surface
@constraint(model, pwl_mu[i in nodes],
    mu[i] <= sum(H[p,q] * lam[i,p,q] for p in 1:P, q in 1:Q))

# SOS2 on each marginal -> weight confined to one adjacent grid cell
@constraint(model, [i in nodes],
    [wm[i,p] for p in 1:P] in MOI.SOS2([Float64(p) for p in 1:P]))
@constraint(model, [i in nodes],
    [zm[i,q] for q in 1:Q] in MOI.SOS2([Float64(q) for q in 1:Q]))

optimize!(model)

println("\n=== Final Solution ===")
status = termination_status(model)
println("Status: $status")
println("PWL grid: $(P) w-breakpoints x $(Q) z-breakpoints")
if primal_status(model) == MOI.FEASIBLE_POINT
    println("Objective = $(round(objective_value(model), digits=6))")

    println("\nRoute:")
    for (i,j) in A
        if value(y[(i,j)]) > 0.5
            println(" $i -> $j")
        end
    end

    println("\nNode details (visited only):")
    println("  (h = true surface value; mu should match it when the grid is fine)")
    for i in nodes
        if value(sum(y[(j,i)] for j in nodes_0 if j != i)) > 0.5
            w = max(0.0, value(Z_bar[i]) - value(Z[i]))
            z = value(Z[i])
            fa = f1(w)
            fb = f2(z)
            println(" Node $i: arrive=$(round(z,digits=2)) ",
                    "depart=$(round(value(Z_bar[i]),digits=2)) ",
                    "w=$(round(w,digits=2)) ",
                    "f1=$(round(fa,digits=4)) ",
                    "f2=$(round(fb,digits=4)) ",
                    "h=$(round(fa*fb,digits=4)) ",
                    "mu=$(round(value(mu[i]),digits=4))")
        end
    end
end