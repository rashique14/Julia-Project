using JuMP, Gurobi
import MathOptInterface as MOI
using Random
Random.seed!(60)

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

# ---------------------------------------------------------------------------
# Reward primitives (named to avoid clashing with the f1/f2 locals used in
# the final reporting block).
#   dwell gain  gW(w) = w/(1+w)                 (concave, increasing in dwell)
#   time  gain  gT(z) = 1 - 0.001 (T_end - z)^2 (concave in arrival time)
# per-node reward  h(w,z) = gW(w) * gT(z)
# ---------------------------------------------------------------------------
gW(w)  = w / (1.0 + w)
gWp(w) = 1.0 / (1.0 + w)^2
gT(z)  = 1.0 - 0.001 * (T_end - z)^2
gTp(z) = 0.002 * (T_end - z)

# ---------------------------------------------------------------------------
# Q_ij(t_i, B):  best TOTAL reward the consecutive pair (i -> j) can earn if
# they split a combined dwell budget B = tau_i + tau_j optimally, while
# respecting the per-node wait cap (<= 20) and the coupling in time:
# if i keeps s of the budget, j arrives at  t_i + s + t_ij.
#
#   g(s) = gW(s)*gT(t_i) + gW(B-s)*gT(t_i + s + t_ij),   s in [max(0,B-20), min(20,B)]
#
# Returns the optimal value plus the partials dQ/dt_i and dQ/dB (envelope
# theorem: hold the optimal split s* fixed) used to build the tangent plane.
# ---------------------------------------------------------------------------
gpair(s, t_i, B, t_ij) = begin
    zj = t_i + s + t_ij
    gW(s) * gT(t_i) + gW(B - s) * gT(zj)
end

function solve_pair(t_i, B, t_ij)
    B  = max(B, 0.0)
    lo = max(0.0, B - 20.0)
    hi = max(min(20.0, B), lo)          # guard against empty range (B > 40)

    # coarse grid to locate the global region, then golden-section refine so
    # Qval is an accurate (non-under-estimated) value of the max.
    N = 400
    best_s, best_val = lo, -Inf
    for k in 0:N
        s = lo + (hi - lo) * k / N
        v = gpair(s, t_i, B, t_ij)
        if v > best_val; best_val, best_s = v, s; end
    end
    if hi > lo
        a = max(lo, best_s - (hi - lo) / N)
        b = min(hi, best_s + (hi - lo) / N)
        phi = (sqrt(5.0) - 1.0) / 2
        c = b - phi*(b - a); d = a + phi*(b - a)
        fc = gpair(c, t_i, B, t_ij); fd = gpair(d, t_i, B, t_ij)
        for _ in 1:60
            if fc < fd
                a, c, fc = c, d, fd
                d = a + phi*(b - a); fd = gpair(d, t_i, B, t_ij)
            else
                b, d, fd = d, c, fc
                c = b - phi*(b - a); fc = gpair(c, t_i, B, t_ij)
            end
        end
        s_ref = (a + b) / 2
        if gpair(s_ref, t_i, B, t_ij) > best_val
            best_s, best_val = s_ref, gpair(s_ref, t_i, B, t_ij)
        end
    end

    s    = best_s
    zj   = t_i + s + t_ij
    Qval = best_val
    # envelope theorem: differentiate holding s* fixed.
    dQ_dti = gW(s) * gTp(t_i) + gW(B - s) * gTp(zj)   # zj = t_i + s + t_ij  (slope 1 in t_i)
    dQ_dB  = gWp(B - s) * gT(zj)                      # tau_j = B - s
    return Qval, dQ_dti, dQ_dB, s
end

# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Lazy separation:  single-node outer-approximation cuts  (as before)
#                 + Equation-30 pair cuts                 (new)
# ---------------------------------------------------------------------------
M_pair = 100.0   # big-M guard: makes a pair cut vacuous when arc (i,j) is unused.

function plane_cut_callback(cb_data)
    Z_val     = callback_value.(cb_data, Z)
    Z_bar_val = callback_value.(cb_data, Z_bar)
    mu_val    = callback_value.(cb_data, mu)

    # ---- single-node cut:  mu_i <= h(w_i, z_i) ----
    for i in nodes
        w0 = max(0.0, Z_bar_val[i] - Z_val[i])
        z0 = Z_val[i]
        h_val = gW(w0) * gT(z0)
        if h_val < mu_val[i] - 1e-5
            df_dw = gT(z0) * gWp(w0)
            df_dz = gW(w0) * gTp(z0)
            con = @build_constraint(
                mu[i] <= h_val
                       + df_dw * ((Z_bar[i] - Z[i]) - w0)
                       + df_dz * (Z[i] - z0)
            )
            MOI.submit(model, MOI.LazyConstraint(cb_data), con)
        end
    end

    # ---- pair cut (Eq. 30):  mu_i + mu_j <= Q_ij(t_i, tau_i + tau_j) ----
    # Only for arcs actually used in this candidate; B is the coupling term
    #   B = (Z_bar[i]-Z[i]) + (Z_bar[j]-Z[j]),   t_i = Z[i].
    for i in nodes, j in nodes
        i == j && continue
        callback_value(cb_data, y[(i,j)]) < 0.5 && continue   # not consecutive -> skip

        t_i = Z_val[i]
        w_i = max(0.0, Z_bar_val[i] - Z_val[i])
        w_j = max(0.0, Z_bar_val[j] - Z_val[j])
        B   = w_i + w_j

        Qval, dQ_dti, dQ_dB, s = solve_pair(t_i, B, t[(i,j)])

        if mu_val[i] + mu_val[j] > Qval + 1e-5
            # tangent of Q_ij in (Z[i], Z_bar[i], Z[j], Z_bar[j]);
            # chain rule via B = (Z_bar[i]-Z[i]) + (Z_bar[j]-Z[j]) and t_i = Z[i].
            con = @build_constraint(
                mu[i] + mu[j] <=
                    Qval
                    + dQ_dti * (Z[i] - t_i)
                    + dQ_dB  * ((Z_bar[i] - Z[i]) + (Z_bar[j] - Z[j]) - B)
                    + M_pair * (1 - y[(i,j)])
            )
            MOI.submit(model, MOI.LazyConstraint(cb_data), con)
            println("   → PAIR cut ($i,$j): μi+μj=",
                    round(mu_val[i] + mu_val[j], digits=4),
                    " > Q=", round(Qval, digits=4),
                    "  (split s*=", round(s, digits=2), ")")
        end
    end
end

MOI.set(model, MOI.LazyConstraintCallback(), plane_cut_callback)
optimize!(model)

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

    # ---- pair-cut diagnostics on the final route ----
    println("\nConsecutive-pair check (Eq. 30):")
    for i in nodes, j in nodes
        i == j && continue
        value(y[(i,j)]) > 0.5 || continue
        t_i = value(Z[i])
        B   = max(0.0, value(Z_bar[i]) - value(Z[i])) +
              max(0.0, value(Z_bar[j]) - value(Z[j]))
        Qval, _, _, s = solve_pair(t_i, B, t[(i,j)])
        println(" ($i→$j): μi+μj=",
                round(value(mu[i]) + value(mu[j]), digits=4),
                "  Q_ij=", round(Qval, digits=4),
                "  (B=", round(B, digits=2), ", best split s*=", round(s, digits=2), ")")
    end
end