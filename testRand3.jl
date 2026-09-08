using JuMP, Gurobi
import MathOptInterface as MOI
using Random

const GRB_ENV = Gurobi.Env()
new_model() = Model(() -> Gurobi.Optimizer(GRB_ENV))

Random.seed!(90)

const N      =    8             # <- set to 15 for the larger instance
const NODES  = 1:N
const NODES0 = 0:N
const T_END  = 30.0
const U_MAX  = 100.0

const coords = Dict(i => (10rand(), 10rand()) for i in NODES0)
const ARCS   = [(i, j) for i in NODES0 for j in NODES0 if i != j]
const travel = Dict((i, j) => hypot(coords[i][1] - coords[j][1],
                                    coords[i][2] - coords[j][2]) for (i, j) in ARCS)
const reward = Dict(i => 5.0 + 10rand() for i in NODES)
const TOTAL_REWARD = sum(values(reward))

function reward_value_and_gradient(arrival, departure)
    wait = max(0.0, departure - arrival)
    f1   = wait / (1 + wait)
    f2   = 1 - 0.001 * (T_END - arrival)^2
    h        = f1 * f2
    dh_dwait = f2 / (1 + wait)^2
    dh_darr  = f1 * 0.002 * (T_END - arrival)
    return (h = h, dh_dwait = dh_dwait, dh_darr = dh_darr, wait = wait)
end

function make_subproblem()
    s = new_model(); set_silent(s)
    @variable(s, Z[NODES0]    >= 0)
    @variable(s, Zbar[NODES0] >= 0)
    @variable(s, 0 <= μ[i in NODES] <= 1)

    @constraint(s, Zbar[0] == 0)
    @constraint(s, Z[0] <= U_MAX)
    @constraint(s, [i in NODES], Z[i]           <= Zbar[i])
    @constraint(s, [i in NODES], Zbar[i]        <= T_END)
    @constraint(s, [i in NODES], Zbar[i] - Z[i] <= 20)
    @constraint(s, [i in NODES], Z[i]           <= T_END)

    @objective(s, Max, sum(reward[i] * μ[i] for i in NODES))
    return s, Z, Zbar, μ
end

const SUB, SUB_Z, SUB_Zbar, SUB_μ = make_subproblem()
const SUB_DYN = Any[]

function route_reward(active_arcs)
    s = SUB
    Z, Zbar, μ = SUB_Z, SUB_Zbar, SUB_μ

    for c in SUB_DYN
        delete(s, c)
    end
    empty!(SUB_DYN)

    entered = Dict(i => any(j == i for (_, j) in active_arcs) for i in NODES)

    for i in NODES
        set_upper_bound(μ[i], entered[i] ? 1.0 : 0.0)
    end

    for (i, j) in active_arcs
        if j != 0
            push!(SUB_DYN, @constraint(s, Z[j] >= Zbar[i] + travel[(i, j)]))
        else
            push!(SUB_DYN, @constraint(s, Z[0] >= Zbar[i] + travel[(i, 0)]))
        end
    end

    for _ in 1:50
        optimize!(s)
        termination_status(s) == MOI.OPTIMAL || return (:infeasible, 0.0)

        arr = Dict(i => value(Z[i])    for i in NODES if entered[i])
        dep = Dict(i => value(Zbar[i]) for i in NODES if entered[i])
        mv  = Dict(i => value(μ[i])    for i in NODES if entered[i])

        added = false
        for i in NODES
            entered[i] || continue
            g = reward_value_and_gradient(arr[i], dep[i])
            if mv[i] > g.h + 1e-6
                push!(SUB_DYN, @constraint(s, μ[i] <= g.h
                    + g.dh_dwait * ((Zbar[i] - Z[i]) - g.wait)
                    + g.dh_darr  * (Z[i] - arr[i])))
                added = true
            end
        end
        added || break
    end
    return (:optimal, objective_value(s))
end

# ---------------------------------------------------------------------------
# Helpers for the logic cuts (single-threaded use of the shared SUB model).
# ---------------------------------------------------------------------------

# Successor map of the selected arcs (each node has out-degree <= 1).
_succ(route) = Dict{Int,Int}(i => j for (i, j) in route)

# Arcs of the depot tour, in visiting order 0 -> ... -> 0.
function depot_tour_arcs(route)
    succ = _succ(route)
    seq  = Tuple{Int,Int}[]
    cur  = 0
    for _ in 0:length(NODES0)
        haskey(succ, cur) || break
        nxt = succ[cur]
        push!(seq, (cur, nxt))
        cur = nxt
        cur == 0 && break
    end
    return seq
end

# Cycles that do NOT contain the depot (each returned as a node set).
function subtours(route)
    succ = _succ(route)
    seen = Set{Int}([0])
    for (i, _) in depot_tour_arcs(route)
        push!(seen, i); push!(seen, succ[i])
    end
    tours = Vector{Set{Int}}()
    for (i, _) in route
        i in seen && continue
        comp = Set{Int}()
        c = i
        for _ in 0:length(NODES0)
            (c in comp) && break
            push!(comp, c); push!(seen, c)
            c = succ[c]
        end
        push!(tours, comp)
    end
    return tours
end

# Minimal prefix 0 -> ... -> v of the depot tour whose earliest possible
# arrival at v already exceeds T_END (=> scheduling subproblem infeasible).
function infeasible_prefix(route)
    seq  = depot_tour_arcs(route)
    c    = 0.0
    pref = Tuple{Int,Int}[]
    for (i, j) in seq
        c += travel[(i, j)]
        push!(pref, (i, j))
        (j != 0 && c > T_END + 1e-9) && return pref
    end
    return pref   # fallback == the old no-good cut (rare; still valid)
end

function build_and_solve()
    m = new_model(); set_silent(m)

    @variable(m, y[a in ARCS], Bin)
    @variable(m, 0 <= η <= TOTAL_REWARD)

    @constraint(m, sum(y[(0, j)] for j in NODES) == 1)
    @constraint(m, sum(y[(i, 0)] for i in NODES) == 1)
    @constraint(m, [i in NODES],
        sum(y[(i, j)] for j in NODES0 if j != i) ==
        sum(y[(j, i)] for j in NODES0 if j != i))
    @constraint(m, [i in NODES],
        sum(y[(i, j)] for j in NODES0 if j != i) <= 1)

    # (L1) 2-cycle elimination: i -> j -> i (i,j != depot) is always an
    #      infeasible subtour, so forbid it statically.
    @constraint(m, [i in NODES, j in NODES; i < j],
        y[(i, j)] + y[(j, i)] <= 1)

    # (L2) Reward bound: μ_i <= 1 and μ_i = 0 unless i is entered, so collected
    #      reward can never exceed the reward of the visited nodes. Ties η to
    #      the arcs directly instead of relying only on per-route Benders cuts.
    @constraint(m, η <=
        sum(reward[i] * sum(y[(j, i)] for j in NODES0 if j != i) for i in NODES))

    # (L3) Reachability preprocessing: with Euclidean (triangle-inequality)
    #      travel, the earliest arrival at i is travel[0,i]; if that already
    #      exceeds T_END, i can never be served.
    for i in NODES
        if travel[(0, i)] > T_END
            for j in NODES0
                j == i && continue
                fix(y[(i, j)], 0; force = true)
                fix(y[(j, i)], 0; force = true)
            end
        end
    end

    @objective(m, Max, η)

    cuts = Ref(0)
    function benders_callback(cb_data)
        callback_node_status(cb_data, m) == MOI.CALLBACK_NODE_STATUS_INTEGER || return
        route = [a for a in ARCS if callback_value(cb_data, y[a]) > 0.5]

        # (L4) Subtour elimination (lazy). Cut any cycle not through the depot
        #      with a DFJ constraint -- far stronger than a no-good cut.
        st = subtours(route)
        if !isempty(st)
            for S in st
                con = @build_constraint(
                    sum(y[(i, j)] for i in S for j in S if i != j) <= length(S) - 1)
                MOI.submit(m, MOI.LazyConstraint(cb_data), con)
                cuts[] += 1
            end
            return
        end

        # Connected depot tour: evaluate the true (concave) reward.
        claimed = callback_value(cb_data, η)
        state, v = route_reward(route)

        if state == :infeasible
            # (L5) Infeasible-path cut: forbid the minimal time-infeasible
            #      prefix instead of the whole route.
            pref = infeasible_prefix(route)
            con  = @build_constraint(sum(y[a] for a in pref) <= length(pref) - 1)
            MOI.submit(m, MOI.LazyConstraint(cb_data), con)
            cuts[] += 1
        elseif claimed > v + 1e-6
            con = @build_constraint(
                η <= v + TOTAL_REWARD * sum(1 - y[a] for a in route))
            MOI.submit(m, MOI.LazyConstraint(cb_data), con)
            cuts[] += 1
        end
    end

    set_attribute(m, MOI.LazyConstraintCallback(), benders_callback)

    # Keep Threads=1: route_reward() reuses one global SUB model and is NOT
    # thread-safe. To parallelize, give each thread its own subproblem.
    set_attribute(m, "Threads", 1)
    set_attribute(m, "TimeLimit", 3600.0)
    optimize!(m)

    return m, y, η, cuts[]
end

function main()
    m, y, η, ncuts = build_and_solve()

    println("\n=== Final Solution ===")
    println("Status     : ", termination_status(m))
    println("Benders cuts added: ", ncuts)

    if primal_status(m) == MOI.FEASIBLE_POINT
        println("Objective  : ", round(objective_value(m), digits=4))
        println("Bound      : ", round(objective_bound(m), digits=4))
        println("Gap        : ", round(100 * relative_gap(m), digits=4), " %")
        println("Route:")
        for a in ARCS
            value(y[a]) > 0.5 && println("  $(a[1]) -> $(a[2])")
        end
    else
        println("No feasible route found.")
        println("Bound      : ", round(objective_bound(m), digits=4))
    end
end

main()