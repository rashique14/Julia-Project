using JuMP, Gurobi
import MathOptInterface as MOI
using Random

const GRB_ENV = Gurobi.Env()
new_model() = Model(() -> Gurobi.Optimizer(GRB_ENV))

Random.seed!(90)

const NODES  = 1:12
const NODES0 = 0:12
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

    @objective(m, Max, η)

    cuts = Ref(0)
    function benders_callback(cb_data)
        status = callback_node_status(cb_data, m)
        status == MOI.CALLBACK_NODE_STATUS_INTEGER || return

        route   = [a for a in ARCS if callback_value(cb_data, y[a]) > 0.5]
        claimed = callback_value(cb_data, η)
        state, v = route_reward(route)

        if state == :infeasible
            con = @build_constraint(sum(y[a] for a in route) <= length(route) - 1)
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
