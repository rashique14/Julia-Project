using JuMP, Gurobi
import MathOptInterface as MOI
using Random

# ======================================================================
# 0. ONE shared Gurobi environment  (prints the license banner just once)
# ======================================================================
const GRB_ENV = Gurobi.Env()
new_model() = Model(() -> Gurobi.Optimizer(GRB_ENV))

# ======================================================================
# 1. DATA  (unchanged)
# ======================================================================
Random.seed!(90)

const NODES     = 1:5
const NODES0    = 0:5
const T_END     = 30.0
const U_MAX     = 100.0
const DWELL_MAX = 20.0        # cap on wait/dwell at a node (was Zbar - Z <= 20)
const BIG_M     = 1.0e4       # big-M used to switch cuts off for other routes

coords = Dict(i => (10rand(), 10rand()) for i in NODES0)
ARCS   = [(i, j) for i in NODES0 for j in NODES0 if i != j]
travel = Dict((i, j) => hypot(coords[i][1] - coords[j][1],
                              coords[i][2] - coords[j][2]) for (i, j) in ARCS)
reward = Dict(i => 5.0 + 10rand() for i in NODES)
const TOTAL_REWARD = sum(values(reward))

# ======================================================================
# 2. NONLINEAR REWARD PIECES        h(wait, arrival) = f1(wait) * f2(arrival)
#      f1(wait)    = wait / (1 + wait)                  (grows with waiting)
#      f2(arrival) = 1 - 0.001 (T_END - arrival)^2      (grows with later arrival)
#
#   NEW SPLIT:
#     * `wait`  is a MASTER decision  (variable w[i]).
#     * `arrival` ("age") is tracked in the SUBPROBLEM.
# ======================================================================
f1(w)  = w / (1 + w)
f1p(w) = 1 / (1 + w)^2
f2(a)  = 1 - 0.001 * (T_END - a)^2
f2p(a) = 0.002 * (T_END - a)

# ======================================================================
# 3. SUBPROBLEM  —  AGE TRACKING for a FIXED route AND FIXED waits.
#
#    The master hands over (route, waits).  The only thing left to do is
#    propagate the arrival time ("age") along the chain, check that the
#    horizon still holds, and evaluate the reward together with its
#    supergradients w.r.t. the master's wait variables.
#
#    order :: Vector{Int}       visited nodes in tour order (depot implicit)
#    waits :: Dict(node=>value) wait time chosen by the master
#
#    returns  (:infeasible, 0.0, ., .)
#          or (:optimal,    V,  arr, grad)   with grad[v] = dV / dw[v]
# ======================================================================
function age_and_reward(order, waits)
    s = new_model(); set_silent(s)
    @variable(s, 0 <= A[order] <= T_END)          # A[i] = arrival time ("age")

    # forward propagation of age along the chain
    prev = 0
    for (p, v) in enumerate(order)
        pw = (prev == 0) ? 0.0 : waits[prev]      # depot carries no dwell
        @constraint(s, A[v] >= (p == 1 ? 0.0 : A[order[p-1]]) + pw + travel[(prev, v)])
        prev = v
    end
    # every departure and the final return must stay inside the horizon
    for v in order
        @constraint(s, A[v] + waits[v] <= T_END)
    end
    @constraint(s, A[order[end]] + waits[order[end]] + travel[(order[end], 0)] <= T_END)

    @objective(s, Min, sum(A[v] for v in order))  # earliest feasible arrivals
    optimize!(s)

    termination_status(s) == MOI.OPTIMAL ||
        return (:infeasible, 0.0, Dict{Int,Float64}(), Dict{Int,Float64}())

    arr = Dict(v => value(A[v]) for v in order)

    # reward value:  V = Σ reward[v] * f1(wait_v) * f2(age_v)
    V = sum(reward[v] * f1(waits[v]) * f2(arr[v]) for v in order)

    # total supergradient  dV / dw[v_r]:
    #   direct : reward_r f1'(w_r) f2(A_r)
    #   delay  : raising w_r pushes EVERY later arrival back by 1, so it
    #            adds  Σ_{p>r} reward_p f1(w_p) f2'(A_p)
    grad = Dict{Int,Float64}()
    tail = 0.0
    for p in length(order):-1:1
        v = order[p]
        grad[v] = reward[v] * f1p(waits[v]) * f2(arr[v]) + tail
        tail   += reward[v] * f1(waits[v]) * f2p(arr[v])
    end

    return (:optimal, V, arr, grad)
end

# ======================================================================
# 4. MASTER  —  ROUTING (y) *and* WAITING (w), solved once by Gurobi.
#    A lazy callback runs the age-tracking subproblem on every integer
#    route and returns feasibility / optimality cuts.
# ======================================================================
function build_and_solve()
    m = new_model(); set_silent(m)

    @variable(m, y[a in ARCS], Bin)                 # routing            (MASTER)
    @variable(m, 0 <= w[i in NODES] <= DWELL_MAX)   # wait / dwell time  (MASTER)
    @variable(m, 0 <= η <= TOTAL_REWARD)            # reward proxy

    # --- routing constraints (unchanged) ---
    @constraint(m, sum(y[(0, j)] for j in NODES) == 1)
    @constraint(m, sum(y[(i, 0)] for i in NODES) == 1)
    @constraint(m, [i in NODES],
        sum(y[(i, j)] for j in NODES0 if j != i) ==
        sum(y[(j, i)] for j in NODES0 if j != i))
    @constraint(m, [i in NODES],
        sum(y[(i, j)] for j in NODES0 if j != i) <= 1)

    # --- a node may only wait if it is actually visited ---
    @constraint(m, [i in NODES],
        w[i] <= DWELL_MAX * sum(y[(j, i)] for j in NODES0 if j != i))

    @objective(m, Max, η)

    cuts = Ref(0)
    function benders_callback(cb_data)
        callback_node_status(cb_data, m) == MOI.CALLBACK_NODE_STATUS_INTEGER || return

        # ---- rebuild the tour from the integer routing ----
        active = [a for a in ARCS if callback_value(cb_data, y[a]) > 0.5]
        succ   = Dict(i => j for (i, j) in active)

        order = Int[]; seen = Set{Int}()
        node  = get(succ, 0, nothing)
        while node !== nothing && node != 0 && !(node in seen)
            push!(order, node); push!(seen, node)
            node = get(succ, node, nothing)
        end

        # ---- subtour elimination: any active arc off the depot chain ----
        stray = [(i, j) for (i, j) in active if i != 0 && !(i in seen)]
        if !isempty(stray)
            con = @build_constraint(sum(y[a] for a in stray) <= length(stray) - 1)
            MOI.submit(m, MOI.LazyConstraint(cb_data), con); cuts[] += 1
            return
        end
        isempty(order) && return

        # ---- waits fixed by the master, then AGE TRACKING subproblem ----
        waits            = Dict(v => callback_value(cb_data, w[v]) for v in order)
        state, V, _, grd = age_and_reward(order, waits)
        route_travel     = sum(travel[a] for a in active)   # includes return leg

        if state == :infeasible
            # route + chosen waits overflow the horizon:
            #     Σ w[v] ≤ (T_END − travel(route))     when this route is on
            con = @build_constraint(
                sum(w[v] for v in order) <=
                (T_END - route_travel) + BIG_M * sum(1 - y[a] for a in active))
            MOI.submit(m, MOI.LazyConstraint(cb_data), con); cuts[] += 1
            return
        end

        # ---- optimality cut on η  (supergradient in the master waits) ----
        claimed = callback_value(cb_data, η)
        if claimed > V + 1e-6
            con = @build_constraint(
                η <= V
                   + sum(grd[v] * (w[v] - waits[v]) for v in order)
                   + BIG_M * sum(1 - y[a] for a in active))
            MOI.submit(m, MOI.LazyConstraint(cb_data), con); cuts[] += 1
        end
    end

    set_attribute(m, MOI.LazyConstraintCallback(), benders_callback)
    set_attribute(m, "TimeLimit", 120.0)     # safety stop for the demo
    optimize!(m)

    return m, y, w, η, cuts[]
end

# ======================================================================
# 5. RUN
# ======================================================================
function main()
    m, y, w, η, ncuts = build_and_solve()

    println("\n=== Final Solution ===")
    println("Status            : ", termination_status(m))
    println("Benders cuts added: ", ncuts)

    if primal_status(m) == MOI.FEASIBLE_POINT
        println("Objective (reward): ", round(objective_value(m), digits=4))

        active = [a for a in ARCS if value(y[a]) > 0.5]
        succ   = Dict(i => j for (i, j) in active)
        println("Route & schedule:")
        prev = 0; t = 0.0; node = get(succ, 0, nothing); seen = Set{Int}()
        while node !== nothing && node != 0 && !(node in seen)
            push!(seen, node)
            t += travel[(prev, node)]                       # arrival ("age")
            wv = value(w[node])
            println("  ", prev, " -> ", node,
                    "   arrive ", round(t, digits=2),
                    "   wait ",   round(wv, digits=2),
                    "   depart ", round(t + wv, digits=2))
            t   += wv
            prev = node
            node = get(succ, node, nothing)
        end
        t += travel[(prev, 0)]
        println("  ", prev, " -> 0   return ", round(t, digits=2))
    else
        println("No feasible route found.")
    end
end

main()