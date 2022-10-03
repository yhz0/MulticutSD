include("../algo.jl")

using JuMP, CPLEX, Distributions
using LinearAlgebra, SparseArrays

a1 = 0.046
a2 = 0.184
b = 3.147
sigma = 0.39
dist = Normal(0, sigma)

function build_stage_one()
    model = Model(CPLEX.Optimizer)

    @variable(model, x[1:2])
    set_lower_bound.(x, [0.7, 0.0])
    set_upper_bound.(x, [200.0, 50.0])

    @constraint(model, c1, x[1] + x[2] <= 200)
    @constraint(model, c2, x[1] - 0.5*x[2] >= 0)
    @objective(model, Min, 0.1*x[1] + 0.5*x[2])

    model.ext[:first_stage_variables] = x

    set_silent(model)
    return model
end

# Get a dual vertex
function get_dual_point(x, noise)
    A = [
        1 0
        0 2
        3 2
        1 1
    ]
    rhs = [8, 24, 36, a1*x[1] + a2*x[2] + b + noise]
    c = [-3, -5]
    model = Model(CPLEX.Optimizer)
    @variable(model, p[1:4] <= 0)
    @objective(model, Max, rhs' * p)
    @constraint(model, A' * p .<= c)
    set_silent(model)
    optimize!(model)
    @assert(termination_status(model) == OPTIMAL)
    return value.(p)
end

# calculate pi(r-Tx), used in argmax
function evaluate_dual(p, noise, x)
    rhs = [8.0, 24.0, 36.0, b+noise+a1*x[1]+a2*x[2]]
    return p' *rhs
end

# build cut, returns alpha, beta x
function build_leo_wyndor_cut(p, noise)::spOneCut
    rhs = [8.0, 24.0, 36.0, b+noise]
    alpha = p' * rhs
    beta = [p[4]*a1, p[4]*a2]
    return spOneCut(alpha, beta)
end

# function leo_wyndor_iid(rng)
#     return [rand(rng, dist), rand(rng, dist)]
# end

# prob = spProblem(
#     [0.0, 0.0],
#     build_stage_one,
#     get_dual_point,
#     build_leo_wyndor_cut,
#     evaluate_dual,
#     leo_wyndor_iid,     # Sample Group Generator
#     2,                  # Number of epigraphs
#     [0.5, 0.5],         # Epigraph weights
#     -72.0               # Assumed lower bounds
# )
hist = run_sd(prob)

# Print estm obj
using DataFrames, Plots, StatsPlots
result = DataFrame(hist)
@df result plot(:estimated_obj)