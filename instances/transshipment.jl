include("../algo.jl")

using JuMP, CPLEX, Distributions
using LinearAlgebra, SparseArrays

const N = 7
const h = 0.5
const c = 1.0
const p = 4.0

# Specify master problem and first stage annotation
# Do not solve
function build_stage_one()::Model
    model = Model(CPLEX.Optimizer)

    # First stage variables
    @variable(model, s[1:N] >= 0)
    @objective(model, Min, 0)

    # First stage annotations
    model.ext[:first_stage_variables] = s
    set_silent(model)

    return model
end

# begin
#     master = build_stage_one()
#     println(master)
#     println(master.ext[:first_stage_variables])
# end

# Initial starting point
initial_x = zeros(N)

# Initial lower bound
lower_bound = 0.0

# Get an optimal dual point of second stage given first stage x
# and observation
# s: first stage decision
# d: demand realization
function get_dual_point(s, d)::Vector{Float64}
    @assert(length(s) == N)
    @assert(length(d) == N)

    model = Model(CPLEX.Optimizer)

    @variables(model, begin
       B[1:N]
       M[1:N]
       R
       E[1:N] 
    end)

    @constraints(model, begin
        r1[i=1:N], B[i]+E[i] <= h
        r2[i=1:N], B[i]+M[i] <= 0
        r3[(i, j) = ((i, j) for i = 1:N for j = 1:N if i != j)], B[i]+M[j]<=c
        r4[i=1:N], M[i]+R <= p
        r5[i=1:N], R+E[i] <= 0
    end)

    @objective(
        model, Max,
        sum(s[i]*B[i] + d[i]*M[i] + d[i]*R + s[i]*E[i] for i = 1:N)
    )
    
    set_silent(model)
    optimize!(model)
    @assert(termination_status(model) == OPTIMAL)
    # @info(objective_value(model))
    
    dual_point::Vector{Float64} = vcat(value.(B), value.(M), value(R), value.(E))
    return dual_point
end

# calculate pi(r-Tx) for given dual point p, observation and first stage x
function evaluate_dual(p, d, s)::Float64
    dual_cost = vcat(s, d, sum(d), s)
    dual_obj = p' * dual_cost
    return dual_obj
end

# build cut, returns alpha, beta x
function build_cut(p, d)::spOneCut
    B::Vector{Float64} = p[1:N]
    M::Vector{Float64} = p[(N+1):(2*N)]
    R::Float64 = p[2*N+1]
    E::Vector{Float64} = p[(2*N+2):(3*N+1)]

    alpha = sum(d[i] * M[i] + d[i] * R for i = 1:N)
    beta = B+E
    return spOneCut(alpha, beta)
end

# Test
# s0 = [100., 100, 100, 100, 100, 100, 100]
# dp = get_dual_point(s0, s0.-1.0)
# build_cut(dp, s0 .-1.0)

# Distribution:
const mu = [100.0, 200, 150, 170, 180, 170, 170]
const sigma = [20.0, 50, 30, 50, 40, 30, 50]

# IMPORTANT: We truncate the demand at plus or minus 3 sigmas to avoid negative demands
const d_dist = product_distribution(
    truncated.(Normal.(mu, sigma), mu - 3*sigma, mu + 3*sigma)
)

# Generate a group of observation
function observe(rng::AbstractRNG)
    u = rand(rng, d_dist)
    v = rand(rng, d_dist)
    # # Antithetic
    # diff = u - mu
    # v = mu - diff
    return [u, v]
end

# Number of epigraphs
num_epigraph = 2

# Weight for each epigraph
epigraph_weights = [0.5, 0.5]

@assert(length(observe(Random.GLOBAL_RNG)) == num_epigraph)
@assert(length(epigraph_weights) == num_epigraph)

prob = spProblem(
    initial_x,
    build_stage_one,
    get_dual_point,
    build_cut,
    evaluate_dual,
    observe,            # Sample Group Generator
    num_epigraph,       # Number of epigraphs
    epigraph_weights,   # Epigraph weights
    lower_bound         # Assumed lower bounds
)

run_sd(prob)