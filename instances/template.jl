include("../algo.jl")

using JuMP, CPLEX, Distributions
using LinearAlgebra, SparseArrays

# Specify master problem and first stage annotation
# Do not solve
function build_stage_one()::Model
    model = Model(CPLEX.Optimizer)

    # Variables

    # Constraints

    # Objectives

    # First stage annotations
    model.ext[:first_stage_variables] = ...

    return model
end

# println(build_stage_one())
# println(build_stage_one.ext[:first_stage_variables])

# Initial starting point
initial_x = [0.0, 0.0]

# Initial lower bound
lower_bound = -72.0

# Get an optimal dual point of second stage given first stage x
# and observation
function get_dual_point(x, obs)::Vector{Float64}
    model = Model(CPLEX.Optimizer)

    @assert(termination_status(model) == OPTIMAL)
    return value.(p)
end

# calculate pi(r-Tx) for given dual point p, observation and first stage x
function evaluate_dual(p, obs, x)::Float64

end

# build cut, returns alpha, beta x
function build_cut(p, obs)::spOneCut
    rhs = [...]
    alpha = p' * rhs
    beta = [...]
    return spOneCut(alpha, beta)
end

# Generate a group of observation
function observe(rng::AbstractRNG)
    return [rand(rng, dist), rand(rng, dist)]
end

# Number of epigraphs
num_epigraph = 2

# Weight for each epigraph
epigraph_weights = [0.5, 0.5]

@assert(length(observe(Random.GLOBAL_RNG)) == num_epigraph)
@assert(length(epigraph_weights) = num_epigraph)

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