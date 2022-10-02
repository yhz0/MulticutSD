using JuMP

# Represent a cut or an affine function
struct spOneCut
    alpha::Float64
    beta::Vector{Float64}
end

# Evaluate point at specified cut
function evaluate_cut(cut::spOneCut, x::Vector{Float64})::Float64
    return cut.alpha + cut.beta' * x
end

# Evaluate the pointwise max of cut pool at x.
# If the pool is empty, then return lb.
function evaluate_epigraph(cuts::Vector{spOneCut}, x::Vector{Float64})
    if isempty(cuts)
        return -Inf
    end
    return maximum([evaluate_cut(cut, x) for cut in cuts])
end

# Add the epigraph variable and epigraph constraint represented
# into the model. Assuming the variable is named x.
# Returning the constraints added.
# The vectors marked in disable will not be added.
# Returning a vector of pairs: index of added constraint =>
# the corresponding index in the cut_pool.
function add_pool!(
    model::Model,
    cut_pool::Vector{spOneCut},
    disable::Union{Vector{Int}, Set{Int}},
    xref::Vector{VariableRef},
    weight::Float64
    )::Vector{Pair{ConstraintRef, Int}}

    cons = Pair{ConstraintRef, Int}[]
    theta = @variable(model)

    for i in eachindex(cut_pool)
        if !(i in disable)
            con = @constraint(model, theta >= cut_pool[i].alpha + cut_pool[i].beta' * xref)
            push!(cons, con => i)
        end
    end
    set_objective_function(model, objective_function(model) + weight * theta)

    return cons
end
