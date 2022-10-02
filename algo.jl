include("cut.jl")
include("config.jl")
include("utils.jl")

# Add regularization term
# Must be performed last to ensure the results correct.
function add_regularization!(model::Model, rho::Float64, x_cur::Vector{Float64}, xref::Vector{VariableRef})
    @assert(length(x_cur) == length(xref))

    # Record original obj expression
    model.ext[:original_obj] = objective_function(model)

    set_objective_function(model,
        objective_function(model) + rho / 2 * sum((xref - x_cur).^2)
    )

    return
end

# incumbent selection, return true if x_candidate replaces x_incumb
# incumb_frac is R1: reduction in cost needed to replace incumbent
# c is the cost vector
# Returns a tuple (accept, new_rho)
# accept: Bool true if the pool should be replaced
# new_rho: the next rho
function incumbent_selection(
    x_candidate::Vector{Float64},
    x_incumb::Vector{Float64},
    cost_vector::Vector{Float64},
    cut_pools::Vector{Vector{spOneCut}},
    old_cut_pools::Vector{Vector{spOneCut}},
    pool_weights::Vector{Float64},
    current_rho::Vector{Float64};
    config::spConfig)::Tuple{Bool, Float64}

    @assert(length(x_candidate) == length(x_incumb) == length(cost_vector))
    @assert(length(cut_pools) == length(old_cut_pools) == length(pool_weights))

    r1, r2, r3, r4 = 0.0, 0.0, 0.0, 0.0
    for (cut_pool, old_cut_pool, weight) in zip(cut_pools, old_cut_pools, pool_weights)
        r1 += weight * (evaluate_epigraph(cut_pool, x_candidate))
        r2 += weight * (evaluate_epigraph(cut_pool, x_incumb))
        r3 += weight * (evaluate_epigraph(old_cut_pool, x_candidate))
        r4 += weight * (evaluate_epigraph(old_cut_pool, x_incumb))
    end

    # Add the x part
    r1 += (cost_vector' * x_candidate)
    r2 += (cost_vector' * x_incumb)
    r3 += (cost_vector' * x_candidate)
    r4 += (cost_vector' * x_incumb)

    L = r1 - r2
    R = r3 - r4
    accept::Bool = L < config.INCUMBENT_IMPROVEMENT * R

    # Now calculate current scalar rho
    if L < config.R2 * R

    end
    
    return accept
end

function argmax_procedure(
    dual_vertices::Union{Set{Vector{Float64}}, Vector{Vector{Float64}}},
    obs,
    x::Vector{Float64},
    build_cut::Function)::spOneCut

    cuts = Vector{spOneCut}()
    for omega in obs
        # argmax procedure at x
        args = [p => evaluate_dual(p, omega, x) for p in dual_vertices]

        max_pair = argmax(x->x.second, args)
        p_opt = max_pair.first

        cut = build_cut(p_opt, omega)
        push!(cuts, cut)
    end

    return aggregate_cuts(cuts)
end

function aggregate_cuts(cuts::Vector{spOneCut})
    N = length(cuts)
    a = sum(cut.alpha for cut in cuts) / N
    b = sum(cut.beta for cut in cuts) / N
    return spOneCut(a, b)
end

struct spProblem
    x_init::Vector{Float64}

    build_stage_one::Function # () -> first stage model with first stage variables annotated in model.ext[:first_stage_variables]
    get_dual_point::Function # (x, obs) -> second stage dual vertex
    build_cut::Function # (p, omega) -> spOneCut of dual cut
    evaluate_dual::Function # (p, omega, x) -> second stage value function

    sample_generator::Function # rng -> Vector of sample groups
    num_epigraph::Int
    epigraph_weights::Vector{Float64}

    lower_bound::Float64
end

function run_sd(prob::spProblem; config::spConfig=DEFAULT_CONFIG)

    # Initialization
    x_candidate::Vector{Float64} = copy(prob.x_init)
    x_incumb::Vector{Float64} = copy(prob.x_init)

    dual_vertices = Set{Vector{Float64}}()

    incumb_iter::Int = 1

    # Initialize the sample pools, cut pools
    obs = []
    cut_pools::Vector{Vector{spOneCut}} = []
    last_cut_pools::Vector{Vector{spOneCut}} = []

    for _ = 1:prob.num_epigraph
        push!(obs, [])
        push!(cut_pools, [])
        push!(last_cut_pools, [])
    end
    
    # which constraints are left out
    disabled_cuts = Set{Int}()

    # initial regularization strength
    rho = 0.1

    # outputs
    history = []

    for k = 1:config.MAX_ITER
        @info " ====== Iteration $k"

        # Generate omega
        omega = prob.sample_generator(config.RNG)
        @assert(length(omega) == prob.num_epigraph)

        for i = 1:prob.num_epigraph
            push!(obs[i], omega[i])
        end

        # @info "omega = $omega"

        # Solve subprob to discover dual ext pt
        pi_candidate = []
        pi_incumb = []
        for i = 1:prob.num_epigraph
            push!(pi_candidate, prob.get_dual_point(x_candidate, omega[i]))
            push!(pi_incumb, prob.get_dual_point(x_incumb, omega[i]))
            push!(dual_vertices, pi_candidate[i])
            push!(dual_vertices, pi_incumb[i])
        end

        # Shrink all the cuts first. The new cut will be added
        # and the incumbent cut will be updated.
        for cut_pool in cut_pools
            for i in eachindex(cut_pool)
                if !(i in disabled_cuts)
                    new_alpha = (k/(k+1)) * cut_pool[i].alpha + (prob.lower_bound/(k+1))
                    new_beta = (k/(k+1)) * cut_pool[i].beta
                    cut_pool[i] = spOneCut(new_alpha, new_beta)
                end
            end
        end

        for i = 1:prob.num_epigraph
            # argmax procedure to build the new cut
            new_cut = argmax_procedure(dual_vertices, obs[i], x_candidate, prob.build_cut)
            push!(cut_pools[i], new_cut)

            # update the incumbent cut: assuming no cut can be deleted, but disabled instead
            incumb_cut = argmax_procedure(dual_vertices, obs[i], x_incumb, prob.build_cut)
            cut_pools[i][incumb_iter] = incumb_cut
        end

        # Build Master Program
        # add regularization and epigraph variables
        master = prob.build_stage_one()

        # Get the user specified first stage variables reference
        @assert(:first_stage_variables in keys(master.ext),
            ":first_stage_variables not found in master.ext[]. Please annotate first stage variables.")
        xref = master.ext[:first_stage_variables]

        cost_vector = get_cost_coefficient(master)

        # add the cuts into master, getting the cuts => index
        # skipping the disabled cuts
        all_cuts = []
        for i = 1:prob.num_epigraph
            crefs = add_pool!(master, cut_pools[i], disabled_cuts, xref, prob.epigraph_weights[i])
            push!(all_cuts, crefs)
        end

        # Add regularization term
        add_regularization!(master, rho, x_incumb, xref)
        optimize!(master)

        @assert(termination_status(master) == OPTIMAL, "master solver error: $(termination_status(master))")

        @show objective_value(master)
        # @show value(master.ext[:original_obj])

        # Now we can identify which cuts are disabled and remove them
        # by setting a "disabled" flag
        for j in eachindex(all_cuts[1])
            # all_cuts[i][j] contains the jth cut in ith pool
            # we poll the cuts generated from each iteration
            cut = [all_cuts[i][j] for i = 1:prob.num_epigraph]

            # If cut is disabled, skip.
            # Note They are either all disabled or all enabled
            if cut[1].second in disabled_cuts
                continue
            end

            # set remove_flag if all slack for specific instance
            remove_flag = all(
                value(cut[i].first) != normalized_rhs(cut[i].first) for i in 1:prob.num_epigraph
            )

            # If remove_flag (slackness) and not incumbent and not current Iteration
            # then we are safe to remove that cut.
            if remove_flag &&
                !(cut[1].second == k) && !(cut[1].second == incumb_iter)
                @info "Disabling cut $(cut[1].second)."
                push!(disabled_cuts, cut[1].second)
            end
        end
        @info "Cut count in each pool: $(length(cut_pools[1]) - length(disabled_cuts))"

        x_candidate = value.(xref)

        # test candidate
        if incumbent_selection(x_candidate, x_incumb, cost_vector, cut_pools, last_cut_pools, prob.epigraph_weights; config=config)
            # replace incumbent
            x_incumb = copy(x_candidate)
            incumb_iter = k
            @info "Replaced incumbent"
        end

        @info "x_candidate: $x_candidate"
        @info "x_incumb ($incumb_iter): $x_incumb"

        # Retain the cut pool
        for i = 1:prob.num_epigraph
            last_cut_pools[i] = copy(cut_pools[i])
        end

        # push record stuff
        # push!(history, value(master.ext[:original_obj]))

    end
    return history
end
