using JuMP

# get cost coefficient
function get_cost_coefficient(model::Model)
    @assert(:first_stage_variables in keys(model.ext), ":first_stage_variables not found in model.ext.")
    vars = model.ext[:first_stage_variables]
    @assert(length(vars) == length(all_variables(model)), "first stage variable length do not match")

    f = objective_function(model)

    terms = f.terms
    c = zeros(length(vars))
    for (i, v) in enumerate(vars)
        if v in keys(terms)
            c[i] = terms[v]
        end
    end
    return c
end

