include("cut.jl")
include("utils.jl")

# Some unit tests
using Test

# Build a dummy first stage model
model = Model()
@variable(model, x[1:2] >= 10)
@objective(model, Min, x[1] + 2 * x[2])
cut = spOneCut(0.1, [0.2, 0.3])
model.ext[:first_stage_variables] = x

# Test cost coefficients extraction
@test(get_cost_coefficient(model) == [1.0, 2.0])

# Test f cuts are added successfully
add_pool!(model, [cut], Int[], x, 3.0)
@test(length(all_variables(model)) > 2)
