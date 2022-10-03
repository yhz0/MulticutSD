using Random

struct spConfig
    MAX_ITER::Int
    MIN_QUAD_SCALAR::Float64
    MAX_QUAD_SCALAR::Float64
    INITIAL_QUAD_SCALAR::Float64
    INCUMBENT_IMPROVEMENT::Float64      # R1
    R2::Float64
    R3::Float64
    RNG::AbstractRNG
end

const DEFAULT_CONFIG = spConfig(200, 0.0001, 10000.0, 0.01, 0.2, 0.95, 2.0, Random.GLOBAL_RNG)