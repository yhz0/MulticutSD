using Random

struct spConfig
    MAX_ITER::Int
    MIN_QUAD_SCALAR::Float64
    MAX_QUAD_SCALAR::Float64
    INCUMBENT_IMPROVEMENT::Float64
    R2::Float64
    R3::Float64
    RNG::AbstractRNG
end

const DEFAULT_CONFIG = spConfig(1000, 0.001, 10000.0, 0.95, 2.0, 0.2, Random.GLOBAL_RNG)