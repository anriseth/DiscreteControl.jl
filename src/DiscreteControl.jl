module DiscreteControl
using Interpolations
using Optim, ForwardDiff
using Distributions

export DynamicSystem1D, DynamicSystemTrajectory1D, OfflineSystemControl1D,
    MPCSystem1D
export initializestate!, step!
export solvebellman!, onlinedecision

include("types.jl")
include("bellman.jl")
include("mpc.jl")
include("api.jl")

end # module
