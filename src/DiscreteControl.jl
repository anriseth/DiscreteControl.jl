module DiscreteControl
using Interpolations
using Optim, ForwardDiff
using Distributions

export DynamicSystem1D, DynamicSystemTrajectory1D, OfflineSystemControl1D,
    MPCSystem1D
export DynamicSystem
export initializestate!, step!, simulatetrajectories
export solvebellman!, onlinedecision, olfcdecision

include("types.jl")
include("bellman.jl")
include("mpc.jl")
include("olfc.jl")
include("api.jl")
include("utils.jl")

end # module
