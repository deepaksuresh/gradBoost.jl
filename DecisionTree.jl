struct Leaf{T}
    logOdds     :: T
    # probability :: T
    function Leaf(logOdds)
        # probability = 1/(1 + exp(-logOdds))
        # return new(logOdds, probability)
        return new(logOdds)
    end
end

struct Node{S, T}
    featid  :: Int
    featval :: S
    left    :: Union{Leaf{T}, Node{S, T}}
    right   :: Union{Leaf{T}, Node{S, T}}
end

const LeafOrNode{S, T} = Union{Leaf{T}, Node{S, T}}

struct Ensemble{S, T}
    trees :: Vector{LeafOrNode{S, T}}
end

is_leaf(l::Leaf) = true
is_leaf(n::Node) = false

##############################
########## Includes ##########
#
# include("measures.jl")
# include("load_data.jl")
# include("util.jl")
# include("classification/main.jl")
# include("regression/main.jl")
# include("scikitlearnAPI.jl")


#############################
########## Methods ##########

length(leaf::Leaf) = 1
length(tree::Node) = length(tree.left) + length(tree.right)
length(ensemble::Ensemble) = length(ensemble.trees)

depth(leaf::Leaf) = 0
depth(tree::Node) = 1 + max(depth(tree.left), depth(tree.right))
