mutable struct Leaf
    output      :: Float32
end

mutable struct Node
    # indices     :: Vector{Int}
    output      :: Float32
    isleaf      :: Bool
    splitpoint  :: Float32
    score       :: Float32
    left_node   ::  Node
    right_node  :: Node

    function Node(output, score)
        node = new()
        node.isleaf = false
        node.output = output
        node.score = score
        return node
    end
end
