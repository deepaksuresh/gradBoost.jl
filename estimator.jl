mutable struct Leaf
    output      :: Float32
end

mutable struct Node
    # indices     :: Vector{Int}
    output      :: Float32
    isleaf      :: Bool
    splitpoint  :: Float32
    score       :: Float32
    left_node   ::  Union{Leaf, Node}
    right_node  :: Union{Leaf, Node}
    split_feat  :: Int32

    function Node(output, score)
        node = new()
        node.isleaf = false
        node.output = output
        node.score = score
        return node
    end
end

LeafOrNode = Union{Leaf, Node}

mutable struct Tree
    nodes :: Vector{LeafOrNode}
    max_depth :: Int32
    #init with max_depth, num_boosters....
    function Tree(node, max_depth)
        tree = new()
        tree.nodes = [node]
        tree.max_depth = max_depth
        return tree
    end
end
