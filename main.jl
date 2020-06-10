# Utilities

include("tree.jl")


function _convert(
        node   :: treeclassifier.NodeMeta{S},
        list   :: Vector{T},
        labels :: Vector{T}) where {S, T}

    if node.is_leaf
        return Leaf{T}(list[node.label], labels[node.region])
    else
        left = _convert(node.l, list, labels)
        right = _convert(node.r, list, labels)
        return Node{S, T}(node.feature, node.threshold, left, right)
    end
end

################################################################################


# build_booster(labels, features, residuals, previous_prediction

function build_booster(
        labels      :: Vector{T},
        features    :: Matrix{S};
        residuals      = nothing,
        previous_prediction = nothing,
        rng          = Random.GLOBAL_RNG) where {S, T}

        gradients =  residuals
        hessians = previous_prediction .* (1 .- previous_prediction)

    t = treeclassifier.fit(
        X                   = features,
        Y                   = labels,
        W                   = weights,
        loss                = treeclassifier.util.zero_one,
        max_features        = size(features, 2),
        max_depth           = 1,
        min_samples_leaf    = 1,
        min_samples_split   = 2,
        min_purity_increase = 0.0;
        rng                 = rng)

    return _convert(t.root, t.list, labels[t.labels])
end


apply_booster(leaf::Leaf{T}, feature::Vector{S}) where {S, T} = leaf.logOdds

function apply_tree(tree::Node{S, T}, features::Vector{S}) where {S, T}
    if features[tree.featid] < tree.featval
        return apply_tree(tree.left, features)
    else
        return apply_tree(tree.right, features)
    end
end

function apply_tree(tree::LeafOrNode{S, T}, features::Matrix{S}) where {S, T}
    N = size(features,1)
    predictions = Array{T}(undef, N)
    for i in 1:N
        predictions[i] = apply_tree(tree, features[i, :])
    end
    if T <: Float64
        return Float64.(predictions)
    else
        return predictions
    end
end


function build_gradBoosters(
        labels       :: Vector{T},
        features     :: Matrix{S},
        n_iterations :: Integer;
        base_score    = 0.5,
        eta           = 0.3,
        rng           = Random.GLOBAL_RNG) where {S, T}
    N = length(labels)
    # weights = ones(N) / N
    residuals = labels .- base_score
    base_leaf = Leaf(base_score)
    previous_prediction = fill(base_score, size(labels))
    # stumps = Node{S, T}[]
    boosters = Node{S, T}[base_leaf]
    # coeffs = Float64[]
    for i in 1:n_iterations
        new_booster = build_booster(labels, features, residuals, previous_prediction; rng=rng)
        predictions = apply_gradboost_stumps(new_booster, features)
        residuals = labels - predictions
        # err = _weighted_error(labels, predictions, weights)
        # new_coeff = 0.5 * log((1.0 + err) / (1.0 - err))
        # matches = labels .== predictions
        # weights[(!).(matches)] *= exp(new_coeff)
        # weights[matches] *= exp(-new_coeff)
        # weights /= sum(weights)
        # push!(coeffs, new_coeff)
        # push!(stumps, new_stump)
        # if err < 1e-6
        #     break
        # end
    end
    return Ensemble{S, T}(boosters)
end


function apply_gradboost_stumps(stumps::Ensemble{S, T}, coeffs::Vector{Float64}, features::Vector{S}) where {S, T}
    n_stumps = length(stumps)
    counts = Dict()
    prediction = 0.0
    for i in 1:n_stumps
        prediction += apply_tree(stumps.trees[i], features)
    end
    return prediction
end

function apply_gradboost_stumps(stumps::Ensemble{S, T}, coeffs::Vector{Float64}, features::Matrix{S}) where {S, T}
    n_samples = size(features, 1)
    predictions = Array{T}(undef, n_samples)
    for i in 1:n_samples
        predictions[i] = apply_gradboost_stumps(stumps, coeffs, features[i,:])
    end
    return predictions
end

function apply_gradboost_stumps(stumps::Ensemble{S, T}, coeffs::Vector{Float64}, features::Matrix{S}) where {S, T}
    n_samples = size(features, 1)
    predictions = Array{T}(undef, n_samples)
    for i in 1:n_samples
        predictions[i] = apply_adaboost_stumps(stumps, coeffs, features[i,:])
    end
    return predictions
end
