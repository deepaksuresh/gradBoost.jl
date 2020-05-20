
unzip(arr) = map(x->getfield.(arr, x), fieldnames(eltype(arr)))

mutable struct Node
    metric      :: Float64
    num_samples :: Int64
    num_samples_class :: Vector{}
    left_child           :: Node
    right_child           :: Node
    predicted_class :: Int64
    feature_index     :: Int64
    threshold   :: Float64
    is_leaf     :: Bool
    depth       :: Int

    function Node(num_samples, depth, predicted_class, num_samples_class)
        node = new()
        node.num_samples = num_samples
        node.depth = depth
        node.is_leaf = false
        node.predicted_class = predicted_class
        node.num_samples_class = num_samples_class
        node
    end
end

mutable struct Tree
    num_classes :: Int
    max_depth :: Int
end

function node_accuracy(predictions, labels)
    # n = length(predictions)
    return sum(predictions .== labels)
    # if n == 0
    #     return 0
    # else
    #     positives = sum(predictions .== 1.0)
    #     negatives = sum(predictions .== 0.0)
    #     if positives > negatives
    #         return positives/n
    #     end
    #     return negatives/n
    # end
end

function _split(tree,features, labels)
    m = length(y)
    num_samples_class = [sum(labels .== i) for i=1:tree.num_classes]
    (thresholds, classes) = unzip(sort(zip(features, labels)))

    num_left = repeat(1, tree.num_classes)
    num_right = num_samples_class[:]

    parent_acc = node_accuracy(repeat(argmax(labels), m), labels)/m

    threshold = nothing
    feat_ind = nothing
    for j=1:size(features)[2]
        for i=2:m
            c = classes[i]
            num_left[c] += 1
            num_right[c] -= 1

            acc_left = node_accuracy(num_left, labels)
            acc_right = node_accuracy(num_right, labels)

            if thresholds[i] == thresholds[i-1]
                continue
            end

            net_acc = (acc_left + acc_right)/m

            if net_acc > parent_acc
                threshold = thresholds[i]
                feat_ind = j
            end
        end
    end
    return feat_ind, threshold
end




function _fit(tree,features,labels, depth)
    num_samples = size(features)[1]
    num_samples_class = [sum(labels .== i) for i in tree.num_classes]
    predicted_class = argmax(num_samples_class)
    depth += 1
    node = Node(num_samples, depth, predicted_class, num_samples_class)

    # if (depth < tree.max_depth)
    feat_ind, threshold = _split(tree, features, labels)
    if !(isnothing(feat_ind))
        left_indices = features[:, feat_ind] .< threshold

        features_left = features[left_indices,:]
        features_right = features[.!(left_indices),:]

        labels_left = labels[left_indices]
        labels_right = labels[.!(left_indices)]

        node.feature_index = feat_ind
        node.threshold = threshold

        node.left_child = _fit(features_left, labels_left, depth)
        node.right_child = _fit(features_right, labels_right, depth)
    end
    return node
end
