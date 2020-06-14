#BASE_SCORE = 0.5

function _fit_stage(node, features, gradients, hessians, curr_depth, max_depth)
    split_point     = nothing
    best_gain       = typemin(Float32)
    best_index      = 0
    best_left_score = nothing
    best_right_score =  nothing
    l_out           = nothing
    r_out           = nothing
    split_feat      = 0

    for i=1:size(features,2)
        ind         = sortperm(features[:,i])
        x_vals      = features[:,i][ind]
        grads_cum   = cumsum(gradients[ind])
        hess_cum    = cumsum(hessians[ind])
        max_grad    = grads_cum[end]
        max_hess    = hess_cum[end]

        for j=1:size(features,1)-1
            l_scoreDr   = hess_cum[j] + 1.0
            r_scoreDr   = max_hess - hess_cum[j] +1.0

            if ((abs(l_scoreDr) <1e-150) || (abs(r_scoreDr) <1e-150)) #in xgb, code
                continue
            end

            l_scoreNr   = grads_cum[j]^2
            r_scoreNr   = (max_grad - grads_cum[j])^2
            left_score  = l_scoreNr/l_scoreDr
            right_score = r_scoreNr/r_scoreDr
            curr_score  = left_score + right_score - node.score

            if curr_score > best_gain
                best_gain   = curr_score
                split_feat = i
                split_point = (x_vals[j] + x_vals[j+1])/2.0
                best_index  = j
                best_left_score = left_score
                best_right_score = right_score
                l_out = grads_cum[j]/l_scoreDr
                r_out = (max_grad - grads_cum[j])/r_scoreDr
            end
        end
    end

    if best_gain > 0
        if curr_depth < max_depth
            node.split_feat = split_feat
            node.splitpoint = split_point
            node.left_node = Node(l_out, best_left_score)
            node.right_node = Node(r_out, best_right_score)
        else
            node.split_feat = split_feat
            node.splitpoint = split_point
            node.left_node = Leaf(l_out)
            node.right_node = Leaf(r_out)
        end
    else
        node.isleaf = true
        node = Leaf(output)
    end
    return node
end

function fit(tree::Tree,features, gradients, hessians)
    curr_depth = 1
    return _fit_stage(tree.nodes[1], features, gradients, hessians, curr_depth, tree.max_depth)
end
