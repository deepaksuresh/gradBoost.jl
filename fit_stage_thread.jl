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

    num_feat = size(features,2)

    split_point_list    = fill([], num_feat)
    gain_list           = fill([], num_feat)
    index_list          = fill([], num_feat)
    left_score_list     = fill([], num_feat)
    right_score_list    = fill([], num_feat)
    l_out_list          = fill([], num_feat)
    r_out_list          = fill([], num_feat)
    split_feat_list     = fill([], num_feat)

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

            push!(left_score_list[i], left_score)
            push!(right_score_list[i], right_score)
            push!(gain_list[i], curr_score)
            push!(split_point_list[i], (x_vals[j] + x_vals[j+1])/2.0)
            push!(index_list[i], j)
            push!(l_out_list[i], grads_cum[j]/l_scoreDr)
            push!(r_out_list[i], (max_grad - grads_cum[j])/r_scoreDr)



            # if curr_score > best_gain
            #     best_gain   = curr_score
            #     split_feat = i
            #     split_point = (x_vals[j] + x_vals[j+1])/2.0
            #     best_index  = j
            #     best_left_score = left_score
            #     best_right_score = right_score
            #     l_out = grads_cum[j]/l_scoreDr
            #     r_out = (max_grad - grads_cum[j])/r_scoreDr
            # end
        end
    end

    gain_mat    = reduce(hcat, gain_list)
    best_ind    = argmax(gain_mat)

    node.split_feat  = best_ind.I[1]
    best_score_ind = best_ind.I[2]
    node.split_point = split_point_list[split_feat][best_score_ind]
    node.left_node = Node(l_out_list[split_feat][best_score_ind], left_score_list[split_feat][best_score_ind])
    node.right_node = Node(r_out_list[split_feat][best_score_ind], right_score_list[split_feat][best_score_ind])




    # if best_gain > 0
    #     if curr_depth < max_depth
    #         node.split_feat = split_feat
    #         node.splitpoint = split_point
    #         node.left_node = Node(l_out, best_left_score)
    #         node.right_node = Node(r_out, best_right_score)
    #     else
    #         node.split_feat = split_feat
    #         node.splitpoint = split_point
    #         node.left_node = Leaf(l_out)
    #         node.right_node = Leaf(r_out)
    #     end
    # else
    #     node.isleaf = true
    #     node = Leaf(output)
    # end
    return node
end

function fit(tree::Tree,features, gradients, hessians)
    curr_depth = 1
    return _fit_stage(tree.nodes[1], features, gradients, hessians, curr_depth, tree.max_depth)
end
