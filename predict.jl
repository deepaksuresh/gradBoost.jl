function predict(node :: Leaf, x)
    return node.output
end

function predict(node :: Node, x)
    if x[node.split_feat] < node.splitpoint
        return predict(node.left_node, x)
    else
        return predict(node.right_node, x)
    end
end

function predict(tree :: Tree, x)
    log_odds = fill(0.0, size(x,1))
    for j=1:size(x,1)
        for i in tree.nodes
            log_odds[j] += predict(i,x[j,:])
        end
    end
    return log_odds
end

function logOdds2probab(v)
    return exp(v)/(1.0 + exp(v))
end
