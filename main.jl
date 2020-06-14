include("estimator.jl")
include("fit_stage.jl")

include("predict.jl")
include("utils.jl")


features,labels = getData()

pred = fill(0.5, size(features, 1)) #initial predictions

gradients = labels .- pred
hessians = pred .* (1 .- pred)

base_out = sum(gradients)/(sum(hessians) + 1.0) #ouput for 1st node
base_score = sum(gradients)^2/(sum(hessians) + 1.0)


n = Node(base_out, base_score)
t = Tree(n,1)

fit(t,features, gradients, hessians)

p = predict(t, features)

logOdds2probab.(p)[1]
