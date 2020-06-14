using FileIO, JLD2

function getData()
    features = FileIO.load("./data/feat.jld2")["v"]
    labels = FileIO.load("./data/lab.jld2")["v"]
    return features,labels
end
