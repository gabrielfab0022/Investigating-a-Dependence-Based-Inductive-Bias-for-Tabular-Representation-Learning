using Associations
using Base.Threads
using DelimitedFiles
using Statistics

function empirical_cdf(x)
    n = length(x)
    y = [length(findall(x.<=x[i]))/n for i in 1:n]
    return y
end

#loading data
cd("isolet\\data\\")

X = readdlm("X_train.txt")

#equalization of the marginals 
@threads for i in 1:size(X,2)
    global X[:,i] = empirical_cdf(X[:,i])
end 

N = size(X,1) #n_samples
d = size(X,2) #n_features
mi_map = zeros(d,d)

est = KSG2(MIShannon(base=ℯ), k=5) #defining estimator


@threads for i in 1:d
    for j in i+1:d 
        x = X[:,i]
        y = X[:,j]
        MI =  association(est, x, y)
        global mi_map[i,j] = MI 
        global mi_map[j,i] = MI
    end 
end 
