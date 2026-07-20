using DelimitedFiles 
using Clustering 
using LinearAlgebra

cd("isolet\\dependence_graphs\\")

map = readdlm("mi_map.txt")
map = map/maximum(map)

D = -map .+ 1
D = D - Diagonal(D)

result = hclust(D, linkage=:average, branchorder=:barjoseph)
ordering = result.order
