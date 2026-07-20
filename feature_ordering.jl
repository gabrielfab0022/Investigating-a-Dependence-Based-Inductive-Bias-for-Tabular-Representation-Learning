using DelimitedFiles 
using Clustering 
using LinearAlgebra

cd("isolet\\dependence_graphs\\")

mi_map = readdlm("mi_map.txt")
mi_map = mi_map/maximum(mi_map)

D = -mi_map .+ 1
D = D - Diagonal(D)

result = hclust(D, linkage=:average, branchorder=:barjoseph)
ordering = result.order