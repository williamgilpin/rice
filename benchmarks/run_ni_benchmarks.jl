# import Pkg; Pkg.add("NetworkInference")
using Pkg
Pkg.activate("NetworkInference.jl/.")
Pkg.instantiate()
using NetworkInference
# using Pkg
# Pkg.activate(".")
# using Test
using DelimitedFiles

println("Getting nodes...")
# data_path = joinpath(dirname(@__FILE__), "data")
# data_file_path = joinpath(data_path, "network_data.txt")
data_file_path = "dump/temp.txt"
nodes = get_nodes(data_file_path)

println("Inferring networks...")

arg = ARGS[1]
println("Running method: $arg")
if arg == "mi"
    mi_network = InferredNetwork(MINetworkInference(), nodes)
    write_network_file("dump/mi_output.txt", mi_network)
elseif arg == "clr"
    clr_network = InferredNetwork(CLRNetworkInference(), nodes)
    write_network_file("dump/clr_output.txt", clr_network)
elseif arg == "puc"
    puc_network = InferredNetwork(PUCNetworkInference(), nodes)
    write_network_file("dump/puc_output.txt", puc_network)
elseif arg == "pidc"
    pidc_network = InferredNetwork(PIDCNetworkInference(), nodes)
    write_network_file("dump/pidc_output.txt", pidc_network)
else
    println("Invalid method")
end
