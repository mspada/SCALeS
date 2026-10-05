import numpy as np
import matplotlib.pyplot as plt

from scb import SupplyChainBackbone
from chunglu_converter import ChungLuConverter
from supplychain import SupplyChain
from perturber import SupplyChainPerturber
from analyser import SupplyChainAnalyser


#####################################################################################################
#  This file demonstrates how to use the three classes SupplyChainBackbone, ChungLuConverter and SupplyChain.   #
#####################################################################################################


#####################################################################################################
#                                               SupplyChainBackbone                                             #
#####################################################################################################
#                                                                                                   #
# This is a powerful class that allows to sample random SupplyChainBackbone(-like) directed graphs.             #
# It allows to sample both simple and partite graphs. It also provides methods to visualise         #
# the network.                                                                                      #
#                                                                                                   #
#####################################################################################################

#####################
#   Simple SupplyChainBackbone  #
#####################

# Generate a simple directed SCB (Supply Chain Backbone) graph with 8 nodes with the prodiced weights.
# cl = SupplyChainBackbone( weights=[1,2,3,4,5,6,7,8] )
# cl.draw()

# Generate a simple directed SCB (Supply Chain Backbone) graph with 10 nodes, and weights sampled according to a
# power-law with exponent 2.2
# cl = SupplyChainBackbone( size=10, tau=2.2)
# cl.draw()

# Generate a simple directed SCB (Supply Chain Backbone) graph with pre-chosen parameters. Three sizes are available:
# tiny - 6 nodes; small - 10 nodes; medium -- 20 nodes
# cl = SupplyChainBackbone.tiny_simple_example()
# cl = SupplyChainBackbone.small_simple_example()
# cl = SupplyChainBackbone.medium_simple_example()
# cl.draw()


#####################
#  Partite SupplyChainBackbone  #
#####################

# Generate a 3-partite directed SCB (Supply Chain Backbone) graph with 3 + 5 + 3 nodes with the prodiced weights.
# cl = SupplyChainBackbone( weights=[ [1,2,1], [2,3,2,3,2], [2,1,2] ] )
# cl.draw()

# Generate a 3-partite directed SCB (Supply Chain Backbone) graph with 3 + 5 + 3 nodes, and weights sampled according to
# power-laws with exponent 2.5.
# cl = SupplyChainBackbone( size=[3,5,3], tau=2.5)
# cl.draw()

# Generate a 3-partite directed SCB (Supply Chain Backbone) graph with 3 + 5 + 3 nodes, and weights sampled according to
# power-laws with exponent 2.2, 2.3, 2.4 respectively.
# cl = SupplyChainBackbone( size=[3,5,3], tau=[2.2, 2.3, 2.4])
# cl.draw()

# Generate a partite directed SCB (Supply Chain Backbone) graph with pre-chosen parameters. Three sizes are available:
# tiny - 2+4+2 nodes; small - 3+6+6+3 nodes; medium -- 3+6+12+6+3 nodes
# cl = SupplyChainBackbone.tiny_example()
# cl = SupplyChainBackbone.small_example()
# cl = SupplyChainBackbone.medium_example()
# cl.draw()

#####################
#      Pruning      #
#####################

# Pruning removes all vertices that have no in- or out-degree.
# It is only available for partite graphs. By default there is no pruning.
# Because pruning may leave the graph empty, one may add an 'integrity' 
# condition: the graph is generated until this condition is met.

# Generate a graph and don't prune it.
# cl = SupplyChainBackbone( size=[ 3, 6, 6, 3 ], alpha=.5, prune=False )
# cl.draw()

# Generate a graph and prune it. No integrality condition.
# cl = SupplyChainBackbone( size=[ 3, 6, 6, 3 ], alpha=.5, prune=True, integrity='none')
# cl.draw()

# Generate a graph and prune it. Require each part to keep at least half the nodes.
# cl = SupplyChainBackbone( size=[ 3, 6, 6, 3 ], alpha=.5, prune=True, integrity='partial', vratio=.5)
# cl.draw()

# Generate a graph and prune it. Require each part to each keep a fraction of the nodes.
# cl = SupplyChainBackbone( size=[ 3, 6, 6, 3 ], alpha=.5, prune=True, integrity='partial', vratio=[.5, .8, .8, .5])
# cl.draw()

# Generate a graph and prune it. Require all nodes to remain. Equivalent to setting vratio=1.
# cl = SupplyChainBackbone( size=[ 3, 6, 6, 3 ], alpha=.5, prune=True, integrity='full')
# cl.draw()


#####################################################################################################
#                                           ChungLuConverter                                        #
#####################################################################################################
#                                                                                                   #
# This class allows to turn a *partite* SCB (Supply Chain Backbone) graph into a supply chain.                        #
# Allows to display the result and write it in a .json file readable by SupplyChain.                #
# Currently not implemented for simple SCB (Supply Chain Backbone) graph.                                            #
#                                                                                                   #
#####################################################################################################

# Generate a partite SupplyChainBackbone example and turn it into a supply chain.
# Saves the supply chain to a json file and then display it. 
# Each node of the first layer receives 10 units of supply at each step. 
# The total production at each step must be 1.
# cl = SupplyChainBackbone.tiny_example(prune=True, integrity='full')
# clc = ChungLuConverter(cl)
# clc.to_json('supply_chains/example.json', duration=5, supply=10, demand=1)
# clc.draw_supplychain()


#####################################################################################################
#                                             SupplyChain                                           #
#####################################################################################################
#                                                                                                   #
# This class represents a supply chain. It reads from a json file and can be used to compute        #
# the optimal production allocation. Also allows to visualise the output.                           #
#                                                                                                   #
#####################################################################################################

# Generate a partite SupplyChainBackbone example and turn it into a supply chain.
# Finds optimal resource allowcation and displays the result.
# Edges with resource flow are coloured blue.
# Nodes which lack resource and in which it is 'magically' created 
# are represented in red.
# cl = SupplyChainBackbone.tiny_example(prune=True, integrity='full')
# clc = ChungLuConverter(cl)
# clc.to_json('supply_chains/example.json', duration=10, supply=50, demand=1)
# sc = SupplyChain('supply_chains/example.json')
# sc.draw_partite_base_network()
# sc.draw_partite_network()
# sc.find_optimum()
# sc.generate_dot(filename='example', show=False)
# /!\ Setting show=True will start a utility (on Mac) to show the generated PNG file


# Some methods allow to quickly generate a supply chain 
# pre-chosen parameters.
# sc = SupplyChain.tiny_pyramid(duration=10, verbose=True)
# sc = SupplyChain.small_pyramid(duration=10)
# sc = SupplyChain.small_medium_pyramid(duration=10)
# sc = SupplyChain.medium_pyramid(duration=10)
# sc = SupplyChain.tiny_double_pyramid(duration=10)
# sc = SupplyChain.small_double_pyramid(duration=10)
# sc = SupplyChain.small_medium_double_pyramid(duration=10)
# sc = SupplyChain.medium_double_pyramid(duration=10)
# sc.generate_dot(show=True, save=False)

#####################################################################################################
#                                        SupplyChainPerturber                                       #
#####################################################################################################
#                                                                                                   #
# This class provide a unique static method pertube that allows to stochastically perturbe          #
# an instance of supply SupplyChain.                                                                #
# Currently only two perturbations methods exist:                                                   #
#   - normal: we add a random normal noise to the cost and capacity of each edge;                   #
#   - binomial: we delete each edge independently with a certain probability.                       #        
#                                                                                                   #
#####################################################################################################

# Get a supply chain with preset parameters and 
# provide a normal pertubation of it.
# sc = SupplyChain.tiny_double_pyramid(duration=50, verbose=True)
# psc, perturbation_details = SupplyChainPerturber.perturbe(sc, type='normal', mu=0, sigma=.2, verbose=True)
# sc.find_optimum(verbose=True)
# psc.find_optimum(verbose=True)
# print(f"Original supply chain:  cost={sc.cost():.2f} / production={sc.production():.2f}")
# print(f"Perturbed supply chain: cost={psc.cost():.2f} / production={psc.production():.2f}")

# Note: one can tune the cost and capacities differently by giving independent mean
# and standard deviations for the cost (c_mu, c_sigma) and the capacity (u_mu, u_sigma).
# sc = SupplyChain.tiny_double_pyramid(duration=50, verbose=True)
# psc, perturbation_details = SupplyChainPerturber.perturbe(sc, type='normal', c_mu=0, c_sigma=.2, u_mu=.1, u_sigma=.3 verbose=True)
# sc.find_optimum(verbose=True)
# psc.find_optimum(verbose=True)
# print(f"Original supply chain:  cost={sc.cost():.2f} / production={sc.production():.2f}")
# print(f"Perturbed supply chain: cost={psc.cost():.2f} / production={psc.production():.2f}")

# Get a supply chain with preset parameters and 
# provide a binomial pertubation of it.
# sc = SupplyChain.tiny_double_pyramid(duration=50, verbose=True)
# psc, perturbation_details = SupplyChainPerturber.perturbe(sc, type='binomial', p=.1, verbose=True)
# sc.find_optimum(verbose=True)
# psc.find_optimum(verbose=True)
# print(f"Original supply chain:  cost={sc.cost():.2f} / production={sc.production():.2f}")
# print(f"Perturbed supply chain: cost={psc.cost():.2f} / production={psc.production():.2f}")

#####################################################################################################
#                                         SupplyChainAnalyser                                       #
#####################################################################################################
#                                                                                                   #
# This class provides two static methods to analyse the effect of perturbations on a supply chain.  #        
#                                                                                                   #
#####################################################################################################

# Get a supply chain with preset parameters and analyse the
# effect of a binomial perturbation on its cost and production
# capacity. A total of 100 perturbations are generated and the
# results are stored in the file temp/analysis_results.json.
# The results are then plotted from the file.

# The example from Maxime, tested ok
# sc = SupplyChain.tiny_pyramid(duration=20, verbose=True)
# SupplyChainAnalyser.analyse_perturbation(filename='temp/analysis_results.json', sc=sc, rep=10, type='binomial', p=.1, verbose=True)
# SupplyChainAnalyser.plot_from_data(filename='temp/analysis_results.json', c_binwidth=5, u_binwidth=3)

#####################################################################################################
#                                                                                                   #
# The new examples, Patrick                                                                         #        
#                                                                                                   #
#####################################################################################################
# Chung Lu data for Patrick 
# cl = SupplyChainBackbone(size=[20, 150, 50, 15, 5], prune=True)

#####################################################################################################
#                                                                                                   #
# Update April 26, output added, seed added, truncated power law                                    #        
#                                                                                                   #
#####################################################################################################
# cl = SupplyChainBackbone(size=[5, 12, 6, 2],  kc=[3,4,2], prune=True, output='png', seed=0) # this must generate error
# cl = SupplyChainBackbone(size=[5, 12, 6, 2], prune=True, output='png', seed=0)
# cl = SupplyChainBackbone(size=[20, 150, 50, 15, 5], prune=True)

# cl = SupplyChainBackbone(size=[5, 12, 6, 2], kc=[3,4,3,1], prune=True, output='dot', seed=0)
# cl.draw()
# clc = ChungLuConverter(cl, output='dot', seed=0)
# clc.to_json('networks/supply_chains/example.json', duration=5, supply=10, demand=1)
# clc.draw_supplychain()

# cl = SupplyChainBackbone(size=[5, 12, 6, 2], prune=True, output='png', seed=0)
# cl = SupplyChainBackbone(size=990, prune=True, output='png', integrity='partial', verbose=True)
# cl = SupplyChainBackbone(size=10000, prune=True, output='png', verbose=True, integrity='partial', vratio=0.4)
# print("Drawing ...")
# cl.draw()
# print(cl.original_size())
# print(cl.size())

#####################################################################################################
#                                                                                                   #
# Update 26 example for pyramyd, ident nodes in picture                                             #        
#                                                                                                   #
#####################################################################################################
cl = SupplyChainBackbone(size=100, prune=True, verbose=True, integrity='partial', vratio=0.4, seed = 100)
print(cl.original_size())
print(cl.size())
cl.draw()
my_node=cl.get_node_from_label("B43")
print(f"The tuple is: {my_node}")  # Output: The tuple is: (1, 3)
attributes = cl.network.nodes[my_node]
print(f"Node Attributes: {attributes}")
my_node=cl.get_node_from_label("B28")
print(f"The tuple is: {my_node}")  # Output: The tuple is: (1, 3)
attributes = cl.network.nodes[my_node]
print(f"Node Attributes: {attributes}")

clc = ChungLuConverter(cl, output = 'dot', seed = 100)
clc.draw_supplychain()
my_node=clc.get_node_from_label("B26(O)")
print(f"The tuple is: {my_node}")  # Output: The tuple is: (1, 3)
attributes = clc._supplychain.nodes[my_node]
print(f"Node Attributes: {attributes}")

#####################################################################################################
#                                                                                                   #
# Example sequence: Overall distribution, Figure 3, in the paper                                                      #        
#                                                                                                   #
#####################################################################################################
# def count_max_deg(network):
#     max_input_deg = max_out_deg = max_deg = 0
#     for n in cl.network.nodes:
#         input_deg = out_deg = deg = 0
#         for (n1, n2) in cl.network.edges:
#             if n == n2:
#                 input_deg += 1
#             if n == n1:
#                 out_deg += 1
#             deg = input_deg + out_deg
#         if max_input_deg < input_deg:
#             max_input_deg = input_deg
#         if max_out_deg < out_deg:
#             max_out_deg = out_deg
#         if max_deg < deg:
#             max_deg = deg
#     pi = [0] * (max_input_deg+1)
#     po = [0] * (max_out_deg+1)
#     p =  [0] * (max_deg+1)
#     for n in cl.network.nodes:
#         input_deg = out_deg = deg = 0
#         for (n1, n2) in cl.network.edges:
#             if n == n2:
#                 input_deg += 1
#             if n == n1:
#                 out_deg += 1
#             deg = input_deg + out_deg
#         pi[input_deg] += 1
#         po[out_deg] += 1
#         p[deg] += 1
#     return  max_input_deg,  np.array(pi), max_out_deg,  np.array(po), max_deg,  np.array(p)

# size = int(7 * np.sqrt(5000))
# pi_a = np.zeros(size)
# n_counts_pi = np.zeros(size)
# po_a = np.zeros(size)
# n_counts_po = np.zeros(size)
# p_a = np.zeros(size)
# n_counts_p = np.zeros(size)

# for i in range(100):
#     cl = SupplyChainBackbone(size=5000, prune=True, verbose=True, integrity='partial', vratio=0.4)
#     max_input_deg, pi, max_out_deg, po, max_deg, p = count_max_deg(cl.network)
#     k1 = len(pi)
#     pi_a[:k1] = (n_counts_pi[:k1] * pi_a[:k1] + pi) / (n_counts_pi[:k1] + 1)
#     k2 = len(po)
#     po_a[:k2] = (n_counts_po[:k2] * po_a[:k2] + po) / (n_counts_po[:k2] + 1)
#     k3 = len(p)
#     p_a[:k3] = (n_counts_p[:k3] * p_a[:k3] + p) / (n_counts_p[:k3] + 1)

# print(pi_a, po_a, p_a) 

# def plot_total_dist(pi_a, po_a, p_a):
#     # + 1 has no influence on results, but solves log(0) problem
#     x = np.log(np.array(range(len(pi_a))) + 1)
#     pi_a = np.log(pi_a + 1)
#     po_a = np.log(po_a + 1)
#     p_a = np.log(p_a + 1)

#     # 2. Initialize the figure (width, height in inches)
#     plt.figure(figsize=(8, 6))

#     # 3. Plot the vectors with custom styling
#     plt.plot(x, pi_a, label='Input degree distribution', color='blue', linewidth=1, linestyle='--', marker='o')
#     plt.plot(x, po_a, label='Output degree distribution', color='red', linewidth=1, linestyle='--', marker='o')
#     plt.plot(x, p_a, label='Total degree distribution', color='green', linewidth=1, marker='s')

#     # 4. Apply structural formatting
#     plt.title('Multi-partition')
#     plt.xlabel('log k')
#     plt.ylabel('log P(k)')
#     plt.legend()
#     plt.grid(True, linestyle=':', alpha=0.7)

#     # 5. Save the plot to disk
#     plt.savefig('networks/total-deg-dist.png', format='png', dpi=300, bbox_inches='tight')

#     # add computation of average layer sizes, average pyramid following car industry.
#     # 6. Close the plot to free memory
#     plt.close()


# plot_total_dist(pi_a, po_a, p_a)
#####################################################################################################
#                                                                                                   #
# Example sequence: Warehouse generation                                                            #        
#                                                                                                   #
#####################################################################################################
print()
print("Warehouse test")
cl = SupplyChainBackbone(size=[10, 20, 10, 1], prune=True, w=0.1, betaw=10, output='png', seed=42, beta={(2, 3): 10})
cl.draw()
print("Total nodes:", len(cl.network.nodes))
w_nodes = [n for n, d in cl.network.nodes(data=True) if d.get('node_type') == 'W']
print("Warehouse nodes:", len(w_nodes))

clc = ChungLuConverter(cl, output='png')
clc.draw_supplychain(filename='networks/supply_chain_with_warehouse')
clc.to_json('networks/supply_chains/test_wh.json', duration=5, demand=10, supply=100)
print("Finished conversion and drawing.")
