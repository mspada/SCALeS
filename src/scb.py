import numpy as np
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import matplotlib.pyplot as plt
from matplotlib.colors import CSS4_COLORS
from itertools import product, combinations
from string import ascii_uppercase
import random
import time
import threading
import math

from verboser import Verboser


"""
This class allows one to generate a (simple or multilayer) SCB (Supply Chain Backbone) graph. 
During initialisation, may take several arguments described below.
    - weights List[List[int]]: the weights of each node.
        When given as List[List[int]], generates a multilayer directed SCB (Supply Chain Backbone) graph, where weights[j][i] represents the weight of vertex i of layer j.
    - size List[int] > 0: the number of vertices in the desired network. Is ignored if weights is given.
        When given as List[int], generates a multilayer directed SCB (Supply Chain Backbone) graph where sizes[i] is the size of layer i. If the size is int N then a default pyramid is 
        generated with a total size approximately N. The decrease in layer sizes is approximately exponential following the car industry numbers. There is only 1 bottom layer.
        Default is size = [2,6,2].
    - tau (float or List[float]): the exponent in the power law used when generating the weights. Is ignored if weights is given.
        When given as float, generates the weight identically for all nodes in multilayer.
        When given as List[float], generates the weights in layer i with exponent tau[i] (multilayer only).
        Default is tau = 2.5.
    - alpha (float of Dict[float], <= 1): the outer multiplicative factor in the probability of having an edge between two nodes. 
        When given as a float in multilayer, uses the same alpha for all generations.
        When given as a Dict[float], uses alpha[(i,j)] between layers i and j. 
        Default is alpha = 1, fills the dictionary when not all values are given.
    - beta (float of Dict[float]): the inner multiplicative factor in the probability of having an edge between two nodes. 
        When given as a float in multilayer, uses the same beta for all generations.
        When given as a Dict[float], uses beta[(i,j)] between layers i and j. 
        Default is beta = 2, fills the dictionary when not all values are given.
    - gamma (float): an extra factor to decrease the probability of edges between non-consecutive layers. 
        When given, adds an extra outer factor exp( - gamma * (1 + |i - j|) ) to the probability of generating edges between layers i and j.
        Default is gamma = 1.
    - delta (float or List[float]): an extra outer factor in the probability of generating edges inside a given layer. 
        When given as float, uses the same for all layers.
        When given as List[float], uses delta[i] for layer i.
        Default is delta = .05.
    - prune (bool): if set to True, recursively deletes all vertices from all but the first layer with in-degree 0, 
        and all vertices from all but the last layer with in-degree 0.
    - integrity (str, 'none', 'partial' or 'full'): when pruning, forces re-generation of the graph until it has sufficiently many edges remaining.
        When set to 'none', does not enforce any condition on the number of vertices in the final graph.
        When set to 'full', forces re-generation until all vertices are kept after pruning. 
        When set to 'partial', forces re-generation until each layer keeps at least a fraction vratio from its vertices. 
        Note: if beta is too small, it may be very likely that some vertices have degree 0 and generating with integrity condition may take a very long time.
    - vratio (float or List[float], >= 0, <= 1): the ratio of vertices that needs to be kept in each layer.
        When given as a float, uses the same value for each layer. 
        When given as a List[float], uses vratio[i] for layer i.
        Default is 0.8.
    - verbose (bool): details the execution of each step when set to True.
    - output (string): 'png', 'dot', 'window'
        Default is 'png'.
    - seed (unsigned int): robust implementation, type is tested, global generators not seeded, using local instances.
        The data produced are identical for the same seed (network_diagram - tested with dot).
        Default is no seed.
    - kc (List[int]): Cut-off for truncated power law. 
        If kc =[i1, i2, ..., i3], the length of the vector must equal legth of the size.
        Befault cut-off is kc=[sqrt(size[0]), ... etc]
    
Once initialised, provides several methods. The most useful is:
    - draw(): draws a visual representation of the graph according to the output (png, dot or window).
    
The class also provides some static methods for easy generation of SCB (Supply Chain Backbone) graph with preset parameters. Available are:
    - tiny_example: multilayer SCB (Supply Chain Backbone) graph with 3 layers of 2, 4, 2 nodes repectively.
    - small_example: multilayer SCB (Supply Chain Backbone) graph with 4 layers of 3, 6, 6, 3 nodes repectively.
    - medium_example: multilayer SCB (Supply Chain Backbone) graph with 5 layers of 3, 6, 12, 6, 3 nodes repectively.

Data structures
    - vertex layers are indexed from 0, the direction (arrows are going from layer 0 - bot_layer to L-1 - top_layer)
    - vertex is represented by a pair (nx, ny) where nx is the layer index, ny is the node id.
    - a vertex with attributes is represented with a 2-element list of attributes is assigned to every node where the first element is a node
        and the second is a dictionary of attributes. 
"""

class SupplyChainBackbone:
    def __init__(self, **kwargs) -> None:
        # adding seed to have predictable runs for testing
        seed_val = kwargs.get('seed', None)
        # Explicit Type Enforcement
        if seed_val is not None and not isinstance(seed_val, int):
            raise TypeError(f"Seed must be an integer or None. Received: {type(seed_val).__name__}")
        # Safe Instantiation
        self._np_rng = np.random.default_rng(seed_val)
        self._std_rng = random.Random(seed_val)

        self.weights, self._size, self._alpha, self._beta, prune, integrity, vratio, verbose, self._output = self._parse_kwargs(kwargs=kwargs)

        num_attempts = 1 

        with Verboser(verbose, f"Generating SCB (Supply Chain Backbone) graph ({num_attempts} attempt{'s' if num_attempts > 1 else ''} -- {0:.2f}s)") as verboser:
            start = time.time()
            if not prune:
                if self._is_simple: self._generate_simple_chunglu()
                else: self._generate()
                self._edge_number = self.edge_number()
                verboser.terminate(f"SCB (Supply Chain Backbone) graph successfully generated ({time.time()-start:.2f}s).")
            else: 
                while True:
                    self._generate()
                    self._edge_number = self.edge_number()
                    self._prune()
                    if integrity == 'none' \
                        or integrity == 'partial' and (np.array(self.size()) / np.array(self._size) >= vratio).all() \
                        or integrity == 'full' and (np.array(self.size()) == np.array(self._size) ).all():
                        break
                    num_attempts += 1
                    verboser.update(f"Generating SCB (Supply Chain Backbone) graph ({num_attempts} attempt{'s' if num_attempts > 1 else ''} -- {time.time()-start:.2f}s)")
                verboser.terminate(f"SCB (Supply Chain Backbone) graph successfully generated. ({num_attempts} attempt{'s' if num_attempts > 1 else ''} -- {time.time()-start:.2f}s)")    
            
            self.warehouse()
            if prune:
                self.prune_warehouse(verbose=verbose)
            self._edge_number = self.edge_number()

    def original_size(self):
        return self._size

    def size(self):
        return self._current_size(self.network, len(self._size))

    def edge_number(self):
        return { (i, j): sum( self.network.nodes[u]['layer']==i and self.network.nodes[v]['layer']==j for u,v in self.network.edges ) for i, j in combinations(range(len(self._size)), 2) }

    def original_edge_number(self):
        return self._edge_number

    def draw(self):
        self._draw(self.network,self._output)

    def _prune(self):
        self._prune_multilayer(self.network, verbose=False)

    def _truncated_power_law(self, gamma, kc, size):
        samples = []
        while len(samples) < size:
            k = self._np_rng.zipf(a=gamma)
            if self._np_rng.random() < np.exp(-k / kc):
                samples.append(k)
        return samples

    def _parse_kwargs(self, kwargs: dict):
        if 'weights' in kwargs:
            weights = kwargs['weights']
            if type(weights[0]) == list and len(weights) == 1:
                weights = weights[0]
            self._tau = [2.5] * len(weights)
        else: 
            size  = kwargs['size'] if 'size' in kwargs else [2, 6, 2]
            if isinstance(size, int):
                size = SupplyChainBackbone._default_pyramid(size)
            # print(size)

            tau     = [kwargs['tau']] * len(size) if 'tau' in kwargs and type(kwargs['tau']) == float \
                    else kwargs['tau'] if 'tau' in kwargs and len(kwargs['tau']) == len(size) \
                    else [2.5] * len(size)
            self._tau = tau
            # this was a simple power law
            # weights = [ list(self._np_rng.zipf(a=t, size=s)) for t,s in zip(tau, size) ]

            # Assuming 'tau' and 'size' are defined, e.g., tau=[2.5, 3.0, 2.1], size=[2, 6, 2]
            kc = kwargs.get('kc', None)
            if 'kc' in kwargs and (len(kc) == len(size)):
                # Assuming tau, size, and kc_list are lists of the same length
                # e.g., tau = [2.5, 3.0, 2.1], size = [2, 6, 2], kc_list = [a, b, c]
                weights = [
                    self._truncated_power_law(
                        gamma=t, 
                        kc=k,       # Explicit cutoff provided by the list
                        size=s
                    ) 
                    for t, s, k in zip(tau, size, kc)
                ]
                # print(size, weights)
            elif 'kc' in kwargs and (len(kc) != len(size)):
                print("Error, kc not consistent with the size")
                exit(1)
            else:
                weights = [
                    self._truncated_power_law(
                        gamma=t, 
                        kc=max(1.0, 1.2*np.sqrt(s)),  # Dynamic kc: square root of the layer's size
                        size=s
                    ) 
                    for t, s in zip(tau, size)
                ]
                # print (size, weights)
        
        weights = weights[0] if type(weights[0]) == list and len(weights) == 1 else weights

        size = [len(w) for w in weights]
        gamma = kwargs['gamma'] if 'gamma' in kwargs and type(kwargs['gamma']) in {float, int} else 1
        delta = kwargs['delta'] if 'delta' in kwargs and type(kwargs['delta']) == list \
                else [kwargs['delta']] * len(size) if 'delta' in kwargs and type(kwargs['delta']) == float \
                else [.05] * len(size) 
        alpha = { (X,Y): kwargs['alpha'] for X,Y in combinations(range(len(size)), 2) } | {(X,X): kwargs['alpha'] for X in range(len(size))} if 'alpha' in kwargs and type(kwargs['alpha']) in {float, int} \
                else { (X,Y): np.exp( gamma * (1 + X - Y) ) for X,Y in combinations(range(len(size)), 2) } | {(X,X): delta[X] for X in range(len(size))} | kwargs['alpha'] if 'alpha' in kwargs \
                else { (X,Y): np.exp( gamma * (1 + X - Y) ) for X,Y in combinations(range(len(size)), 2) } | {(X,X): delta[X] for X in range(len(size))}
        beta  = { (X,Y): kwargs['beta'] for X,Y in combinations(range(len(size)), 2) } | {(X,X): kwargs['beta'] for X in range(len(size))} if 'beta' in kwargs and type(kwargs['beta']) in {float, int} \
                else { (X,Y): 2 for X,Y in combinations(range(len(size)), 2) } | {(X,X): 2 for X in range(len(size))} | kwargs['beta'] if 'beta' in kwargs \
                else { (X,Y): 2 for X,Y in combinations(range(len(size)), 2) } | {(X,X): 2 for X in range(len(size))}
        vratio = [ kwargs['vratio'] ] * len(size) if 'vratio' in kwargs and type(kwargs['vratio']) in [float, int] else \
                    kwargs['vratio'] if 'vratio' in kwargs and type(kwargs['vratio']) == list and len(kwargs['vratio']) == len(size) else\
                    [ .8 ] * len(size)

        self._w = kwargs.get('w', 0.1)
        if 'betaw' in kwargs:
            self._betaw = kwargs['betaw']
        else:
            if 'beta' in kwargs and isinstance(kwargs['beta'], (float, int)):
                self._betaw = kwargs['beta']
            elif 'beta' not in kwargs:
                self._betaw = 2 # default beta
            else:
                raise ValueError("betaw must be defined if beta is not a scalar.")

        prune = kwargs['prune'] if 'prune' in kwargs else False
        integrity = kwargs['integrity'] if 'integrity' in kwargs and kwargs['integrity'] in ['full', 'partial', 'none'] else 'none'
        verbose = kwargs['verbose'] if 'verbose' in kwargs and type(kwargs['verbose']) == bool else False
        output = kwargs['output'] if 'output' in kwargs and kwargs['output'] in ['png', 'dot', 'window'] else 'png'
        return weights, size, alpha, beta, prune, integrity, vratio, verbose, output

    def _generate(self):
        nodes = [ ((X, n), {'layer': X, 'weight': w, 'top_layer': X==len(self._size)-1, 'bot_layer': X==0, 'node_type': 'P' }) for X, wX in enumerate(self.weights) for n, w in enumerate(wX) ]        
        edges = []
        for (X, wX), (Y, wY) in combinations(enumerate(self.weights), 2): 
            edges.extend(self._edges_from_weights(
                            weightsA=wX, weightsB=wY, 
                            alpha=self._alpha[(X,Y)], beta=self._beta[(X,Y)],
                            indexA=X, indexB=Y))
        for (X, wX) in enumerate(self.weights):
            edges.extend(self._edges_from_weights(
                            weightsA=wX, weightsB=wX, 
                            alpha=self._alpha[(X,X)], beta=self._beta[(X,X)],
                            indexA=X, indexB=X))
        self.network = nx.DiGraph()
        self.network.add_nodes_from(nodes)
        self.network.add_edges_from(edges)

    # helper methods
    def _adj_matrix_from_weights(self, weightsA, weightsB, alpha, beta):
        proba_adj_matrix = alpha * np.minimum(beta * np.outer( weightsA, weightsB) / (len(weightsA) + len(weightsB)), 1)
        adj_matrix = self._np_rng.binomial(n=1, p=proba_adj_matrix, size=proba_adj_matrix.shape)
        np.fill_diagonal(adj_matrix, 0)
        return adj_matrix
    
    @staticmethod
    def _edges_from_adj_matrix(adj_matrix, indexA=None, indexB=None):
        return [ ( (indexA, a) if indexA is not None else a, \
                   (indexB, b) if indexB is not None else b ) \
                   for (a, b) in list(zip(*adj_matrix.nonzero())) ]
    
    def _edges_from_weights(self, weightsA, weightsB, alpha, beta, indexA=None, indexB=None):
        return SupplyChainBackbone._edges_from_adj_matrix( 
                        adj_matrix=self._adj_matrix_from_weights(weightsA=weightsA, weightsB=weightsB, alpha=alpha, beta=beta),
                        indexA=indexA,
                        indexB=indexB)
    
    def _draw(self, network, output):
        # Robust layer-to-color mapping (Fixes IndexError)
        layers = set([X for X, _ in network.nodes])
        col_layers = {X: self._std_rng.choice(list(CSS4_COLORS.keys())) for X in layers}

        # Matplotlib branches ('png' or 'window')
        if output in ('png', 'window'):
            col = ['red' if network.nodes[(X,n)].get('node_type') == 'W' else col_layers[X] for X, n in network.nodes]
            lab = {(X, n): f"W{network.nodes[(X,n)].get('w_id')}" if network.nodes[(X,n)].get('node_type') == 'W' else ascii_uppercase[X] + str(n) for X, n in network.nodes}
            pos = nx.multipartite_layout(G=network, subset_key="layer")
            
            plt.figure(figsize=(8, 8))
            nx.draw(G=network, pos=pos, node_color=col, node_size=100, labels=lab, with_labels=True, connectionstyle='arc3, rad=0.1', font_size=8)

            if output == 'png':
                plt.savefig("networks/network_diagram.png", format="png", dpi=300)
                plt.close()
            else:
                plt.show()

        # Graphviz branch ('dot')
        elif output == 'dot':
            for node in network.nodes:
                X, n = node
                if network.nodes[node].get('node_type') == 'W':
                    network.nodes[node]['label'] = f"W{network.nodes[node].get('w_id')}"
                    network.nodes[node]['fillcolor'] = 'red'
                else:
                    network.nodes[node]['label'] = f"{ascii_uppercase[X]}{n}"
                    network.nodes[node]['fillcolor'] = col_layers[X]
                network.nodes[node]['style'] = "filled"
                network.nodes[node]['group'] = str(X) 
                
            write_dot(network, "networks/network_diagram.dot")
        elif output == 'none':
            # this is important in case of many experiments, output is expensive
            pass
            # print("output none")

    def get_node_from_label(self, label: str):
        """
        Takes a string label (e.g., 'A0', 'C12') and returns the corresponding 
        node tuple (X, n) if it exists in the network.
        """
        from string import ascii_uppercase
        
        if not label or len(label) < 2 or not label[0].isalpha():
            raise ValueError(f"Invalid label format: '{label}'. Expected format like 'A0'.")
            
        layer_char = label[0].upper()
        
        try:
            # Reverse the ascii_uppercase mapping to get the layer index X
            X = ascii_uppercase.index(layer_char)
            # Parse the remaining numbers to get the node index n
            n = int(label[1:])
        except ValueError:
            raise ValueError(f"Cannot parse label: '{label}'.")
            
        node_tuple = (X, n)
        
        # Verify the reconstructed node actually survived pruning and exists in the graph
        if node_tuple in self.network.nodes:
            return node_tuple
        else:
            raise KeyError(f"Node {node_tuple} from label '{label}' does not exist in the network.")
        
    def warehouse(self):
        N_std = len(self.network.nodes)
        N_wh = int(self._w * N_std)
        if N_wh <= 0:
            return

        tau_wh = self._tau[0]
        kc_wh = max(1.0, 1.2 * np.sqrt(N_wh))
        
        W_wh = self._truncated_power_law(gamma=tau_wh, kc=kc_wh, size=N_wh)
        
        n_layers = len(self._size)
        if n_layers < 3:
            return # Not enough layers to place warehouses between the lowest and highest layers
        
        L_sizes = self.size()
        
        nodes_wh = []
        max_n = {X: -1 for X in range(n_layers)}
        for (X, n) in self.network.nodes:
            if n > max_n[X]:
                max_n[X] = n
                
        for i in range(N_wh):
            l = self._np_rng.integers(1, n_layers - 1)
            max_n[l] += 1
            n = max_n[l]
            weight = W_wh[i]
            # w_id is used for formatting the node name as W{w_id}
            nodes_wh.append( ((l, n), {'layer': l, 'weight': weight, 'top_layer': l==n_layers-1, 'bot_layer': l==0, 'node_type': 'W', 'w_id': i + 1}) )
            L_sizes[l] += 1
            
        self.network.add_nodes_from(nodes_wh)
        
        edges_wh = []
        for X in range(n_layers):
            for Y in range(X, n_layers):
                nodesA = [n for n in self.network.nodes if self.network.nodes[n]['layer'] == X]
                nodesB = [n for n in self.network.nodes if self.network.nodes[n]['layer'] == Y]
                
                weightsA = np.array([self.network.nodes[n]['weight'] for n in nodesA])
                weightsB = np.array([self.network.nodes[n]['weight'] for n in nodesB])
                
                if X != Y:
                    denom = L_sizes[X] + L_sizes[Y]
                    proba = self._alpha[(X,Y)] * np.minimum(self._betaw * np.outer(weightsA, weightsB) / denom, 1) if denom > 0 else np.zeros((len(weightsA), len(weightsB)))
                else:
                    denom = 2 * L_sizes[X]
                    proba = self._alpha[(X,X)] * np.minimum(self._betaw * np.outer(weightsA, weightsB) / denom, 1) if denom > 0 else np.zeros((len(weightsA), len(weightsB)))
                    
                adj = self._np_rng.binomial(n=1, p=proba, size=proba.shape)
                if X == Y:
                    np.fill_diagonal(adj, 0)
                    
                for i, u in enumerate(nodesA):
                    for j, v in enumerate(nodesB):
                        if adj[i, j] == 1:
                            if self.network.nodes[u].get('node_type') == 'W' or self.network.nodes[v].get('node_type') == 'W':
                                edges_wh.append((u, v))
                                
        self.network.add_edges_from(edges_wh)

    def prune_warehouse(self, verbose=False):
        start_num_nodes, start_num_edges = len(self.network.nodes), len(self.network.edges)
        while True:
            nodes_to_remove = [
                node for node in self.network.nodes 
                if self.network.nodes[node].get('node_type') == 'W' and 
                   (len(list(self.network.predecessors(node))) == 0 or len(list(self.network.successors(node))) == 0)
            ]
            if not nodes_to_remove: 
                break
            self.network.remove_nodes_from(nodes_to_remove)
            
        end_num_nodes, end_num_edges = len(self.network.nodes), len(self.network.edges)
        if verbose: 
            print(f'Pruned warehouses.\nKept {100*end_num_nodes/start_num_nodes:.1f}% of the nodes and {100*end_num_edges/start_num_edges:.1f}% of the edges.')

    @staticmethod
    def _prune_multilayer(network, verbose=False):
        start_num_nodes, start_num_edges = len(network.nodes), len(network.edges)
        while True:
            nodes_no_indegree  = [ node for node in network.nodes if len(list(network.predecessors(node))) == 0 and not network.nodes[node]['bot_layer'] ]
            nodes_no_outdegree = [ node for node in network.nodes if len(list(network.successors(node))) == 0 and not network.nodes[node]['top_layer']]
            if not (nodes_no_indegree or nodes_no_outdegree): break
            network.remove_nodes_from(nodes_no_indegree)
            network.remove_nodes_from(nodes_no_outdegree)
        end_num_nodes, end_num_edges = len(network.nodes), len(network.edges)
        if verbose: print(f'Pruned graph.\nKept {100*end_num_nodes/start_num_nodes:.1f}% of the nodes and {100*end_num_edges/start_num_edges:.1f}% of the edges.')

    @staticmethod
    def _default_pyramid(N):
        z = [141, 5241, 23433, 84945, 127419, 44597]
        p = [1, z[1]/z[0], z[2]/z[0], z[3]/z[0], z[4]/z[0], z[5]/z[0]]
        r = 1 + z[1]/z[0] + z[2]/z[0] + z[3]/z[0] + z[4]/z[0] + z[5]/z[0]
        x = N / r
        sum = 0
        size = [0] * 6
        for i in range(6):
            size[i] = math.ceil(x*p[i])
            sum = sum + size[i]
        size[0] = size[0]/3
        for i in range(4):
            if size[0] < 2:
                del size[0]
        size.reverse()
        size[0] = int(size[0]/1.5)
        # size[2] = int(size[2]/2)
        return size

    @staticmethod
    def _current_size(network, num_layers):
        size = [0] * num_layers
        for X, n in network.nodes:
            size[X] += 1
        return size

    # generation methods
    @staticmethod
    def _example_generation(size, tau=2.5, prune=False, complete='no', vratio=.9):
        complete = complete if complete in {'no', 'partial', 'yes'} else 'no'    
        if not prune: 
            cl = SupplyChainBackbone(size=size, tau=tau, beta=2)
            return cl
        while True:
            cl = SupplyChainBackbone(size=size, tau=tau, beta=2)
            SupplyChainBackbone._prune(cl.network)
            if complete == 'no' or complete == 'partial' and len(cl.network.nodes) >= vratio * sum(size) or len(cl.network.nodes) == sum(size):
                return cl

    @staticmethod
    def tiny_example(tau=2.5, prune=False, integrity='none', vratio=.5):
        return SupplyChainBackbone(size=[2,4,2], tau=tau, prune=prune, integrity=integrity, vratio=vratio)
        # return SupplyChainBackbone._example_generation(size=[2,4,2], tau=tau, prune=prune, complete=complete, vratio=vratio)
    
    @staticmethod
    def small_example(tau=2.5, prune=False, integrity='none', vratio=.7):
        return SupplyChainBackbone(size=[3, 6, 6 ,3], beta=3, tau=tau, prune=prune, integrity=integrity, vratio=vratio)
        # return SupplyChainBackbone._example_generation(size=[3, 6, 6 ,3], tau=tau, prune=prune, complete=complete, vratio=vratio)

    @staticmethod
    def medium_example(tau=2.5, prune=False, integrity='none', vratio=.8):
        return SupplyChainBackbone(size=[3,6,12,6,3], beta=4, tau=tau, prune=prune, integrity=integrity, vratio=vratio)
        # return SupplyChainBackbone._example_generation(size=[3,6,12,6,3], tau=tau, prune=prune, complete=complete, vratio=vratio)


if __name__ == '__main__':

    cl = SupplyChainBackbone(size=[10, 10, 10], prune=True)
    cl.draw()


    # small = SupplyChainBackbone.small_example()
    # small.draw()

    # cl = SupplyChainBackbone(size=[4,10,4], beta=2, prune=True, integrity='partial', vratio=[1, .8, .2])
    # cl.draw()

    # tiny   = SupplyChainBackbone.tiny_example(prune=True, integrity='partial')
    # tiny.draw()
    # small  = SupplyChainBackbone.small_example(prune=True, integrity='full')
    # small.draw()
    # medium = SupplyChainBackbone.medium_example(prune=True, integrity='partial')
    # medium.draw()

    # tiny.draw()
    # small.draw()
    # medium.draw()
