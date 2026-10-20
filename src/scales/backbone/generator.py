import numpy as np
import networkx as nx
from itertools import combinations
from string import ascii_uppercase
import random
import time
import math

from ..verboser import Verboser
from .. import config
from .warehouse import inject_warehouses, prune_warehouses


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
        Default is tau = 2.5. The tau for warehouses is inherited from global tau.
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
        Default is 'png'. Sets the output type for the draw() method.
    - seed (unsigned int): robust implementation, type is tested, global generators not seeded, using local instances.
        The data produced are identical for the same seed (network_diagram - tested with dot).
        Default is no seed.
    - kc (List[int]): Cut-off for truncated power law. 
        If kc =[i1, i2, ..., i3], the length of the vector must equal legth of the size.
        Befault cut-off is kc=[sqrt(size[0]), ... etc]
    - w (float): w*100 defines how mane percent of additional warehouse nodes are generated.
        Default is 0.1.
    -betawin (float): the beta for indegree of a warehouses.
        Default is inherited from the global beta parameter, which defaults to 2.
    -betawout (float): the beta for outdegree of a warehouses.
        Default is inherited from the global beta parameter, which defaults to 2.

Once initialised, provides several methods. The most useful is:
    - draw(filename): draws a visual representation of the graph according to the output (png, svg, dot or window).
        Default filename is networks/backbones/backbone (see scales.config).
    
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
                self._generate()
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

    def original_size(self) -> list[int]:
        """Returns the original, pre-pruned layer sizes of the backbone."""
        return self._size

    def size(self) -> list[int]:
        """Returns the current, post-pruned layer sizes of the backbone."""
        return self._current_size(self.network, len(self._size))

    def edge_number(self) -> dict[tuple[int, int], int]:
        """Computes and returns a dictionary of edge counts between each pair of layers."""
        return { (i, j): sum( self.network.nodes[u]['layer']==i and self.network.nodes[v]['layer']==j for u,v in self.network.edges ) for i, j in combinations(range(len(self._size)), 2) }

    def original_edge_number(self) -> dict:
        """Returns the initial pre-pruning edge distribution counts between layers."""
        return self._edge_number

    def draw(self, filename: str = None) -> None:
        """Draws the backbone graph to the configured output format (png, svg, dot, window)."""
        if filename is None:
            filename = str(config.networks_dir("backbones") / "backbone")
        self._draw(self.network, self._output, filename=filename)

    def _prune(self) -> None:
        """Helper method to execute multilayer degree-based pruning on the network."""
        self._prune_multilayer(self.network, verbose=False)

    def _truncated_power_law(self, gamma: float, kc: float, size: int) -> list[int]:
        """Generates random integers from a Zipf distribution bounded/truncated by a cutoff."""
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
                raise ValueError(f"kc has {len(kc)} entries but size has {len(size)} layers; they must match.")
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
        
        # Parse incoming beta for warehouses
        if 'betawin' in kwargs:
            self._betawin = kwargs['betawin']
        else:
            self._betawin = kwargs.get('beta', 2) if isinstance(kwargs.get('beta', 2), (float, int)) else 2
            
        # Parse outgoing beta for warehouses
        if 'betawout' in kwargs:
            self._betawout = kwargs['betawout']
        else:
            self._betawout = kwargs.get('beta', 2) if isinstance(kwargs.get('beta', 2), (float, int)) else 2
        
        prune = kwargs['prune'] if 'prune' in kwargs else False
        integrity = kwargs['integrity'] if 'integrity' in kwargs and kwargs['integrity'] in ['full', 'partial', 'none'] else 'none'
        verbose = kwargs['verbose'] if 'verbose' in kwargs and type(kwargs['verbose']) == bool else False
        output = kwargs['output'] if 'output' in kwargs and kwargs['output'] in ['png', 'svg', 'dot', 'window'] else 'png'
        return weights, size, alpha, beta, prune, integrity, vratio, verbose, output

    def _generate(self):
        nodes = [ ((X, n), {'layer': X, 'weight': w, 'top_layer': X==len(self._size)-1, 'bot_layer': X==0, 'type': 'P' }) for X, wX in enumerate(self.weights) for n, w in enumerate(wX) ]        
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
    def _adj_matrix_from_weights(self, weightsA: list[float], weightsB: list[float], alpha: float, beta: float) -> np.ndarray:
        """Computes a random adjacency matrix between two sets of nodes using standard Zipf probabilities."""
        proba_adj_matrix = alpha * np.minimum(beta * np.outer( weightsA, weightsB) / (len(weightsA) + len(weightsB)), 1)
        adj_matrix = self._np_rng.binomial(n=1, p=proba_adj_matrix, size=proba_adj_matrix.shape)
        np.fill_diagonal(adj_matrix, 0)
        return adj_matrix
    
    @staticmethod
    def _edges_from_adj_matrix(adj_matrix: np.ndarray, indexA: int = None, indexB: int = None) -> list[tuple]:
        """Translates a binary adjacency matrix into a list of node-to-node edge tuples."""
        return [ ( (indexA, a) if indexA is not None else a, \
                   (indexB, b) if indexB is not None else b ) \
                   for (a, b) in list(zip(*adj_matrix.nonzero())) ]
    
    def _edges_from_weights(self, weightsA: list[float], weightsB: list[float], alpha: float, beta: float, indexA: int = None, indexB: int = None) -> list[tuple]:
        """Generates a list of random layer-to-layer or layer-self edges based on node weight parameters."""
        return SupplyChainBackbone._edges_from_adj_matrix( 
                        adj_matrix=self._adj_matrix_from_weights(weightsA=weightsA, weightsB=weightsB, alpha=alpha, beta=beta),
                        indexA=indexA,
                        indexB=indexB)
    
    def _draw(self, network: nx.DiGraph, output: str, filename: str) -> None:
        """Helper call invoking NetworkVisualizer to render the network using multipartite positioning."""
        from ..viz.visualiser import NetworkVisualizer
        NetworkVisualizer.draw(
            network=network,
            output=output,
            filename=filename,
            node_size=100,
            figsize=(8, 8),
            font_size=8,
            rng=self._std_rng,
            node_label_func=lambda n, d: f"{ascii_uppercase[n[0]]}{n[1]}"
        )

    def get_node_from_label(self, label: str):
        """
        Takes a string label (e.g., 'A0', 'C12') and returns the corresponding 
        node tuple (X, n) if it exists in the network.
        """
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
        
    def warehouse(self) -> None:
        """Generates additional warehouse nodes and connects them with direction-aware beta weights (Algorithm 3)."""
        inject_warehouses(self)

    def prune_warehouse(self, verbose: bool = False) -> None:
        """Recursively removes any warehouse node with an in-degree or out-degree of 0."""
        prune_warehouses(self.network, verbose=verbose)

    @staticmethod
    def _prune_multilayer(network: nx.DiGraph, verbose: bool = False) -> None:
        """Prunes nodes lacking in-degree (except bottom layer) or out-degree (except top layer) in multi-layer graph."""
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
    def _default_pyramid(N: int) -> list[int]:
        """Generates a list of default layer sizes approximating total size N, shaped exponentially."""
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
    def _current_size(network: nx.DiGraph, num_layers: int) -> list[int]:
        """Helper returning the number of active nodes within each distinct layer of the network."""
        size = [0] * num_layers
        for X, n in network.nodes:
            size[X] += 1
        return size

    # generation methods
    @staticmethod
    def tiny_example(tau: float = 2.5, prune: bool = False, integrity: str = 'none', vratio: float = .5):
        """Factory static method generating a tiny 3-layer (2-4-2) backbone graph with custom options."""
        return SupplyChainBackbone(size=[2,4,2], tau=tau, prune=prune, integrity=integrity, vratio=vratio)
    
    @staticmethod
    def small_example(tau: float = 2.5, prune: bool = False, integrity: str = 'none', vratio: float = .7):
        """Factory static method generating a small 4-layer (3-6-6-3) backbone graph with custom options."""
        return SupplyChainBackbone(size=[3, 6, 6 ,3], beta=3, tau=tau, prune=prune, integrity=integrity, vratio=vratio)

    @staticmethod
    def medium_example(tau: float = 2.5, prune: bool = False, integrity: str = 'none', vratio: float = .8):
        """Factory static method generating a medium 5-layer (3-6-12-6-3) backbone graph with custom options."""
        return SupplyChainBackbone(size=[3,6,12,6,3], beta=4, tau=tau, prune=prune, integrity=integrity, vratio=vratio)


