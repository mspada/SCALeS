import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import CSS4_COLORS
from itertools import product, combinations
from scb import SupplyChainBackbone
from string import ascii_uppercase
import random
import json
import time
import threading
from networkx.drawing.nx_pydot import write_dot

from verboser import Verboser

"""
This class allows to convert an instance of a SCB (Supply Chain Backbone) graph to a SupplyChain. 
During initialisation, may take two arguments described below:
    - chunglu (SupplyChainBackbone): the SCB (Supply Chain Backbone) graph to be converted.
    - verbose (bool): details the execution of each step when set to True.

Once initialised, provides methods. The most useful is:
    - to_json(filename, duration, demand, supply, verbose)
        This writes the description of the converted supply chain into the file with desired filename.
        Parameter duration is a positive int, and denotes the number of time steps to consider for this supply chain.
        Parameter supply is a float, and denotes the supply at each step of every node of the first layer.    
        Parameter demand is a float, and denotes the demand at each step of every node of the last layer.    
        Parameter verbose is a bool, and details the execution when set to True.
    - draw_chunglu(): draws the original SCB (Supply Chain Backbone) graph.
    - draw_supplychain(): draws the base network of the resulting supply chain.
    - output, same as chunglu.py
    - seed, same as chunglu.py. Results tested - same dot, same json for the same seed.
"""

# TODO -- Implement more general conversions (may require reworking on the SupplyChain Class)

class ChungLuConverter:

    def __init__(self, chunglu: SupplyChainBackbone, verbose=False, output='png', seed=None): # Added output, seed
        if seed is not None and not isinstance(seed, int):
            raise TypeError(f"Seed must be an integer or None.")
        
        # Instance-bound random generators
        self._np_rng = np.random.default_rng(seed)
        self._std_rng = random.Random(seed)

        with Verboser(verbose, f"Initialising converter...") as verboser:
            start = time.time()
            self._chunglu = chunglu
            self._output = output # Stored as instance variable
            self._supplychain = self._supply_chain_from_chunglu(self._chunglu)
            end = time.time()
            verboser.terminate(f"Chung-Lu Converter successfully initialised. ({end - start:.2f}s)")

    # main methods
    def draw_chunglu(self):
        self._chunglu.draw()
    
    def draw_supplychain(self, filename='networks/supply_chain'):
        # Inject self._output here
        self._draw_partite_supply_chain(supply_chain=self._supplychain, output=self._output, filename=filename)

    def to_json(self, filename: str, duration=20, demand=10, supply=1e9, verbose=False):
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)

        with Verboser(verbose, "Converting SCB (Supply Chain Backbone) graph...\n   -- Preparing data (nodes)...") as verboser:
            start = time.time() 

            json_data, num_nodes, num_edges = self._json_data(verboser, duration=duration, demand=demand, supply=supply)

            mid = time.time()
            verboser.update(f"Converting SCB (Supply Chain Backbone) graph...\n   -- Dumping data... ({num_nodes} nodes; {num_edges} edges)")
            with open(filename, 'w') as json_file:
                json.dump(json_data, json_file, cls=NumpyEncoder)

            end = time.time()
            verboser.terminate(f"Conversion of SCB (Supply Chain Backbone) graph to Supply Chain successful. (data prep: {mid-start:.2f}s; data dump: {end - mid:.2f}s; total: {end - start:.2f}s)")

    def to_json_str(self, duration=20, demand=10, supply=1e9, verbose=False):
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)

        with Verboser(verbose, "Converting SCB (Supply Chain Backbone) graph...\n   -- Preparing data (nodes)...") as verboser:
            start = time.time() 

            json_data, num_nodes, num_edges = self._json_data(verboser, duration=duration, demand=demand, supply=supply)

            mid = time.time()
            verboser.update(f"Converting SCB (Supply Chain Backbone) graph...\n   -- Dumping data... ({num_nodes} nodes; {num_edges} edges)")
            json_data_str = json.dumps(json_data, cls=NumpyEncoder)

            end = time.time()
            verboser.terminate(f"Conversion of SCB (Supply Chain Backbone) graph to Supply Chain successful. (data prep: {mid-start:.2f}s; data dump: {end - mid:.2f}s; total: {end - start:.2f}s)")
        return json_data_str

    # helper methods
    def _json_data(self, verboser, duration=20, demand=20, supply=1e9):
        json_data = {}

        # duration
        json_data['duration'] = duration

        # nodes
        nodes = {}
        nodes['OEM'] = {'type': 'OEM', 'layer': (len(self._chunglu.size()), 0)}
        for (X, n, t), param in self._supplychain.nodes.items():
            ntype = param.get('node_type', 'P')
            w_id = param.get('w_id')
            if t == '(P)':
                nodes[self._name_of_vertex(X,n,t,ntype,w_id)] = {'type': 'PRODUCTION', 
                                                    'rule': { self._name_of_vertex(X, n, f"(I{k})", ntype, w_id): int(v) for k, v in param['prod_rules'][(X,n)].items() } if (X,n) in param.get('prod_rules', {}) else {}, 
                                                    'layer': (X, 1)}
            else: 
                nodes[self._name_of_vertex(X,n,t,ntype,w_id)] = {'type': 'LOCAL_STORAGE',
                                                    'layer': (X, 2 if t=='(O)' else 0) }
        json_data['nodes'] = nodes

        # edges
        verboser.update("Converting SCB (Supply Chain Backbone) graph...\n   -- Preparing data (edges)...")
        edges = []
        for ((X,n,t), (Y,m,s)), param in self._supplychain.edges.items():
            cap, cost, edge_time = float(min(1e9, param['cap'])), float(param['cost']), int(param['time'])
            ntypeX = self._supplychain.nodes[(X,n,t)].get('node_type', 'P')
            w_idX = self._supplychain.nodes[(X,n,t)].get('w_id')
            ntypeY = self._supplychain.nodes[(Y,m,s)].get('node_type', 'P')
            w_idY = self._supplychain.nodes[(Y,m,s)].get('w_id')
            edges.append( { 'from': self._name_of_vertex(X,n,t,ntypeX,w_idX), 'to': self._name_of_vertex(Y,m,s,ntypeY,w_idY), 'cost': cost, 'cap': cap, 'time': edge_time } )
            # edges.append( { 'from': (X,n,t), 'to': (Y,m,s), 'cost': cost, 'cap': cap, 'time': time } )
        for X,n,t in self._supplychain.nodes: 
            if X == len(self._chunglu.size()) - 1 and t == '(O)':
                ntype = self._supplychain.nodes[(X,n,t)].get('node_type', 'P')
                w_id = self._supplychain.nodes[(X,n,t)].get('w_id')
                edges.append( {'from': self._name_of_vertex(X,n,t,ntype,w_id), 'to': 'OEM', 'cost': 0, 'cap': 1e9, 'time': 0} )
        json_data['edges'] = edges

        # demand
        verboser.update("Converting SCB (Supply Chain Backbone) graph...\n   -- Preparing data (demands)...")
        json_data['demand'] = [{'node': 'OEM', 'time': 0, 'demand': demand}]

        # supply
        verboser.update("Converting SCB (Supply Chain Backbone) graph...\n   -- Preparing data (supply)...")
        json_supply = [] 
        for (X, n, t) in self._supplychain.nodes:
            if X == 0 and t == '(I0)':
                ntype = self._supplychain.nodes[(X,n,t)].get('node_type', 'P')
                w_id = self._supplychain.nodes[(X,n,t)].get('w_id')
                json_supply.append( {'node': self._name_of_vertex(X,n,t,ntype,w_id), 'time': 0, 'supply': supply} )
        json_data['supply'] = json_supply

        # deviations
        json_data['deviations'] = {}

        return json_data, len(nodes), len(edges)

    # static methods
    def _supply_chain_from_chunglu(self, chunglu: SupplyChainBackbone):
        prod_rules = { (X, n): { r: self._np_rng.geometric(p=.7) 
                            for r in range( self._np_rng.integers(low=1, high=1+chunglu.network.in_degree((X,n))) ) }
                            if chunglu.network.in_degree((X,n)) > 0 else {} 
                            for X, n in chunglu.network.nodes }

        nodes = []
        for X, n in chunglu.network.nodes:
            attrs = chunglu.network.nodes[(X, n)]
            node_type = attrs.get('node_type', 'P')
            w_id = attrs.get('w_id')
            if X == 0: nodes.append( ((X, n, f'(I0)'), {'type': 'I', 'layer': (X, 0), 'node_type': node_type, 'w_id': w_id}) )
            else: nodes.extend( [ ((X, n, f'(I{r})'), {'type': 'I', 'layer': (X, 0), 'node_type': node_type, 'w_id': w_id}) for r in prod_rules[(X,n)] ] )
            nodes.append( ((X, n, '(P)'), { 'type': 'P', 'layer': (X, 1), 'prod_rules': prod_rules, 'node_type': node_type, 'w_id': w_id }) )
            nodes.append( ((X, n, '(O)'), { 'type': 'O', 'layer': (X, 2), 'node_type': node_type, 'w_id': w_id}) )
        edges = []
        for X, n in chunglu.network.nodes: 
            if X == 0: edges.append( ((X, n, f'(I0)'), (X, n, '(P)'), {'time': 0, 'cost': 0, 'cap': float('inf')}) )
            else: edges.extend( [((X, n, f'(I{r})'), (X, n, '(P)'), {'time': 0, 'cost': 0, 'cap': float('inf')}) 
                                for r in prod_rules[(X,n)]] )
            edge_time = int(np.ceil(self._np_rng.gamma( shape=1+X ) / (chunglu.network.nodes[(X,n)]['weight'] ** self._np_rng.uniform(.1,.9)))) # TODO -- Choice of C_0 and a is a bit arbitrary 
            cost = self._np_rng.gamma( shape=np.sqrt(1+X) ) / (chunglu.network.nodes[(X,n)]['weight'] ** self._np_rng.uniform(.1,.9)) # TODO -- Choice of C_0 and b is a bit arbitrary
            cap  = self._np_rng.gamma( shape = chunglu.network.nodes[(X, n)]['weight'] )  
            edges.append(  ((X, n, '(P)'), (X, n, '(O)'), {'time': edge_time, 'cost': cost, 'cap': cap}) )

            supply_chain_inneighbours = self._attribute_supply_chain_inneighbours( 
                                                list(chunglu.network.predecessors((X, n))), len(prod_rules[(X,n)]))
            for r, preds in enumerate(supply_chain_inneighbours):
                for (Y, m) in preds:
                    edge_time = int(self._np_rng.geometric( 1 / np.sqrt( chunglu.network.nodes[(X,n)]['weight'] * chunglu.network.nodes[(Y, m)]['weight'] ) ))
                    cost = self._np_rng.gamma(shape=np.sqrt(edge_time)) 
                    cap  = self._np_rng.gamma(shape=edge_time)
                    # TODO -- Add possibility of going backwards
                    edges.append( ((Y, m, '(O)'), (X, n, f'(I{r})'), {'time': edge_time, 'cost': cost, 'cap': cap}) )
    
        supply_chain = nx.DiGraph()
        supply_chain.add_nodes_from(nodes)
        supply_chain.add_edges_from(edges)
        return supply_chain

    def _draw_partite_supply_chain(self, supply_chain, output='png', filename='networks/supply_chain'):
        layers = set( supply_chain.nodes[node]['layer'] for node in supply_chain.nodes ) 
        # Added sorted() to enforce consistent tuple iteration
        col_layers = { layer: self._std_rng.choice(list(CSS4_COLORS.keys())) for layer in sorted(layers) }
        
        if output in ('png', 'window'):
            col = [ 'red' if supply_chain.nodes[node].get('node_type') == 'W' else col_layers[ supply_chain.nodes[node]['layer'] ] for node in supply_chain.nodes ]
            lab = { (X, n, t): f"W{supply_chain.nodes[(X,n,t)].get('w_id')}{t}" if supply_chain.nodes[(X,n,t)].get('node_type') == 'W' else ascii_uppercase[X] + str(n) + str(t) for X, n, t in supply_chain.nodes }
            pos = nx.multipartite_layout(G=supply_chain, subset_key="layer")
            pos = { (X,n,t): (x + X/4,y) for (X,n,t), (x,y) in pos.items() }
            
            plt.figure(figsize=(14,8))
            nx.draw(G=supply_chain, pos=pos, node_color=col, node_size=100, labels=lab, with_labels=True, connectionstyle='arc3, rad=0.1', font_size=8)
            
            if output == 'png':
                out_name = filename if filename.endswith('.png') else f"{filename}.png"
                plt.savefig(out_name, format="png", dpi=300)
                plt.close()
            else:
                plt.show()
                
        elif output == 'dot':
            for node in supply_chain.nodes:
                X, n, t = node
                if supply_chain.nodes[node].get('node_type') == 'W':
                    supply_chain.nodes[node]['label'] = f"W{supply_chain.nodes[node].get('w_id')}{t}"
                    supply_chain.nodes[node]['fillcolor'] = 'red'
                else:
                    supply_chain.nodes[node]['label'] = f"{ascii_uppercase[X]}{n}{t}"
                    supply_chain.nodes[node]['fillcolor'] = col_layers[supply_chain.nodes[node]['layer']]
                supply_chain.nodes[node]['style'] = "filled"
                # Grouping by tuple layer (X, type_index) ensures proper hierarchical alignment in Graphviz
                supply_chain.nodes[node]['group'] = str(supply_chain.nodes[node]['layer'])
                
            out_name = filename if filename.endswith('.dot') else f"{filename}.dot"
            write_dot(supply_chain, out_name)

    def _attribute_supply_chain_inneighbours(self, inneighbours, num_input_storage):
        n = len(inneighbours)
        if num_input_storage <= 1: return [inneighbours]     
        cut_pos = [-1] + sorted( self._np_rng.choice( len(inneighbours)-1, num_input_storage-1, replace=False) ) + [n-1]
        degrees = [ cut_pos[i+1] - cut_pos[i] for i in range(num_input_storage) ]
        supply_chain_inneighbours, rem_inneighbours = [], set(range(n))
        # old version
        # for d in degrees:
        #    supply_chain_inneighbours.append( self._np_rng.choice( list(rem_inneighbours), d, replace=False ) )
        #    rem_inneighbours -= set(supply_chain_inneighbours[-1])
        # fix Python's Hash Randomization
        for d in degrees:
            # Added sorted() to guarantee strict order regardless of the set's internal hash state
            chosen = self._np_rng.choice( sorted(list(rem_inneighbours)), d, replace=False )
            supply_chain_inneighbours.append( chosen )
            rem_inneighbours -= set(chosen)
        return [ [inneighbours[i] for i in N] for N in supply_chain_inneighbours ] 
    
    def get_node_from_label(self, label: str):
        """
        Takes a string label (e.g., 'A5(P)', 'C12(I0)') and returns the 
        corresponding node tuple (X, n, t) if it exists in the supply chain.
        """
        from string import ascii_uppercase
        
        try:
            # 1. The first character is the layer letter
            layer_char = label[0].upper()
            X = ascii_uppercase.index(layer_char)
            
            # 2. Find where the parenthesis starts to separate the number from the type
            paren_index = label.index('(')
            
            # 3. Extract the node index (everything between the letter and the parenthesis)
            n = int(label[1:paren_index])
            
            # 4. Extract the type suffix (everything from the parenthesis to the end)
            t = label[paren_index:]
            
        except (ValueError, IndexError):
            raise ValueError(f"Invalid label format: '{label}'. Expected format like 'A5(P)' or 'C12(I0)'.")
            
        node_tuple = (X, n, t)
        
        # Verify the reconstructed node exists in the converted supply chain graph
        if node_tuple in self._supplychain.nodes:
            return node_tuple
        else:
            raise KeyError(f"Node {node_tuple} from label '{label}' does not exist in the converted network.")

    @staticmethod
    def _name_of_vertex(X, n, t, node_type='P', w_id=None):
        if node_type == 'W':
            return f"W{w_id}{t}"
        return ascii_uppercase[X] + str(n) + str(t)

if __name__ == '__main__':
    cl = SupplyChainBackbone(size=[3,6,6])
    clc = ChungLuConverter(cl)
    clc.draw_chunglu()
    clc.draw_supplychain()
    
    # cl = SupplyChainBackbone.tiny_example(prune=True, integrity='full')
    # conv = ChungLuConverter(cl)
    # conv.to_json('test_conversion.json')
    # conv.draw_supplychain(save=True, filename='test_conversion.png')
