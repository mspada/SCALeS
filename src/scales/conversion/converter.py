import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from string import ascii_uppercase
import random
import json
import os
import re
import time
from collections import defaultdict

from ..backbone import SupplyChainBackbone
from ..verboser import Verboser
from .. import config
from . import parameters


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


"""
This class allows converting an instance of an SCB (Supply Chain Backbone) graph into a static SupplyChain representation. 
During initialization, it takes the following arguments:
    - scb (SupplyChainBackbone): The SCB graph to be converted.
    - verbose (bool): Details the execution of each step when set to True.
    - output (str): 'png', 'dot', 'window', 'none' for drawing outputs. Default is 'png'.
    - seed (int): Global random seed to ensure reproducible generation across runs.

Tuning Parameters (paper §3.5, implemented in conversion/parameters.py):
    - a (float): Zipf distribution parameter for production rules.
    - sigma (float): Log-normal standard deviation for production capacities.
    - c_base (float): Base capacity required per outgoing connection.
    - pdelay_c, pdelay_l, pdelay_e (float): Complexity, layer, and efficiency tuning for production delay.
    - pcost_base, pcost_sigma (float): Base rate and variance for production costs.
    - storage_buffer, storage_sigma (float): Buffer limits and variance for local (I/O) storages.
    - t_layer_penalty, t_hub_base (float): Distance penalty and hub efficiency for transport edge delays.
    - t_over_prov, t_cap_sigma (float): Route over-provisioning scalar and variance for transport capacity.
    - t_cost_base, t_cost_sigma (float): Distance rate multiplier and variance for transport edge costs.
    - w_vol_mult, w_sigma (float): Multiplier and variance for systemic Warehouse node buffers.

Once initialized, it provides the following methods:
    - to_json(filename, verbose):
        Writes the topological description of the converted static supply chain into a JSON file.
        Default filename is supply_chains/supply_chain.json (see scales.config).
        Notice that `duration`, `supply` and `demand` are not handled here, but in the `SupplyChain` class directly.
    - to_json_str(verbose):
        Same as `to_json`, but returns the raw JSON string instead of writing to disk.
    - draw(filename):
        Draws the static base network of the resulting supply chain to the specified output format using `NetworkVisualizer`.
    - draw_node_diagnostic(label, filename):
        Draws a static zoomed diagnostic network view centered around a targeted backbone node (e.g., 'B14').
"""

class ScbConverter:

    def __init__(self, scb: SupplyChainBackbone, verbose=False, output='png', seed=None, 
                 a=2.0, 
                 sigma=0.2, c_base=100.0, 
                 pdelay_c=0.5, pdelay_l=1.0, pdelay_e=10.0,
                 pcost_base=100.0, pcost_sigma=0.10,
                 storage_buffer=4.0, storage_sigma=0.20,
                 t_layer_penalty=3.0, t_hub_base=10.0,
                 t_over_prov=1.25, t_cap_sigma=0.30,
                 t_cost_base=50.0, t_cost_sigma=0.10,
                 w_vol_mult=500.0, w_sigma=0.30): # Added output, seed
        if seed is not None and not isinstance(seed, int):
            raise TypeError(f"Seed must be an integer or None.")
        
        # Instance-bound random generators
        self._np_rng = np.random.default_rng(seed)
        self._std_rng = random.Random(seed)

        self._a = a
        self._sigma = sigma
        self._c_base = c_base
        self._pdelay_c = pdelay_c
        self._pdelay_l = pdelay_l
        self._pdelay_e = pdelay_e
        self._pcost_base = pcost_base
        self._pcost_sigma = pcost_sigma
        self._storage_buffer = storage_buffer
        self._storage_sigma = storage_sigma
        self._t_layer_penalty = t_layer_penalty
        self._t_hub_base = t_hub_base
        self._t_over_prov = t_over_prov
        self._t_cap_sigma = t_cap_sigma
        self._t_cost_base = t_cost_base
        self._t_cost_sigma = t_cost_sigma
        self._w_vol_mult = w_vol_mult
        self._w_sigma = w_sigma

        with Verboser(verbose, f"Initialising converter...") as verboser:
            start = time.time()
            self._scb = scb
            self._output = output # Stored as instance variable
            self._supplychain = self._supply_chain_from_scb(self._scb)
            end = time.time()
            verboser.terminate(f"SCB Converter successfully initialised. ({end - start:.2f}s)")

    # main methods
    def draw(self, filename=None):
        """
        Draws the base network of the resulting supply chain.
        - `filename` (str): Target path/filename to save the output diagram. Default is networks/supply_chains/supply_chain.
        """
        if filename is None:
            filename = str(config.networks_dir("supply_chains") / "supply_chain")
        # Inject self._output here
        self.draw_static_graph(supply_chain=self._supplychain, output=self._output, filename=filename, rng=self._std_rng)

    def to_json(self, filename: str = None, verbose=False) -> str:
        """
        Writes the description of the converted supply chain into the file with desired filename.
        - `filename` (str): Output path for the JSON file. Default is supply_chains/supply_chain.json.
        - `verbose` (bool): Logs output progress if True.
        Returns the path of the written file.
        """
        if filename is None:
            filename = str(config.supply_chains_dir() / "supply_chain.json")
        with Verboser(verbose, "Converting SCB (Supply Chain Backbone) graph...\n   -- Preparing data (nodes)...") as verboser:
            start = time.time() 

            json_data, num_nodes, num_edges = self._json_data(verboser)

            mid = time.time()
            verboser.update(f"Converting SCB (Supply Chain Backbone) graph...\n   -- Dumping data... ({num_nodes} nodes; {num_edges} edges)")
            
            out_name = filename if filename.endswith('.json') else f"{filename}.json"
            if os.path.dirname(out_name):
                os.makedirs(os.path.dirname(out_name), exist_ok=True)
            with open(out_name, 'w') as json_file:
                json.dump(json_data, json_file, cls=_NumpyEncoder)

            end = time.time()
            verboser.terminate(f"Conversion of SCB (Supply Chain Backbone) graph to Supply Chain successful. (data prep: {mid-start:.2f}s; data dump: {end - mid:.2f}s; total: {end - start:.2f}s)")
        return out_name

    def to_json_str(self, verbose=False):
        """
        Returns the description of the converted supply chain as a JSON formatted string.
        - `verbose` (bool): Logs output progress if True.
        """
        with Verboser(verbose, "Converting SCB (Supply Chain Backbone) graph...\n   -- Preparing data (nodes)...") as verboser:
            start = time.time() 

            json_data, num_nodes, num_edges = self._json_data(verboser)

            mid = time.time()
            verboser.update(f"Converting SCB (Supply Chain Backbone) graph...\n   -- Dumping data... ({num_nodes} nodes; {num_edges} edges)")
            json_data_str = json.dumps(json_data, cls=_NumpyEncoder)

            end = time.time()
            verboser.terminate(f"Conversion of SCB (Supply Chain Backbone) graph to Supply Chain successful. (data prep: {mid-start:.2f}s; data dump: {end - mid:.2f}s; total: {end - start:.2f}s)")
        return json_data_str

    # helper methods
    def _get_json_type(self, X, t, ntype):
        """Helper to map node properties to their final JSON node type."""
        if t == '(PI)':
            return 'PRODUCTION_INPUT'
        elif t == '(PO)':
            return 'PRODUCTION_OUTPUT'
        elif ntype == 'W':
            return 'WAREHOUSE'
        elif t.startswith('(I') or t.startswith('(O'):
            if X == 0 and t.startswith('(I'):
                return 'MATERIAL'
            elif X == len(self._scb.size()) - 1 and t.startswith('(O'):
                return 'OEM'
            else:
                return 'LOCAL_STORAGE'
        else:
            return 'LOCAL_STORAGE'

    def _json_data(self, verboser: Verboser) -> tuple[dict, int, int]:
        """Prepares and returns serialized JSON-compatible dictionary representation of the supply chain graph."""
        json_data = {}

        # nodes
        nodes = {}
        for (X, n, t), param in self._supplychain.nodes.items():
            ntype = param.get('type', 'P')
            w_id = param.get('w_id')
            if ntype == 'AS':
                continue # Skip AuxSource in JSON nodes if desired, or map it.

            # Map type key based on function
            out_type = self._get_json_type(X, t, ntype)

            rule_dict = {}
            if t == '(PI)' and 'prod_rules' in param and (X,n) in param['prod_rules']:
                rule_dict = { self._name_of_vertex(X, n, f"(I{k})", param.get('type', 'P'), w_id): int(v) for k, v in param['prod_rules'][(X,n)].items() }

            node_dict = {
                'type': out_type, 
                'rule': rule_dict, 
                'layer': param.get('layer', (X, 0))
            }
            if 'storage_cap' in param:
                node_dict['storage_cap'] = float(param['storage_cap'])
            if 'backbone_id' in param:
                node_dict['backbone_id'] = param['backbone_id']

            if ntype != 'AS':
                nodes[self._name_of_vertex(X,n,t,ntype,w_id)] = node_dict
        json_data['nodes'] = nodes

        # edges
        verboser.update("Converting SCB (Supply Chain Backbone) graph...\n   -- Preparing data (edges)...")
        edges = []
        for ((X,n,t), (Y,m,s)), param in self._supplychain.edges.items():
            if X == 'AS' or Y == 'AS':
                continue # AuxSource not in json
            cap, cost, edge_time = float(min(1e9, param['cap'])), float(param['cost']), int(param['time'])
            ntypeX = self._supplychain.nodes[(X,n,t)].get('type', 'P')
            w_idX = self._supplychain.nodes[(X,n,t)].get('w_id')
            ntypeY = self._supplychain.nodes[(Y,m,s)].get('type', 'P')
            w_idY = self._supplychain.nodes[(Y,m,s)].get('w_id')
            edges.append( { 'from': self._name_of_vertex(X,n,t,ntypeX,w_idX), 'to': self._name_of_vertex(Y,m,s,ntypeY,w_idY), 'cost': cost, 'cap': cap, 'time': edge_time } )
            if 'backbone_id' in param:
                edges[-1]['backbone_id'] = param['backbone_id']
            # edges.append( { 'from': (X,n,t), 'to': (Y,m,s), 'cost': cost, 'cap': cap, 'time': time } )
        json_data['edges'] = edges

        # deviations
        json_data['deviations'] = {}

        return json_data, len(nodes), len(edges)

    def _supply_chain_from_scb(self, scb: SupplyChainBackbone):
        """
        Main conversion method that transforms the input SCB graph into a full dynamic Supply Chain.
        Uses configured parameter generators to assign attributes and create structural nodes/edges.
        """
        # We need exactly one input local storage per incoming backbone edge.
        prod_rules = parameters.production_rules(self._np_rng, scb, a=self._a)

        out_edges = {n: list(scb.network.successors(n)) for n in scb.network.nodes}

        prod_caps = {}
        for X, n in scb.network.nodes:
            prod_caps[(X, n)] = parameters.production_capacity(self._np_rng, scb, (X, n), out_edges, sigma=self._sigma, c_base=self._c_base)

        nodes = []

        for X, n in scb.network.nodes:
            attrs = scb.network.nodes[(X, n)]
            node_type = attrs.get('type', 'P')
            w_id = attrs.get('w_id')
            w = attrs['weight']

            if node_type == 'W':
                cap_W = parameters.warehouse_capacity(self._np_rng, w, w_vol_mult=self._w_vol_mult, w_sigma=self._w_sigma)
                nodes.append( ((X, n, '(W)'), {'type': 'W', 'layer': (X, 1), 'w_id': w_id, 'storage_cap': cap_W}) )
                continue
            
            # Map P to PI and PO
            p_node_type_in = 'PI' if node_type == 'P' else node_type
            p_node_type_out = 'PO' if node_type == 'P' else node_type
            
            prod_cap = prod_caps[(X, n)]

            # Generate storage capacity for the nodes (like transport cap)
            # The weight defines the scale.
            
            if X == 0: 
                cap_I0 = parameters.local_storage_capacity(self._np_rng, prod_cap, storage_buffer=self._storage_buffer, storage_sigma=self._storage_sigma)
                nodes.append( ((X, n, f'(I0)'), {'type': 'I', 'layer': (X, 0), 'w_id': w_id, 'storage_cap': cap_I0}) )
            else: 
                for r in prod_rules[(X,n)]:
                    cap_Ir = parameters.local_storage_capacity(self._np_rng, prod_cap, storage_buffer=self._storage_buffer, storage_sigma=self._storage_sigma)
                    nodes.append( ((X, n, f'(I{r})'), {'type': 'I', 'layer': (X, 0), 'w_id': w_id, 'storage_cap': cap_Ir}) )
                    
            nodes.append( ((X, n, '(PI)'), { 'type': p_node_type_in, 'layer': (X, 1), 'prod_rules': prod_rules, 'w_id': w_id }) )
            nodes.append( ((X, n, '(PO)'), { 'type': p_node_type_out, 'layer': (X, 1.5), 'w_id': w_id }) )
            
            # Output local storages
            successors = out_edges[(X, n)]
            if len(successors) == 0:
                # Top layer or disconnected, add one default output storage
                cap_O0 = parameters.local_storage_capacity(self._np_rng, prod_cap, storage_buffer=self._storage_buffer, storage_sigma=self._storage_sigma)
                nodes.append( ((X, n, '(O0)'), { 'type': 'O', 'layer': (X, 2), 'w_id': w_id, 'storage_cap': cap_O0}) )
            else:
                for idx, (Y, m) in enumerate(successors):
                    cap_Oidx = parameters.local_storage_capacity(self._np_rng, prod_cap, storage_buffer=self._storage_buffer, storage_sigma=self._storage_sigma)
                    nodes.append( ((X, n, f'(O{idx})'), { 'type': 'O', 'layer': (X, 2), 'w_id': w_id, 'storage_cap': cap_Oidx}) )
            
        edges = []
        for X, n in scb.network.nodes: 
            node_type = scb.network.nodes[(X, n)].get('type', 'P')
            w = scb.network.nodes[(X, n)]['weight']

            if node_type != 'W':
                if X == 0: edges.append( ((X, n, f'(I0)'), (X, n, '(PI)'), {'time': 0, 'cost': 0, 'cap': float('inf')}) )
                else: 
                    edges.extend( [((X, n, f'(I{r})'), (X, n, '(PI)'), {'time': 0, 'cost': 0, 'cap': float('inf')}) 
                                    for r in prod_rules[(X,n)]] )

                edge_time = parameters.production_delay(self._np_rng, scb, (X, n), w, pdelay_c=self._pdelay_c, pdelay_l=self._pdelay_l, pdelay_e=self._pdelay_e)
                cost = parameters.production_cost(self._np_rng, w, edge_time, pcost_base=self._pcost_base, pcost_sigma=self._pcost_sigma)
                cap = prod_caps[(X, n)]
                
                # PI -> PO edge
                edges.append( ((X, n, '(PI)'), (X, n, '(PO)'), {'time': edge_time, 'cost': cost, 'cap': cap}) )
                
                successors = out_edges[(X, n)]
                if len(successors) == 0:
                    edges.append( ((X, n, '(PO)'), (X, n, '(O0)'), {'time': 0, 'cost': 0, 'cap': float('inf')}) )
                else:
                    for idx, (Y, m) in enumerate(successors):
                        # PO -> Output Storage
                        edges.append( ((X, n, '(PO)'), (X, n, f'(O{idx})'), {'time': 0, 'cost': 0, 'cap': float('inf')}) )

            # With one input local storage per incoming edge, we directly map each predecessor to one r
            # Let's map predecessor r to input storage I{r}
            preds = list(scb.network.predecessors((X, n)))
            for r, pred in enumerate(preds):
                Y, m = pred
                y_node_type = scb.network.nodes[(Y, m)].get('type', 'P')
                w_X = scb.network.nodes[(X, n)]['weight']
                w_Y = scb.network.nodes[(Y, m)]['weight']
                
                # Source node
                if y_node_type == 'W':
                    src = (Y, m, '(W)')
                else:
                    y_successors = out_edges[(Y, m)]
                    out_idx = y_successors.index((X, n))
                    src = (Y, m, f'(O{out_idx})')
                
                # Target node
                if node_type == 'W':
                    dst = (X, n, '(W)')
                else:
                    dst = (X, n, f'(I{r})')
                
                edge_time = parameters.transport_time(self._np_rng, X, Y, w_X, w_Y, t_layer_penalty=self._t_layer_penalty, t_hub_base=self._t_hub_base)
                src_prod_cap = prod_caps.get((Y, m), 1.0)
                cap = parameters.transport_capacity(self._np_rng, (Y, m), src_prod_cap, out_edges, t_over_prov=self._t_over_prov, t_cap_sigma=self._t_cap_sigma)
                cost = parameters.transport_cost(self._np_rng, edge_time, cap, t_cost_base=self._t_cost_base, t_cost_sigma=self._t_cost_sigma)
                
                edges.append( (src, dst, {'time': edge_time, 'cost': cost, 'cap': cap}) )
    
        for i in range(len(nodes)):
            n_id, n_data = nodes[i]
            n_data['backbone_id'] = [n_id[0], n_id[1]]

        for i in range(len(edges)):
            src, dst, e_data = edges[i]
            e_data['backbone_id'] = [src[0], src[1]]

        supply_chain = nx.DiGraph()
        supply_chain.add_nodes_from(nodes)
        supply_chain.add_edges_from(edges)
        return supply_chain

    @staticmethod
    def draw_static_graph(supply_chain: nx.DiGraph, output: str = 'png', filename: str = None, rng=None) -> None:
        """Utility method to draw the converted static supply chain graph, ignoring AuxSource nodes."""
        if filename is None:
            filename = str(config.networks_dir("supply_chains") / "supply_chain")
        # Filter out AuxSource
        draw_nodes = [n for n in supply_chain.nodes if supply_chain.nodes[n].get('type') != 'AS']
        draw_graph = supply_chain.subgraph(draw_nodes).copy()
        
        # Ensure all nodes have a 'layer' attribute for multipartite_layout
        for n, d in draw_graph.nodes(data=True):
            if 'layer' not in d:
                # Assign a default layer if missing, e.g., based on node structure
                d['layer'] = (n[0] if isinstance(n, tuple) else 0, 0)

        def get_layer_X(n, d):
            if isinstance(n, tuple):
                return n[0]
            layer = d.get('layer')
            if layer and isinstance(layer, (list, tuple)):
                return layer[0]
            return 0
            
        def get_label(n, d):
            if isinstance(n, tuple):
                return f"{ascii_uppercase[n[0]]}{n[1]}{n[2]}"
            return str(n)

        from ..viz.visualiser import NetworkVisualizer
        NetworkVisualizer.draw(
            network=draw_graph,
            output=output,
            filename=filename,
            node_size=100,
            figsize=(14, 8),
            font_size=8,
            rng=rng,
            pos_shift=lambda n, p: (p[0] + get_layer_X(n, draw_graph.nodes[n])/4, p[1]),
            node_label_func=get_label,
            node_layer_func=lambda n, d: tuple(d.get('layer', (0,0))),
            is_warehouse_func=lambda n, d: d.get('type') in ['WAREHOUSE', 'W']
        )

    def draw_node_diagnostic(self, label: str, filename=None, output=None):
        """
        Draws the supply chain structure around a specific backbone node (e.g., 'B14').
        Shows inputs, production, outputs, and connections to other nodes.
        Default filename is networks/diagnostics/static/diagnostic_<label>.
        """
        try:
            layer_char = label[0].upper()
            X = ascii_uppercase.index(layer_char)
            n = int(label[1:])
        except (ValueError, IndexError):
            raise ValueError(f"Invalid label format: '{label}'. Expected format like 'B14'.")

        # Find internal nodes for this backbone node
        internal_nodes = [node for node in self._supplychain.nodes if node[0] == X and node[1] == n]
        if not internal_nodes:
            raise KeyError(f"Node {label} not found in the supply chain.")

        # Gather subgraph nodes: internal + immediate predecessors + immediate successors
        subgraph_nodes = set(internal_nodes)
        for node in internal_nodes:
            subgraph_nodes.update(self._supplychain.predecessors(node))
            subgraph_nodes.update(self._supplychain.successors(node))
        
        # Don't show AS (AuxSource) to keep it clean unless requested, actually let's keep it to show the infinite supply
        
        G = self._supplychain.subgraph(subgraph_nodes)
        
        plt.figure(figsize=(16, 10))
        
        # Determine layout
        pos = {}
        # Columns:
        # -1: Predecessors
        # 0: Inputs (I)
        # 1: PI
        # 2: PO
        # 3: Outputs (O)
        # 4: Successors
        
        counts = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        nodes_by_col = defaultdict(list)
        
        for node in G.nodes():
            if node in internal_nodes:
                t = node[2]
                if t.startswith('(I'): col = 0
                elif t == '(PI)': col = 1
                elif t == '(PO)': col = 2
                elif t.startswith('(O'): col = 3
                else: col = 1.5
            else:
                # Neighbor
                if any(G.has_edge(node, inner) for inner in internal_nodes):
                    col = -1
                else:
                    col = 4
            nodes_by_col[col].append(node)
            
        def natural_sort_key_conv(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]
            
        for col, nodes_in_col in nodes_by_col.items():
            nodes_in_col.sort(key=lambda x: natural_sort_key_conv(x))
            y_offset = len(nodes_in_col) / 2.0
            for idx, node in enumerate(nodes_in_col):
                pos[node] = (col * 3, y_offset - idx)
                
        # Node labels
        node_labels = {}
        for node in G.nodes():
            if node[0] == 'AS':
                node_labels[node] = 'AS'
            else:
                lbl = f"{ascii_uppercase[node[0]]}{node[1]}{node[2]}"
                
                # Get the JSON type
                ntype = G.nodes[node].get('type', 'P')
                json_type = self._get_json_type(node[0], node[2], ntype)
                lbl += f"\n{json_type}"
                
                if 'storage_cap' in G.nodes[node]:
                    cap = G.nodes[node]['storage_cap']
                    if cap != float('inf'):
                        lbl += f"\ncap:{cap:.1f}"
                node_labels[node] = lbl
                
        # Edge labels
        edge_labels = {}
        for u, v, d in G.edges(data=True):
            lbl = f"t:{d.get('time',0)}\nc:{d.get('cost',0):.2f}"
            cap = d.get('cap', float('inf'))
            if cap != float('inf'):
                lbl += f"\ncap:{cap:.2f}"
            edge_labels[(u, v)] = lbl

        node_colors = []
        for node in G.nodes():
            if node in internal_nodes:
                if node[2].startswith('(I'): node_colors.append('lightblue')
                elif node[2] == '(PI)': node_colors.append('orange')
                elif node[2] == '(PO)': node_colors.append('coral')
                elif node[2].startswith('(O'): node_colors.append('lightgreen')
                elif node[2] == '(W)': node_colors.append('red')
                else: node_colors.append('grey')
            else:
                if node[2] == '(W)': node_colors.append('lightcoral')
                else: node_colors.append('lightgrey')
                
        nx.draw(G, pos, with_labels=True, labels=node_labels, node_color=node_colors, 
                node_size=2000, font_size=8, font_weight='bold', arrows=True,
                connectionstyle='arc3,rad=0.1')
                
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
        
        plt.title(f"Diagnostic view for Backbone Node {label}")
        plt.axis('off')
        
        # Add text block with attributes and production rules at the bottom
        internal_pi = [node for node in internal_nodes if node[2] == '(PI)']
        internal_w = [node for node in internal_nodes if node[2] == '(W)']
        
        if internal_pi or internal_w:
            rules_text = f"Backbone Node {label} Attributes:\n"
            if (X, n) in self._scb.network.nodes:
                bb_attrs = self._scb.network.nodes[(X, n)]
                for k, v in bb_attrs.items():
                    if isinstance(v, float):
                        rules_text += f"{k}: {v:.2f}\n"
                    else:
                        rules_text += f"{k}: {v}\n"
            
            if internal_pi:
                pi_node = internal_pi[0]
                rules_text += f"\nProduction Node (PI) Attributes:\n"
                pi_attrs = self._supplychain.nodes[pi_node]
                for k, v in pi_attrs.items():
                    if k != 'prod_rules':
                        if isinstance(v, float):
                            rules_text += f"{k}: {v:.2f}\n"
                        else:
                            rules_text += f"{k}: {v}\n"
                
                prod_rules = pi_attrs.get('prod_rules', {})
                if (X, n) in prod_rules and prod_rules[(X, n)]:
                    rules_text += "\nProduction Rules:\n"
                    for r, val in prod_rules[(X, n)].items():
                        rules_text += f"I{r} needed: {val}\n"
            
            elif internal_w:
                w_node = internal_w[0]
                rules_text += f"\nWarehouse Node (W) Attributes:\n"
                w_attrs = self._supplychain.nodes[w_node]
                for k, v in w_attrs.items():
                    if isinstance(v, float):
                        rules_text += f"{k}: {v:.2f}\n"
                    else:
                        rules_text += f"{k}: {v}\n"
                    
            # Add text box to the plot
            plt.text(0.5, 0.05, rules_text.strip(), transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='bottom', horizontalalignment='center',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

        out = output if output is not None else self._output
        if out in ('png', 'svg', 'save', 'save_and_show'):
            if not filename:
                filename = str(config.networks_dir("diagnostics", "static") / f"diagnostic_{label}")
            out_dir = os.path.dirname(filename)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            if out == 'svg':
                plt.savefig(filename if filename.endswith('.svg') else f"{filename}.svg", format="svg", bbox_inches='tight')
            else:
                plt.savefig(filename if filename.endswith('.png') else f"{filename}.png", dpi=300, bbox_inches='tight')
            
        if out in ('window', 'show', 'save_and_show'):
            plt.show()
            
        if out in ('png', 'svg', 'save', 'none'):
            plt.close()

    def get_node_from_label(self, label: str):
        """
        Takes a string label (e.g., 'A5(P)', 'C12(I0)') and returns the 
        corresponding node tuple (X, n, t) if it exists in the supply chain.
        """
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
    def _name_of_vertex(X: int, n: int, t: str, node_type: str = 'P', w_id: int = None) -> str:
        """Determines and returns the string identifier/label of a supply chain node tuple."""
        return ascii_uppercase[X] + str(n) + str(t)

