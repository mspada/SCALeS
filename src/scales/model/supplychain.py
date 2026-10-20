from collections import defaultdict
from itertools import product
import json
import time

import networkx as nx
import numpy as np

from ..backbone import SupplyChainBackbone
from ..conversion import ScbConverter
from ..verboser import Verboser
from .. import config
from .._utils import natural_sort_key
from .ampl import AmplExportMixin
from . import solver
from ..analysis.diagnostics import DiagnosticsMixin
from ..viz.drawing import DrawingMixin


"""
This class represents a dynamic Flow Supply Chain, managing the time-expanded constraints and LP logic (paper §3).
During initialisation, it takes several arguments described below:
    - JSONfilename (str): the filename from which the static topology should be loaded (as written by ScbConverter.to_json).
        Ignored when providing a non-None parameter `supply_chain` or `data`.
    - supply_chain (SupplyChain): an existing supply chain instance to copy from.
        Default is None.
    - data (dict): the static topology as a dictionary with keys "nodes" and "edges" (same layout as the JSON file).
        Ignored when providing a non-None parameter `supply_chain`. See also `SupplyChain.from_converter`.
    - duration (int): The number of discrete time steps to expand the dynamic network over. Both 
        duration and delays operate in the exact same unit: discrete time steps.  duration (Absolute Bound)is an absolute integer defining the total number of discrete time steps the dynamic network is expanded over. It establishes the global bounds of the simulation, generating nodes from $t=0$ up to $t=\text{duration}-1$. Delays (Relative Steps): Delays (production and transport times) represent the relative number of discrete steps required to traverse an operation. If an edge has a delay of edge_time, flow entering at time t will emerge at time t + edge_time. 
    - supply (float | list | ndarray): Supply configuration for `MATERIAL` (Layer 0) nodes.
        Can be scalar, 1D (time), or 2D (node x time). If a scalar, the code applies a constant, static value to all target nodes across every time step in the simulation. If 1D array, the code applies a time-varying value that is identical across all target nodes. The array index maps directly to the time step t. If 2D array, the code applies specific time-varying values to specific individual nodes. The row index maps to the naturally sorted node index (A1, A2, A12), and the column index maps to the time step $t$. Padding: if the provided 1D or 2D arrays are smaller than the network's total duration or the total number of target nodes, the code automatically caps the index (e.g., min(t, len(config) - 1)), effectively padding the remainder of the simulation or the remaining nodes with the last provided value in the array.
    - demand (float | list | ndarray): Demand configuration for `OEM` (Top Layer) nodes.
        Can be scalar, 1D (time), or 2D (node x time). The semantics in case of scalar, 1D and 2D is the same as for supply except that it applies to OEM nodes in top layer.
    - verbose (bool): details the execution of each step when set to True.
    - state (tuple): A two-integer tuple formatted as (current_step, total_steps) (defaulting to (1, 1)). It is used exclusively by the logging system to display progress tracking when the optimization is executed repeatedly within a loop (such as during perturbation analysis).
    - save_temp (str or bool): if provided, the `.dat` AMPL data generated for solving is preserved on disk under this base path
        (`True` uses results/lp/model) instead of utilizing auto-deleting temporary files in temp/.
    - lp_model_path (str): path to the AMPL `.mod` LP constraints file. Default is the stoichiometric model shipped with the package.
    - solve (bool): if True, find_optimum() is executed at the end of initialisation. Default is False, so that a network can
        be built, inspected, perturbed or drawn without a solver; cost() and production() solve on demand.
    
Once initialised, provides several methods. The most useful are:
    - find_optimum(): computes the optimal cost allocation utilizing BACKORDER soft-constraints logic with GLPK.
    - copy(): returns an independent copy of this supply chain (unsolved).
    - cost(), production(), optimisation_time(): results of the optimisation (solving first if needed).
    - to_ampl_data(): the AMPL data (.dat) for the LP, e.g. to use another solver.
    - set_production / get_production, set_transport / get_transport, set_local / get_local: change or read parameters.
    - draw_dynamic, draw_static, draw_node_diagnostic: drawings (see viz/drawing.py), saved below networks/.
    - lp_node_diagnostic, layer_flow_input, layer_flow_output, diagnostic_backorder, diagnostic_generation,
      diagnostic_absorption: reports and CSV tables (see analysis/diagnostics.py), saved below results/diagnostics/.

The class also provides some static methods for easy generation of full end-to-end SCB models with preset parameters.
"""

TO_SELF_PENALTY = .001


class SupplyChain(AmplExportMixin, DiagnosticsMixin, DrawingMixin):

    def __init__(self, JSONfilename=None, supply_chain=None, data=None, verbose=False, duration=20, supply=1e9, demand=10,
                 output='png', state=(1, 1), save_temp=None, lp_model_path=None, solve=False) -> None:
        # initialises the internal _network representation from another supply chain, a data dictionary or a JSON file (in this order of preference)

        with Verboser(verbose, "Creating Supply Chain...") as verboser:
            start = time.time()
            if supply_chain:
                verboser.update("Creating Supply Chain...\n   -- Initialising from other Supply Chain...")
                duration, nodes, edges, demand, supply = supply_chain.duration, supply_chain.nodes, supply_chain.edges, supply_chain.demand, supply_chain.supply
                self._output = getattr(supply_chain, '_output', output)
                lp_model_path = lp_model_path or getattr(supply_chain, '_lp_model_path', None)
            elif data is not None:
                nodes, edges = self._parse_data(data)
                self._output = output
            elif JSONfilename is not None:
                verboser.update("Creating Supply Chain...\n   -- Loading from json file...")
                nodes, edges = self._dataFromJSON(JSONfilename)
                self._output = output
            else:
                raise ValueError("SupplyChain needs one of `JSONfilename`, `data` or `supply_chain`.")

            self._lp_model_path = lp_model_path or config.LP_MODEL_PATH
            self.duration, self.nodes, self.edges, self.demand, self.supply = duration, nodes, edges, demand, supply
            verboser.update("Creating Supply Chain...\n   -- Building base network...")
            self._baseNetworkFromData(duration, nodes, edges, demand, supply)
            
            # Analyze critical propagation delay and warn the user if duration is insufficient
            self._critical_delay = self.calculate_critical_propagation_delay()
            if self.duration < self._critical_delay:
                print(f"\n[WARNING] The specified simulation duration ({self.duration}) is shorter than "
                      f"the minimum critical propagation delay ({self._critical_delay} steps) required to complete production "
                      f"from raw materials to the OEM. Optimization will yield 0 flow and accumulate backorders.\n")
                      
            verboser.update("Creating Supply Chain...\n   -- Building network...")
            self._networkFromData(duration, nodes, edges, demand, supply)
            self._reset_results()
            
            self._verbose = verbose
            self._state = state
            self._save_temp = save_temp

            end = time.time()
            verboser.terminate(f"Supply Chain successfully generated. ({end - start:.2f}s)")

        if solve:
            self.find_optimum()

    @classmethod
    def from_converter(cls, converter: ScbConverter, **kwargs) -> "SupplyChain":
        """Builds a SupplyChain directly from an ScbConverter, without writing a JSON file. kwargs are passed to __init__."""
        return cls(data=json.loads(converter.to_json_str()), **kwargs)

    def copy(self) -> "SupplyChain":
        """Returns a new, unsolved SupplyChain with the same topology, parameters, supply and demand."""
        return SupplyChain(supply_chain=self)

    # main methods
    def find_optimum(self) -> None:
        """Solves the LP with glpsol and stores flows and GENERATION/ABSORPTION/BACKORDER values in the network."""
        cur, tot = self._state

        with Verboser(self._verbose, f"Computing optimum cost allocation... ({cur}/{tot})") as verboser:
            start = time.time()
            verboser.update(f"Computing optimum cost allocation... ({cur}/{tot})\n   -- Generating LP data...")
            data = self.to_ampl_data()
            verboser.update(f"Computing optimum cost allocation... ({cur}/{tot})\n   -- Solving LP...")
            stdout = solver.run_glpsol(data, self._lp_model_path, dat_path=self._dat_path())

            verboser.update(f"Computing optimum cost allocation... ({cur}/{tot})\n    -- Updating internal representation... (  0%)")
            solver.apply_solution(self._network, stdout, progress=lambda frac: verboser.update(
                f"Computing optimum cost allocation... ({cur}/{tot})\n    -- Updating internal representation... ({int(100 * frac):>{3}}%)"))
            end = time.time()
            self._reset_results()
            self._opt_time = end - start
            self._optimum_found = True
            verboser.terminate(f"Optimum allocation successfully completed. ({cur}/{tot}) ({end - start:.2f}s)")

    def calculate_critical_propagation_delay(self) -> int:
        """
        Calculates the minimum simulation duration required for any final production
        to reach the OEM nodes in the supply chain, considering stoichiometric rules 
        and cumulative transport/assembly delays.
        """
        earliest_time = {}
        in_progress = set()
        
        def compute_earliest(node: str) -> float:
            if node in earliest_time:
                return earliest_time[node]
            if node in in_progress:
                return float('inf')
            
            in_progress.add(node)
            ntype = self._baseNetwork.nodes[node].get('type')
            if ntype == 'MATERIAL':
                earliest_time[node] = 0.0
                in_progress.remove(node)
                return 0.0
                
            rule = self._baseNetwork.nodes[node].get('rule', {})
            preds = list(self._baseNetwork.predecessors(node))
            
            if not preds:
                earliest_time[node] = float('inf')
                in_progress.remove(node)
                return float('inf')
                
            if not rule:
                # For non-production nodes (Storages, Warehouses, OEM, etc.), satisfy from ANY predecessor.
                min_arrival = float('inf')
                for p in preds:
                    p_earliest = compute_earliest(p)
                    delay = self._baseNetwork[p][node].get('time', 0.0)
                    min_arrival = min(min_arrival, p_earliest + delay)
                earliest_time[node] = min_arrival
                in_progress.remove(node)
                return min_arrival
            else:
                # For PRODUCTION_INPUT nodes: must satisfy ALL rule keys simultaneously.
                max_key_arrival = 0.0
                for key in rule:
                    matching_preds = [p for p in preds if p.startswith(key)]
                    if not matching_preds:
                        earliest_time[node] = float('inf')
                        in_progress.remove(node)
                        return float('inf')
                        
                    min_matching_arrival = float('inf')
                    for mp in matching_preds:
                        mp_earliest = compute_earliest(mp)
                        delay = self._baseNetwork[mp][node].get('time', 0.0)
                        min_matching_arrival = min(min_matching_arrival, mp_earliest + delay)
                        
                    max_key_arrival = max(max_key_arrival, min_matching_arrival)
                    
                earliest_time[node] = max_key_arrival
                in_progress.remove(node)
                return max_key_arrival

        # Find all OEM nodes
        oem_nodes = [n for n, d in self._baseNetwork.nodes(data=True) if d.get('type') == 'OEM']
        if not oem_nodes:
            return 0
            
        t_max_oem = 0.0
        for oem in oem_nodes:
            t_oem = compute_earliest(oem)
            if t_oem != float('inf'):
                t_max_oem = max(t_max_oem, t_oem)
                
        # Return as an integer (add 1 because a delay of t means product is available at t+1 steps duration)
        return int(t_max_oem) + 1

    def cost(self, include_infinite_supply: bool = False) -> float:
        """
        Computes and returns the total cost of flow in the network.
        If include_infinite_supply is True, returns infinity if auxiliary supply was used.
        """
        if not self._optimum_found: self.find_optimum()
        if self._cost_no_infinite is None:
            self._cost_no_infinite = sum( self._network[u][v]['cost'] * self._network[u][v]['flow'] for u, v in self._network.edges )
            self._cost_infinite = self._cost_no_infinite + (np.inf if any( self._network.nodes[u]['generation'] > 0 for u in self._network.nodes ) else 0)
        return self._cost_infinite if include_infinite_supply else self._cost_no_infinite
    
    def production(self) -> float:
        """Computes and returns the total flow of materials reaching the final layer (OEM nodes)."""
        if not self._optimum_found: self.find_optimum()
        if self._production is None: 
            self._production = sum( self._network[u][v]['flow'] for u, v in self._network.edges 
                                    if self._network.nodes[v]['type'] == 'OEM' and self._network.nodes[u]['type'] != 'OEM')
        return self._production

    def optimisation_time(self) -> float:
        """Returns the computation duration (in seconds) taken by the GLPK LP solver."""
        if not self._optimum_found:
            self.find_optimum()
        return self._opt_time

    def size(self) -> int:
        """Returns the number of nodes in the static base network (prior to time expansion)."""
        return len(self._baseNetwork.nodes) if hasattr(self, '_baseNetwork') else 0

    def get_production(self, verbose: bool, id1: str, id2: str) -> tuple:
        """
        Returns the production parameters between an input node and an output node.
        
        Args:
            verbose (bool): Whether to log the action.
            id1 (str): Production input node label.
            id2 (str): Production output node label.
            
        Returns:
            tuple: (production_time, production_cost, production_capacity)
        """
        with Verboser(verbose, f"Getting production parameters for {id1} -> {id2}...") as verboser:
            for e in self.edges:
                if e['from'] == id1 and e['to'] == id2:
                    return e['time'], e['cost'], e['cap']
            raise ValueError(f"No production edge found between {id1} and {id2}")

    def set_production(self, verbose: bool, id1: str, id2: str, production_time: int, production_cost: float, production_capacity: float) -> None:
        """
        Changes the production parameters for the given production input and output nodes.
        Rebuilds the network and resets optimisation state.
        
        Args:
            verbose (bool): Whether to log the action.
            id1 (str): Production input node label.
            id2 (str): Production output node label.
            production_time (int): New production time.
            production_cost (float): New production cost.
            production_capacity (float): New production capacity.
        """
        with Verboser(verbose, f"Setting production parameters for {id1} -> {id2}...") as verboser:
            found = False
            for e in self.edges:
                if e['from'] == id1 and e['to'] == id2:
                    e['time'] = production_time
                    e['cost'] = production_cost
                    e['cap'] = production_capacity
                    found = True
                    break
            if not found:
                raise ValueError(f"No production edge found between {id1} and {id2}")
            self._rebuild_networks()

    def get_transport(self, verbose: bool, id1: str, id2: str) -> tuple:
        """
        Returns the transport parameters between an output local storage node and an input local storage node.
        
        Args:
            verbose (bool): Whether to log the action.
            id1 (str): Output local storage node label.
            id2 (str): Input local storage node label.
            
        Returns:
            tuple: (transport_time, transport_cost, transport_capacity)
        """
        with Verboser(verbose, f"Getting transport parameters for {id1} -> {id2}...") as verboser:
            for e in self.edges:
                if e['from'] == id1 and e['to'] == id2:
                    return e['time'], e['cost'], e['cap']
            raise ValueError(f"No transport edge found between {id1} and {id2}")

    def set_transport(self, verbose: bool, id1: str, id2: str, transport_time: int, transport_cost: float, transport_capacity: float) -> None:
        """
        Changes the transport parameters for the given output and input local storage nodes.
        Rebuilds the network and resets optimisation state.
        
        Args:
            verbose (bool): Whether to log the action.
            id1 (str): Output local storage node label.
            id2 (str): Input local storage node label.
            transport_time (int): New transport time.
            transport_cost (float): New transport cost.
            transport_capacity (float): New transport capacity.
        """
        with Verboser(verbose, f"Setting transport parameters for {id1} -> {id2}...") as verboser:
            found = False
            for e in self.edges:
                if e['from'] == id1 and e['to'] == id2:
                    e['time'] = transport_time
                    e['cost'] = transport_cost
                    e['cap'] = transport_capacity
                    found = True
                    break
            if not found:
                raise ValueError(f"No transport edge found between {id1} and {id2}")
            self._rebuild_networks()

    def get_local(self, verbose: bool, id_list: list) -> dict:
        """
        Returns the local storage capacities for the provided node labels.
        
        Args:
            verbose (bool): Whether to log the action.
            id_list (list): List of local storage node labels.
            
        Returns:
            dict: Mapping from node label to its storage capacity.
        """
        with Verboser(verbose, f"Getting local storage capacities...") as verboser:
            result = {}
            for node_id in id_list:
                if node_id in self.nodes:
                    result[node_id] = self.nodes[node_id].get('storage_cap', float("inf"))
                else:
                    raise ValueError(f"Node {node_id} not found in the supply chain")
            return result

    def set_local(self, verbose: bool, id_dict: dict) -> None:
        """
        Changes the local storage capacities for the nodes specified in the dictionary.
        Rebuilds the network and resets optimisation state.
        
        Args:
            verbose (bool): Whether to log the action.
            id_dict (dict): Dictionary mapping local storage node label to new capacity.
        """
        with Verboser(verbose, f"Setting local storage capacities...") as verboser:
            for node_id, cap in id_dict.items():
                if node_id in self.nodes:
                    self.nodes[node_id]['storage_cap'] = float(cap)
                else:
                    raise ValueError(f"Node {node_id} not found in the supply chain")
            self._rebuild_networks()


    # helper methods
    def _dataFromJSON(self, filename):
        # simply extracts the data from the JSON file, does not build the internal representation
        with open(filename, 'r') as f:
            return self._parse_data(json.load(f))

    @staticmethod
    def _parse_data(data: dict):
        nodes, edges = data["nodes"], data["edges"]
        for param in nodes.values():
            if 'layer' in param:
                param['layer'] = tuple(param['layer'])
        return nodes, edges 

    def _dat_path(self):
        # where to keep the AMPL data file; None means a temporary file
        if self._save_temp is True:
            return str(config.results_dir("lp") / "model.dat")
        if self._save_temp:
            return f"{self._save_temp}.dat"
        return None

    def _reset_results(self) -> None:
        self._optimum_found = False
        self._opt_time = None
        self._production = None
        self._cost_infinite = None
        self._cost_no_infinite = None
    
    def _rebuild_networks(self) -> None:
        """Helper to safely rebuild the base and dynamic networks after parameters change."""
        self._baseNetworkFromData(self.duration, self.nodes, self.edges, self.demand, self.supply)
        self._critical_delay = self.calculate_critical_propagation_delay()
        self._networkFromData(self.duration, self.nodes, self.edges, self.demand, self.supply)
        self._reset_results()

    def _resolve_config_value(self, config, node_idx, t):
        if isinstance(config, (int, float)):
            return float(config)
        elif isinstance(config, (list, np.ndarray)):
            config = np.array(config)
            if config.ndim == 1:
                t_idx = min(t, len(config) - 1)
                return float(config[t_idx])
            elif config.ndim == 2:
                r_idx = min(node_idx, config.shape[0] - 1)
                c_idx = min(t, config.shape[1] - 1)
                return float(config[r_idx, c_idx])
        return 0.0

    def _baseNetworkFromData(self, duration, nodes, edges, demand, supply):
        # builds the internal representation of the time-less _network from the data extracted from the JSON file
        self._baseNetwork = nx.DiGraph()
        
        self._baseNetwork.add_nodes_from(nodes.items())
        for e in edges:
            u, v, cap, cost, time = e["from"], e["to"], e["cap"], e["cost"], e["time"]
            self._baseNetwork.add_edge(u, v, cap=cap, cost=cost, time=time)
            
    def _networkFromData(self, duration, nodes, edges, demand, supply):
        # builds the internal representation of the _network from the JSON data
        self._network = nx.DiGraph()
        
        # add all nodes         
        for i, t in product(nodes, range(duration)):
            node_type = nodes[i]['type']
            self._network.add_node((i, t), supply=0, demand=0, type=node_type, 
                                   layer=nodes[i]['layer'] if 'layer' in nodes[i] else (0,0))
            if node_type in ("PRODUCTION", "PRODUCTION_INPUT"):
                self._network.nodes[(i,t)]['rule'] = dict(nodes[i]['rule']) 
            
        # add all edges to self
        for i, t in product(nodes, range(duration-1)):
            node_type = nodes[i]['type']
            if node_type not in ("PRODUCTION", "PRODUCTION_INPUT", "PRODUCTION_OUTPUT"):
                storage_cap = float(nodes[i].get('storage_cap', float("inf")))
                self._network.add_edge( (i, t), (i, t+1), cap=storage_cap, cost=TO_SELF_PENALTY )
                    
        # add all 'real' edges
        for e in edges:
            u, v, cap, cost, edge_time = e["from"], e["to"], e["cap"], e["cost"], e["time"]
            for t in range(duration-edge_time):
                self._network.add_edge( (u, t), (v, t+edge_time), cap=cap, cost=cost )
                
        # Handle demand
        oem_nodes = [i for i in nodes if nodes[i]['type'] == 'OEM']
        oem_nodes.sort(key=natural_sort_key)
        for idx, i in enumerate(oem_nodes):
            for t in range(duration):
                self._network.nodes[(i,t)]["demand"] = self._resolve_config_value(demand, idx, t)
                            
        # Handle supply
        material_nodes = [i for i in nodes if nodes[i]['type'] == 'MATERIAL']
        material_nodes.sort(key=natural_sort_key)
        for idx, i in enumerate(material_nodes):
            for t in range(duration):
                self._network.nodes[(i,t)]['supply'] = self._resolve_config_value(supply, idx, t)
    
    def _canProduce(self, i: str, ti: int) -> bool:
        # check if node (i, ti) is a production node and has the required inputs to be able to produce
        if self._network.nodes[(i, ti)]["type"] not in ("PRODUCTION", "PRODUCTION_INPUT"):
            return False
        pred_nodes = {p_node for p_node, p_t in self._network.predecessors((i, ti))}
        return all( p in pred_nodes for p in self._network.nodes[(i,ti)]['rule'] )
       

    def _checkProductionCapabilities(self):
        for i, ti in self._network.nodes:
            node_type = self._network.nodes[(i,ti)]['type']
            if node_type not in ("PRODUCTION", "PRODUCTION_INPUT"):
                continue
            
            # Warehouses are not production nodes and should not have production capabilities
            if node_type == 'WAREHOUSE':
                self._network.nodes[(i,ti)]['prod_pred'] = {}
                self._network.nodes[(i,ti)]['supp_pred'] = []
                continue

            prod_pred = defaultdict(list)
            supp_pred = []
            for (j, tj) in self._network.predecessors((i,ti)):
                if j in self._network.nodes[(i, ti)]['rule']:
                    prod_pred[j].append(tj)
                else:
                    supp_pred.append((j, tj))
            self._network.nodes[(i,ti)]['supp_pred'] = supp_pred
            if len(prod_pred) == len(self._network.nodes[(i,ti)]['rule']):
                self._network.nodes[(i,ti)]['prod_pred'] = prod_pred
            else: 
                self._network.nodes[(i,ti)]['prod_pred'] = {}

    def _node_time_neighbourhood(self, label: str, t: int, dt: int = 4):
        """
        Returns the dynamic nodes within two hops of the backbone node `label` (its PI or W node) in the
        time window [t - dt, t + dt], together with the outputs of its (O) storages, and the window bounds.
        """
        # Determine if the label corresponds to a warehouse
        is_warehouse = any(n.startswith(label) and d.get('type') == 'WAREHOUSE' for n, d in self._baseNetwork.nodes(data=True))

        pi_node = f"{label}(PI)"
        w_node = f"{label}(W)"
        
        # Expand time window to show as much structure in time as fits reasonably
        t_min = max(0, t - dt)
        t_max = min(self.duration - 1, t + dt)
        
        subgraph_nodes = set()
        undirected_G = self._network.to_undirected()
        
        for time_step in range(t_min, t_max + 1):
            if (pi_node, time_step) in self._network.nodes:
                central = (pi_node, time_step)
            elif (w_node, time_step) in self._network.nodes:
                central = (w_node, time_step)
            else:
                continue
                
            subgraph_nodes.update(nx.ego_graph(undirected_G, central, radius=2).nodes())
            
        if not subgraph_nodes:
            # Fallback for single node check if window search fails
            if (w_node, t) in self._network.nodes and is_warehouse:
                central = (w_node, t)
            elif (pi_node, t) in self._network.nodes:
                central = (pi_node, t)
            else:
                # If PI is not found, it might be a pure warehouse, check for W again
                if (w_node, t) in self._network.nodes:
                    central = (w_node, t)
                else:
                    raise KeyError(f"Central node for label {label} around time {t} not found.")
            subgraph_nodes = set(nx.ego_graph(undirected_G, central, radius=2).nodes())

        # Add successors of all (O) nodes that belong to the label to see outputs
        for n, n_t in list(subgraph_nodes):
            if n.startswith(label) and '(O' in n:
                for succ in self._network.successors((n, n_t)):
                    subgraph_nodes.add(succ)

        return subgraph_nodes, t_min, t_max

    @staticmethod
    def _universal_supply_chain_generator(size, tau=2.5, duration=20, beta=4, prune=True, integrity='partial', vratio=.8, verbose=False, supply=1e9, demand=10):
        cl = SupplyChainBackbone(size=size, tau=tau, beta=beta, prune=prune, integrity=integrity, vratio=vratio, verbose=verbose)
        clc = ScbConverter(cl, verbose=verbose)
        return SupplyChain.from_converter(clc, duration=duration, supply=supply, demand=demand)

    @staticmethod
    def tiny_pyramid(tau=2.5, duration=20, verbose=False, supply=1e9, demand=10):
        return SupplyChain._universal_supply_chain_generator(size=[10, 20, 40], tau=tau, duration=duration, beta=4, prune=True, integrity='partial', vratio=.8, verbose=verbose, supply=supply, demand=demand)

    @staticmethod
    def small_pyramid(tau=2.5, duration=20, verbose=False, supply=1e9, demand=10):
        return SupplyChain._universal_supply_chain_generator(size=[10, 50, 250], tau=tau, duration=duration, beta=4, prune=True, integrity='partial', vratio=.8, verbose=verbose, supply=supply, demand=demand)

    @staticmethod
    def small_medium_pyramid(tau=2.5, duration=20, verbose=False, supply=1e9, demand=10):
        return SupplyChain._universal_supply_chain_generator(size=[20, 100, 500], tau=tau, duration=duration, beta=5, prune=True, integrity='partial', vratio=.8, verbose=verbose, supply=supply, demand=demand)

    @staticmethod
    def medium_pyramid(tau=2.5, duration=20, verbose=False, supply=1e9, demand=10):
        return SupplyChain._universal_supply_chain_generator(size=[20, 100, 500, 2500], tau=tau, duration=duration, beta=5, prune=True, integrity='partial', vratio=.8, verbose=verbose, supply=supply, demand=demand)
            
    @staticmethod            
    def tiny_double_pyramid(tau=2.5, duration=20, verbose=False, supply=1e9, demand=10): 
        return SupplyChain._universal_supply_chain_generator(size=[10, 20, 40, 20, 10], tau=tau, duration=duration, beta=4, prune=True, integrity='partial', vratio=.8, verbose=verbose, supply=supply, demand=demand)
            
    @staticmethod
    def small_double_pyramid(tau=2.5, duration=20, verbose=False, supply=1e9, demand=10): 
        return SupplyChain._universal_supply_chain_generator(size=[10, 50, 250, 50, 10], tau=tau, duration=duration, beta=6, prune=True, integrity='partial', vratio=.8, verbose=verbose, supply=supply, demand=demand)

    @staticmethod
    def small_medium_double_pyramid(tau=2.5, duration=20, verbose=False, supply=1e9, demand=10):
        return SupplyChain._universal_supply_chain_generator(size=[20, 100, 500, 100, 20], tau=tau, duration=duration, beta=6, prune=True, integrity='partial', vratio=.8, verbose=verbose, supply=supply, demand=demand)

    @staticmethod
    def medium_double_pyramid(tau=2.5, duration=20, verbose=False, supply=1e9, demand=10):
        return SupplyChain._universal_supply_chain_generator(size=[20, 100, 500, 2500, 500, 100, 20], tau=tau, duration=duration, beta=6, prune=True, integrity='partial', vratio=.8, verbose=verbose, supply=supply, demand=demand)
