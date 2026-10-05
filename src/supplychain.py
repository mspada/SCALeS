from itertools import product, chain, combinations
from collections import defaultdict, deque
import json
import networkx as nx
import matplotlib.pyplot as plt 
from matplotlib.colors import CSS4_COLORS
from string import ascii_uppercase
from enum import Enum

import time
import random
import numpy as np
import subprocess
import os
from tempfile import NamedTemporaryFile

from scb import SupplyChainBackbone
from chunglu_converter import ChungLuConverter
from local import glpsol, neato


from verboser import Verboser

ampl = \
"""
param DURATION := {};

set BASIC_NODES := {} ;

param SUPPLY:=
{}  
;

param DEMAND:=
{}
;
    
param:
    ARCS: COST CAP:= 
{}
;

{}

{}

param NUMBER_RULES_FOR := 
{}
;

param IN_RULE_COEFFICIENTS := 
{}
;

param OUT_RULE_COEFFICIENTS := 
{}
;

end; 
"""

# TODO - Annotate the code
# TODO - It is very likely possible to simplify the model and the generation (while keeping the same level of flexibility).
#        Have a look into it.

"""
This class represents supply chains.  
During initialisation, may take several arguments described below.
    - JSONfilename (str): the filename from which the chain should be initialised
        Ignored when providing a non-None parameter supply_chain
    - supply_chain (SupplyChain): the supply chain from which the initialised supply chain should be copied.
        Default is None.
    - verbose (bool): details the execution of each step when set to True.
    
Once initialised, provides several methods. The most useful are:
    - toAMPL(): generates a string containing the data for the AMPL model in 'models/stoichiometric-lp-model.mod'.
    - find_optimum(save_temp, verbose, state): computes the optimal cost allocation.
        When save_temp is set to True, the .dat model is not destroyed after initialisation is complete.
        When verbose is set to True, details the optimisation process. 
        Parameter state should not be used. It is used by another class while in verbose mode.
    - generate_dot(show, save, filename, filetype, expanding_factor): generates a representation of the graph in the desired file. 
        If show is set to True, opens the file after generation.
        If save is set to True, saves the file.
        Parameter expanding factor may help spread nodes apart in large graphs when they appear all on top of each other.
    
The class also provides some static methods for easy generation of SCB (Supply Chain Backbone) graph with preset parameters. Available are:
    - tiny_pyramid: generates a supply chain from a SCB (Supply Chain Backbone) graph with 3 layers of sizes 10, 20, 40 respectively. Duration of 20 by default may be changed.
    - small_pyramid: generates a supply chain from a SCB (Supply Chain Backbone) graph with 3 layers of sizes 10, 50, 250 respectively. Duration of 20 by default may be changed.
    - small_medium_pyramid: generates a supply chain from a SCB (Supply Chain Backbone) graph with 3 layers of sizes 20, 100, 500 respectively. Duration of 20 by default may be changed.
    - medium_pyramid: generates a supply chain from a SCB (Supply Chain Backbone) graph with 4 layers of sizes 20, 100, 500, 2500 respectively. Duration of 20 by default may be changed.
    - tiny_double_pyramid: generates a supply chain from a SCB (Supply Chain Backbone) graph with 5 layers of sizes 10, 20, 40, 20, 10 respectively. Duration of 20 by default may be changed.
    - small_double_pyramid: generates a supply chain from a SCB (Supply Chain Backbone) graph with 5 layers of sizes 10, 50, 250, 50, 10 respectively. Duration of 20 by default may be changed.
    - small_medium_double_pyramid: generates a supply chain from a SCB (Supply Chain Backbone) graph with 5 layers of sizes 20, 100, 500, 100, 20 respectively. Duration of 20 by default may be changed.
    - medium_double_pyramid: generates a supply chain from a SCB (Supply Chain Backbone) graph with 7 layers of sizes 20, 100, 500, 2500, 500, 100, 20 respectively. Duration of 20 by default may be changed.
"""

TO_SELF_PENALTY = .001

class DataType(Enum):
    DURATION = 1

class NodeType(Enum):
    LOCAL_STORAGE = 1
    PRODUCTION = 2
    EOM = 3

class SupplyChain:
    
    # class functions
    def __init__(self, JSONfilename=None, supply_chain=None, verbose=False) -> None:
        # initialises the internal _network representation from a JSON file or copies another existing supply chain
        # prefers copying over reading from .json
        # fails the initialisation if both arguments are invalid

        with Verboser(verbose, "Creating Supply Chain...") as verboser:
            start = time.time()
            if supply_chain: 
                verboser.update("Creating Supply Chain...\n   -- Initialising from other Supply Chain...")
                duration, nodes, edges, demand, supply = supply_chain.duration, supply_chain.nodes, supply_chain.edges, supply_chain.demand, supply_chain.supply        
            else: 
                verboser.update("Creating Supply Chain...\n   -- Loading from json file...")
                duration, nodes, edges, demand, supply = self._dataFromJSON(JSONfilename)
            # duration, nodes, edges, demand, supply = self._dataFromJSON(JSONfilename)
            self.duration, self.nodes, self.edges, self.demand, self.supply = duration, nodes, edges, demand, supply
            verboser.update("Creating Supply Chain...\n   -- Building base network...")
            self._baseNetworkFromData(duration, nodes, edges, demand, supply)
            verboser.update("Creating Supply Chain...\n   -- Building network...")
            self._networkFromData(duration, nodes, edges, demand, supply) 
            verboser.update("Creating Supply Chain...\n   -- Generating positions for drawing...")
            self._find_positions()
            self._optimum_found = False
            self._opt_time = None
            self._production = None
            self._cost_infinite = None
            self._cost_no_infinite = None
            self._size = None
            end = time.time()
            verboser.terminate(f"Supply Chain successfully generated. ({end - start:.2f}s)")

    # main methods
    def toAMPL(self) -> str:
        return ampl.format( \
                self._toAMPLDuration(),
                self._toAMPLBasicNodes(),
                self._toAMPLSupply(),
                self._toAMPLDemand(),
                self._toAMPLArcsCostCap(),
                self._toAMPLInNeighbours(),
                self._toAMPLOutNeighbours(),
                self._toAMPLNumberRulesFor(),
                self._toAMPLInRuleCoefficients(),
                self._toAMPLOutRuleCoefficients()      
        )   
    
    def find_optimum(self, save_temp=False, verbose=False, state=(1,1)) -> None:
        # uses glpsol to find the optimal solution and updates the network accordingly
        cur, tot = state

        with Verboser(verbose, f"Computing optimum cost allocation... ({cur}/{tot})") as verboser:
            start = time.time()
            solString = self._solveToString(verboser=verboser, save_temp=save_temp, cur=cur, tot=tot)
            lines = solString.split('\n')
                
            verboser.update(f"Computing optimum cost allocation... ({cur}/{tot})\n    -- Updating internal representation... (  0%)")
            for i, line in enumerate(lines):
                try:
                    u, tu, v, tv, flow = line.split()
                    # print(f"u={u} is a {type(u)}")
                    self._network[(u, int(tu))][(v, int(tv))]['flow'] = float(flow)
                    continue
                except ValueError:
                    pass
                except Exception:
                    pass   

                try:
                    # print('Now trying gen-abs.')
                    # print(line.split())
                    u, tu, genu, absu = line.split()
                    # print(u, tu, genu, absu)
                    self._network.nodes[(u, int(tu))]['generation'] = float(genu)
                    self._network.nodes[(u, int(tu))]['absorption'] = float(absu)
                    # print('Success.')
                except ValueError:
                    pass
                except Exception:
                    pass   
                verboser.update(f"Computing optimum cost allocation... ({cur}/{tot})\n    -- Updating internal representation... ({int(100 * (i+1)/len(lines)):>{3}}%)")
            end = time.time()
            self._opt_time = end - start
            self._optimum_found = True
            verboser.terminate(f"Optimum allocation successfully completed. ({cur}/{tot}) ({end - start:.2f}s)")

    def generate_dot(self, show=True, save=True, filename='network', filetype='pdf', expanding_factor=6):
        # turns the internal representation of the graph into a .dot a .pdf file (by default, this may be changed)
        # if keepDOT is False, the .dot file is deleted once the .pdf file is created
        # if show is True, a window with the .odf file is opened once created
        temp_network = self._network.copy()
        for node in temp_network.nodes:
            if 'rule' in temp_network.nodes[node]:
                del temp_network.nodes[node]['rule']
            if 'supp_pred' in temp_network.nodes[node]:
                del temp_network.nodes[node]['supp_pred']
            if 'prod_pred' in temp_network.nodes[node]:
                del temp_network.nodes[node]['prod_pred']        

        for (node, t) in self._network.nodes:
            x, y = self._network.nodes[(node, t)]['pos']
            temp_network.nodes[(node, t)]['pos'] = f"{expanding_factor*x:.3f}, {expanding_factor*y:.3f}!"
            temp_network.nodes[(node, t)]['label'] = node + str(t)

        for x in temp_network.nodes:
            if 'generation' in temp_network.nodes[x] and temp_network.nodes[x]['generation'] > 0:
                temp_network.nodes[x]['color']='red'
                temp_network.nodes[x]['fontcolor'] = 'red'
                temp_network.nodes[x]['fontname'] = 'Arial Bold'
                temp_network.nodes[x]['penwidth'] = 3

        for x, y in temp_network.edges:
            if 'flow' in temp_network[x][y] and temp_network[x][y]['flow']:
                temp_network[x][y]['label']=f"{temp_network[x][y]['flow']:.2f}"
                temp_network[x][y]['penwidth']=3
                temp_network[x][y]['color']='blue'
                temp_network[x][y]['fontcolor'] = 'blue'
                temp_network[x][y]['fontname'] = 'Arial Bold'

    
        nx.drawing.nx_pydot.write_dot(temp_network, f"results/{filename}.dot")
    
        command = [neato, "-T", filetype, f"results/{filename}.dot", "-o", f"results/{filename}." + filetype]
        subprocess.run(command, check=True, text=True, capture_output=True)
        
        if not save:
            os.remove(f"results/{filename}.dot")
        
        if show:
            command = ["open", f"results/{filename}." + filetype]
            subprocess.run(command, check=True, text=True, capture_output=True)

    def draw_partite_base_network(self, show=True, save=False, filename='base_network.png'):
        layers = set( tuple(self._baseNetwork.nodes[node]['layer']) for node in self._baseNetwork.nodes ) 
        col_layers = { layer: random.choice(list(CSS4_COLORS.keys())) for layer in layers }
        col = [ col_layers[ tuple(self._baseNetwork.nodes[node]['layer']) ] for node in self._baseNetwork.nodes ]
        pos = { node: param['pos'] for node, param in self._baseNetwork.nodes.items() }
        plt.figure(figsize=(14,8))
        nx.draw(G=self._baseNetwork, pos=pos, node_color=col, node_size=1500, with_labels=True)
        if save: plt.savefig(filename)
        if show: plt.show()

    def draw_partite_network(self, show=True, save=False, filename='network.png'):
        layers = set( tuple(self._baseNetwork.nodes[node]['layer']) for node in self._baseNetwork.nodes ) 
        col_layers = { layer: random.choice(list(CSS4_COLORS.keys())) for layer in layers }
        col = [ col_layers[ tuple(self._baseNetwork.nodes[node]['layer']) ] for node, t in self._network.nodes ]
        labels = { (node, t): node + str(t) for node, t in self._network.nodes }
        pos = { node: param['pos'] for node, param in self._network.nodes.items() }
        plt.figure(figsize=(14,8))
        nx.draw(G=self._network, pos=pos, node_color=col, node_size=2000, labels=labels, with_labels=True)
        if save: plt.savefig(filename)
        if show: plt.show()

    def cost(self, include_infinite_supply=False):
        if not self._optimum_found: self.find_optimum()
        if self._cost_infinite is None or self._cost_no_infinite:
            self._cost_no_infinite = sum( self._network[u][v]['cost'] * self._network[u][v]['flow'] for u, v in self._network.edges )
            self._cost_infinite = self._cost_no_infinite + (np.inf if any( self._network.nodes[u]['generation'] > 0 for u in self._network.nodes ) else 0)
        return self._cost_infinite if include_infinite_supply else self._cost_no_infinite
    
    def production(self):
        if not self._optimum_found: self.find_optimum()
        if self._production is None: 
            self._production = sum( self._network[u][v]['flow'] for u, v in self._network.edges 
                                    if self._network.nodes[v]['type'] == 'OEM' and self._network.nodes[u]['type'] != 'OEM')
        return self._production

    def optimisation_time(self):
        if not self._optimum_found:
            self.find_optimum()
        return self._opt_time

    def size(self):
        # TODO -- Actually return the size.
        return 0

    # helper methods
    def _dataFromJSON(self, filename):
        # simply extracts the data from the JSON file, does not build the internal representation
        f = open(filename, 'r')
        data = json.load(f)
        f.close
        duration, nodes, edges, demand, supply = data["duration"], data["nodes"], data["edges"], data["demand"], data["supply"]
        demand.sort(key = lambda x: x["time"])
        supply.sort(key = lambda x: x["time"])
        for param in nodes.values():
            if 'layer' in param:
                param['layer'] = tuple(param['layer'])
        return duration, nodes, edges, demand, supply 
    
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
            self._network.add_node((i, t), supply=0, demand=0, type=nodes[i]['type'], 
                                   layer=nodes[i]['layer'] if 'layer' in nodes[i] else (0,0))
            if nodes[i]['type'] == "PRODUCTION":
                self._network.nodes[(i,t)]['rule'] = dict(nodes[i]['rule']) 
            
        # add all edges to self
        for i, t in product(nodes, range(duration-1)):
            if nodes[i]['type'] != "PRODUCTION":
                self._network.add_edge( (i, t), (i, t+1), cap=float("inf"), cost=TO_SELF_PENALTY )            
                    
        # add all 'real' edges
        for e in edges:
            u, v, cap, cost, time = e["from"], e["to"], e["cap"], e["cost"], e["time"]
            for t in range(duration-time):
                self._network.add_edge( (u, t), (v, t+time), cap=cap, cost=cost )
                
        # add all demands
        for dem in demand:
            i, ti, d = dem["node"], dem["time"], dem["demand"]
            # this is highly sub-optimal
            for t in range(ti, self.duration):
                self._network.nodes[(i,t)]["demand"] = d
                            
        # add all supplies
        for sup in supply:
            i, ti, s = sup["node"], sup["time"], sup["supply"]
            for t in range(ti, self.duration):
                self._network.nodes[(i,t)]['supply'] = s
    
    def _toAMPLDuration(self) -> str:
        return self.duration
    
    def _toAMPLBasicNodes(self) -> str:
        return "'" + "' '".join(self.nodes) + "'"
    
    def _toAMPLSupply(self) -> str:
        amplSupply = ""
        for i, ti in self._network.nodes:
            supply = self._network.nodes[(i,ti)]['supply']
            amplSupply += f"'{i}', {ti} {supply} \n"
        
        return amplSupply
    
    def _toAMPLDemand(self) -> str:
        amplDemand = ""
        
        for i, ti in self._network.nodes:
            demand = self._network.nodes[(i,ti)]['demand']
            amplDemand += f"'{i}', {ti} {demand} \n"
        
        return amplDemand
    
    def _toAMPLArcsCostCap(self) -> str:
        amplArcsCostCap = ""
        
        for (i, ti), (j, tj) in self._network.edges:
            cost, cap = self._network[(i, ti)][(j, tj)]['cost'], self._network[(i, ti)][(j, tj)]['cap']
            cost, cap = min(100, cost), min(100, cap)
            amplArcsCostCap += f"    '{i}', {ti}, '{j}', {tj} {cost} {cap} \n"
        
        return amplArcsCostCap
    
    def _toAMPLInNeighbours(self) -> str:
        amplInNeighbours = ""
        
        for i, ti in self._network.nodes:
            amplInNeighbours += f"set IN_NEIGHBOURS['{i}', {ti}] := "
            for j, tj in self._network.predecessors((i, ti)):
                amplInNeighbours += f"('{j}', {tj}) "
            
            amplInNeighbours += "; \n"
        
        return amplInNeighbours
        
    def _toAMPLOutNeighbours(self) -> str:
        amplOutNeighbours = ""
        
        for i, ti in self._network.nodes:
            amplOutNeighbours += f"set OUT_NEIGHBOURS['{i}', {ti}] := "
            for j, tj in self._network.successors((i, ti)):
                amplOutNeighbours += f"('{j}', {tj}) "
            
            amplOutNeighbours += "; \n"
        
        return amplOutNeighbours    
    
    def _toAMPLNumberRulesFor(self) -> str:
        amplRules = ""
        for i, ti in self._network.nodes:
            nRules = len(self._network.nodes[(i, ti)]['rule']) if self._canProduce(i, ti) else 1  
            amplRules += f"'{i}', {ti} {nRules} \n"
        return amplRules
    
    def _toAMPLInRuleCoefficients(self) -> str:
        amplInRules = ""
        
        self._checkProductionCapabilities()
        
        for i, ti in self._network.nodes:
            if self._network.nodes[(i,ti)]['type'] == 'PRODUCTION' and self._network.nodes[(i,ti)]['prod_pred']:
                for index, j in enumerate(self._network.nodes[(i,ti)]['prod_pred']):
                    for k, tk in self._network.nodes[(i, ti)]['supp_pred']:
                        amplInRules += f"'{i}' {ti}, '{k}', {tk}, {index} 1 \n "
                    for tj in self._network.nodes[(i,ti)]['prod_pred'][j]:
                        amplInRules += f"'{i}', {ti}, '{j}', {tj}, {index} {1/self._network.nodes[(i, ti)]['rule'][j]} \n"
            elif self._network.nodes[(i,ti)]['type'] == 'PRODUCTION':
                for j, tj in self._network.nodes[(i, ti)]['supp_pred']:
                    amplInRules += f"'{i}' {ti}, '{j}', {tj}, 0 1 \n "
                for j, tj in set(self._network.predecessors((i,ti))) - set(self._network.nodes[(i,ti)]['supp_pred']):
                    amplInRules += f"'{i}' {ti}, '{j}', {tj}, 0 0 \n "
            else:
                for j, tj in self._network.predecessors((i,ti)):
                    amplInRules += f"'{i}', {ti}, '{j}', {tj}, 0 1 \n"
        
        return amplInRules
    
    def _toAMPLOutRuleCoefficients(self) -> str:
        amplOutRules = ""
        
        for i, ti in self._network.nodes:
            nRules = len(self._network.nodes[(i, ti)]['rule']) if self._canProduce(i, ti) else 1  
            for (j, tj), index in product(self._network.successors((i,ti)), range(nRules)):
                amplOutRules += f"'{i}', {ti}, '{j}', {tj}, {index} 1 \n"
        
        return amplOutRules     
    
    def _canProduce(self, i, ti) -> bool:
        # check if node (i, ti) is a production node and has the required inputs to be able to produce
        # --TODO-- This is currently very poorly implemented and should be improved
        if self._network.nodes[(i, ti)]["type"] != "PRODUCTION":
            return False
        pred, _ = zip(*self._network.predecessors((i, ti))) if self._network.in_degree((i,ti)) else [], None
        pred = set(pred)
        return all( p in pred for p in self._network.nodes[(i,ti)]['rule'] )
       
    def _solveToString(self, verboser, save_temp=False, cur=1, tot=1) -> str: 
        # uses glpsol to find the optimal commodity flow and return the answer as provided by AMPL

        verboser.update(f"Computing optimum cost allocation... ({cur}/{tot})\n   -- Generating LP data...")
        AMPL_string = self.toAMPL()

        verboser.update(f"Computing optimum cost allocation... ({cur}/{tot})\n   -- Writing data to temporary file...")
        with NamedTemporaryFile(mode='w', dir="temp", delete=not save_temp) as temp_file:
            temp_file.write(AMPL_string)
            verboser.update(f"Computing optimum cost allocation... ({cur}/{tot})\n   -- Solving LP...")
            command = [glpsol, "-m", "models/stoichiometric-lp-model.mod", "-d", temp_file.name]
            # command = [glpsol, "-m", "models/stoichiometric-lp-model.mod"]
            com_exec = subprocess.run(command, check=True, text=True, capture_output=True)
            ans = com_exec.stdout
        

        return ans
        
    def _checkProductionCapabilities(self):
        for i, ti in self._network.nodes:
            if self._network.nodes[(i,ti)]['type'] != "PRODUCTION":
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

    def _find_positions(self):
        pos_base = nx.multipartite_layout(G=self._baseNetwork, subset_key="layer")
        pos_base = { node: (x +  self._baseNetwork.nodes[node]['layer'][0] /4,y) for node, (x,y) in pos_base.items() }
        _, y_pos_base = zip(*pos_base.values())
        ymin, ymax = min(y_pos_base), max(y_pos_base)
        for node in self._baseNetwork.nodes:
            x, y = pos_base[node]
            self._baseNetwork.nodes[node]['pos'] = x, y
            for t in range(self.duration):
                self._network.nodes[(node, t)]['pos'] = x+t/2, y - t * (ymax - ymin + 1/2)

    def _drawBaseNetwork(self, layout=None):
        # a method that quickly draws the internal representation of the base network (without time)
        # should mainly be used for debugging
        if layout:
            pos = layout(self._baseNetwork)
        else:
            pos = self._customDownTopLayoutBaseGraph()
        nx.draw(self._baseNetwork, pos=pos, with_labels=True) 
        plt.show()
    
    def _drawNetwork(self, baseLayout=None):
        # a method that quickly draws the internal representation of the network
        # should mainly be used for debugging
        # basePos = baseLayout(self._baseNetwork)
        if baseLayout:
            basePos = baseLayout(self._baseNetwork)
        else:
            basePos = self._customDownTopLayoutBaseGraph()        
        
        xcoord, _ = zip(*basePos.values())
        xmin, xmax = min(xcoord), max(xcoord)
        deltax = (xmax - xmin)
        # deltay = (ymax - ymin)/2
        
        pos = {}
        
        for n, t in product(self.nodes, range(self.duration)):
            x, y = basePos[n]
            pos[(n,t)] = ( x + (2*t - self.duration + 1) * deltax , y )
        
        nx.draw(self._network, pos=pos, with_labels=True)
        plt.show()
    
    def _customDownTopLayoutBaseGraph(self, dx=1, dy=1/3):
        # should only be used if it is absolutely certain that the graph is acyclic
        positions = {}
                
        lay = deque([])
        for node, deg in self._baseNetwork.out_degree:
            if deg == 0:
                lay.append(node)
    
        layers = []
        layer_of = {}
        cur = 0
    
        while lay:
            layers.append(set())
            width = len(lay)
            for _ in range(width):
                node = lay.popleft()
                if node in layer_of:
                    layers[layer_of[node]].remove(node)
                layers[cur].add(node)
                layer_of[node] = cur
                for prev in self._baseNetwork.predecessors(node):
                    lay.append(prev)
            cur += 1
                    
        x, y = 0, 0
        for layer in layers:
            x = 0
            for node in layer:
                positions[node] = np.array([x,y])
                x,y = x-dx, y+dy
            y = y+3*dy
            
        return positions

    @staticmethod
    def _universal_supply_chain_generator(size, tau=2.5, duration=20, beta=4, prune=True, integrity='partial', vratio=.8, verbose=False):
        t_start = time.time()
        cl = SupplyChainBackbone(size=size, tau=tau, beta=beta, prune=prune, integrity=integrity, vratio=vratio, verbose=verbose)
        t_cl = time.time()
        
        clc = ChungLuConverter(cl, verbose=verbose)
        t_clc = time.time()

        with NamedTemporaryFile(mode='w+', delete=True) as temp_file:
            clc.to_json(temp_file.name, duration=duration, supply=1e9, demand=1e9, verbose=verbose)
            t_json = time.time()

            sc = SupplyChain(JSONfilename=temp_file.name)
        t_end = time.time()

        # if verbose: print(f"Total runtime for the generation of the Supply Chain: {t_end - t_start:.2f}s.") 
        return sc

    @staticmethod
    def tiny_pyramid(tau=2.5, duration=20, verbose=False):
        return SupplyChain._universal_supply_chain_generator(size=[10, 20, 40], tau=tau, duration=duration, beta=4, prune=True, integrity='partial', vratio=.8, verbose=verbose)

    @staticmethod
    def small_pyramid(tau=2.5, duration=20, verbose=False):
        return SupplyChain._universal_supply_chain_generator(size=[10, 50, 250], tau=tau, duration=duration, beta=4, prune=True, integrity='partial', vratio=.8, verbose=verbose)

    @staticmethod
    def small_medium_pyramid(tau=2.5, duration=20, verbose=False):
        return SupplyChain._universal_supply_chain_generator(size=[20, 100, 500], tau=tau, duration=duration, beta=5, prune=True, integrity='partial', vratio=.8, verbose=verbose)

    @staticmethod
    def medium_pyramid(tau=2.5, duration=20, verbose=False):
        return SupplyChain._universal_supply_chain_generator(size=[20, 100, 500, 2500], tau=tau, duration=duration, beta=5, prune=True, integrity='partial', vratio=.8, verbose=verbose)
            
    @staticmethod            
    def tiny_double_pyramid(tau=2.5, duration=20, verbose=False): 
        return SupplyChain._universal_supply_chain_generator(size=[10, 20, 40, 20, 10], tau=tau, duration=duration, beta=4, prune=True, integrity='partial', vratio=.8, verbose=verbose)
            
    @staticmethod
    def small_double_pyramid(tau=2.5, duration=20, verbose=False): 
        return SupplyChain._universal_supply_chain_generator(size=[10, 50, 250, 50, 10], tau=tau, duration=duration, beta=6, prune=True, integrity='partial', vratio=.8, verbose=verbose)

    @staticmethod
    def small_medium_double_pyramid(tau=2.5, duration=20, verbose=False):
        return SupplyChain._universal_supply_chain_generator(size=[20, 100, 500, 100, 20], tau=tau, duration=duration, beta=6, prune=True, integrity='partial', vratio=.8, verbose=verbose)

    @staticmethod
    def medium_double_pyramid(tau=2.5, duration=20, verbose=False):
        return SupplyChain._universal_supply_chain_generator(size=[20, 100, 500, 2500, 500, 100, 20], tau=tau, duration=duration, beta=6, prune=True, integrity='partial', vratio=.8, verbose=verbose)


if __name__ == "__main__":
    
    filename = 'test.json'
    
    cl = SupplyChainBackbone(size=[3,6,3])
    clc = ChungLuConverter(cl)
    clc.to_json(filename=filename, duration=10, supply=1e9, demand=1e9)
    sc = SupplyChain(JSONfilename=filename)
    sc.draw_partite_base_network()
    # sc.draw_partite_network()
    # sc.find_optimum()
    # sc.generate_dot(show=True, save=False)
    
    
    # filename = 'test_conversion'

    # cl = SupplyChainBackbone.tiny_example(prune=True, integrity='full')
    # conv = ChungLuConverter(cl)
    # # conv.draw_supplychain(save=True, filename=filename+'.png')
    # conv.to_json(filename=filename+'.json', duration=10, supply=1e9, demand=1)
    # sc = SupplyChain(JSONfilename=filename+'.json')
    # sc.find_optimum(save_temp=True)
    # for u,v in sc._network.edges:
    #     print(f"{(u,v)}: {sc._network[u][v]}")

    # sc.generate_dot(save=False, show=True)
    # print(f"Total cost with infinite source: {sc.cost(include_infinite_supply=True)}")
    # print(f"Total cost without infinite source: {sc.cost(include_infinite_supply=False)}")
    # print(f"Total amount of production: {sc.production():.3f}")
    
    # sc = SupplyChain.medium_pyramid(duration=20, verbose=True)      
    # sc = SupplyChain.tiny_double_pyramid(verbose=True)
    # sc.draw_partite_base_network()
    # sc.generate_dot(show=True, save=False)  
            
            
        
