import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.linear_model import LinearRegression

import time
import json

from scb import SupplyChainBackbone
from chunglu_converter import ChungLuConverter
from supplychain import SupplyChain

from verboser import Verboser

"""
This class allows to perturbe a supply chain. 
No instance can be created, all methods are static. 

One method (perturbe) can be called. It takes a supply chain and return a new supply chain, obtained by perturbing it.
Its arguments are:
    - sc (SupplyChain): the SupplyChain to be perturbed;
    - type (str, 'normal' or 'binomial'): the type of perturbation to be performed.
    - p (float, >= 0, <=1): when perturbing under 'binomial' mode, the probability of deleting each edge
    - mu (float): when perturbing under 'normal' mode, the expected value of the random normal noise added to the cost and capacity of each edge.
        Can be replaced by u_mu and c_mu for finer control of the expected value of the capacity and cost respectively.
    - sigma (float): when perturbing under 'normal' mode, the standard deviation of the random normal noise added to the cost and capacity of each edge.
        Can be replaced by u_sigma and c_sigma for finer control of the expected value of the capacity and cost respectively.
    - verbose (bool): details the execution of each step when set to True. 
"""

class SupplyChainPerturber:

    @staticmethod
    def perturbe(sc: SupplyChain, **parameters):
        verbose = parameters.get('verbose', False)
        cur, tot = parameters.get('state', (1,1))

        p_type = parameters['type'] if 'type' in parameters and parameters['type'] in ['normal', 'binomial']\
                else 'normal'
        if p_type == 'binomial': 
            p = parameters['p'] if 'p' in parameters and type(parameters['p']) in {int, float} and 0 <= parameters['p'] <= 1 else .1
            return SupplyChainPerturber._perturbe_binomial(sc=sc, p=p, verbose=verbose)
        # if none of the above applies, do normal perturbation
        c_mu = parameters['c_mu'] if 'c_mu' in parameters and type(parameters['c_mu']) in {int, float} \
                else parameters['mu'] if 'mu' in parameters and type(parameters['mu']) in {int, float} \
                else 0
        u_mu = parameters['u_mu'] if 'u_mu' in parameters and type(parameters['u_mu']) in {int, float} \
                else parameters['mu'] if 'mu' in parameters and type(parameters['mu']) in {int, float} \
                else 0
        c_sigma = parameters['c_sigma'] if 'c_sigma' in parameters and type(parameters['c_sigma']) in {int, float} \
                else parameters['sigma'] if 'sigma' in parameters and type(parameters['sigma']) in {int, float} \
                else .1
        u_sigma = parameters['c_sigma'] if 'c_sigma' in parameters and type(parameters['c_sigma']) in {int, float} \
                else parameters['sigma'] if 'sigma' in parameters and type(parameters['sigma']) in {int, float} \
                else .1
        return SupplyChainPerturber._perturbe_normal(sc=sc, c_mu=c_mu, u_mu=u_mu, c_sigma=c_sigma, u_sigma=u_sigma, verbose=verbose, cur=cur, rep=tot)



    @staticmethod
    def _perturbe_binomial(sc: SupplyChain, p=.1, verbose=False, cur=1, rep=1):
        with Verboser(verbose, f"Perturbing Supply Chain... ({cur}/{rep})") as verboser:
            start = time.time()
            perturbation_details = { 'Delta': 0 }
            verboser.update(f"Perturbing Supply Chain... ({cur}/{rep})\n   -- Copying original Chain...")
            perturbed = SupplyChain(supply_chain=sc)
            verboser.update(f"Perturbing Supply Chain... ({cur}/{rep})\n   -- Deleting edges... (  0%)")
            tot = len(perturbed._network.edges)
            for i, (remove, (u, v)) in enumerate(list(zip( np.random.binomial(n=1, p=p, size=len(perturbed._network.edges)), perturbed._network.edges ))):
                if remove: 
            # for u, v in sc._network.edges:
            #     if np.random.binomial(n=1, p=p):
                    perturbation_details['Delta'] += 1
                    perturbed._network.remove_edge(u, v)
                verboser.update(f"Perturbing Supply Chain... ({cur}/{rep})\n   -- Deleting edges... ({int(100*(i+1)/tot):>{3}}%)")
            end = time.time()
            perturbation_details['time'] = end - start     
            verboser.terminate(f"Perturbation successfully computed. ({cur}/{rep}) ({end-start:.2f}s)")   
        return perturbed, perturbation_details

    @staticmethod
    def _perturbe_normal(sc: SupplyChain, c_mu=0, c_sigma=1, u_mu=0, u_sigma=1, verbose=False, cur=1, rep=1):
        with Verboser(verbose, f"Perturbing Supply Chain... ({cur}/{rep})") as verboser:
            start = time.time()
            perturbation_details = { 'Delta': 0 }
            verboser.update(f"Perturbing Supply Chain... ({cur}/{rep})\n   -- Copying original Chain...")
            perturbed = SupplyChain(supply_chain=sc)
            verboser.update(f"Perturbing Supply Chain... ({cur}/{rep})\n   -- Perturbing edges... (  0%)")
            tot = len(perturbed._network.edges)
            for i, (u, v) in enumerate(perturbed._network.edges):
                c_delta, p_delta = np.random.normal(c_mu, c_sigma), np.random.normal(u_mu, u_sigma)
                perturbation_details['Delta'] += abs(c_delta) + abs(p_delta)
                perturbed._network[u][v]['cost'] = max(0, perturbed._network[u][v]['cost'] + c_delta)
                perturbed._network[u][v]['cap'] = max(0, perturbed._network[u][v]['cap'] + p_delta)
                verboser.update(f"Perturbing Supply Chain... ({cur}/{rep})\n   -- Perturbing edges... ({int(100*(i+1)/tot):>{3}}%)")
            end = time.time()
            perturbation_details['time'] = end-start
            verboser.terminate(f"Perturbation successfully computed. ({cur}/{rep}) ({end-start:.2f}s)")   
        return perturbed, perturbation_details
    

if __name__ == "__main__":
    # cl = SupplyChainBackbone.tiny_example(prune=True, integrity='full')
    # clc = ChungLuConverter(cl)
    # clc.to_json(filename='temp/temp.json', duration=20, supply=1e9, demand=1e9)
    # sc = SupplyChain(JSONfilename='temp/temp.json')
    # print(f"Original cost = {sc.cost()}")
    # print(f"Original production = {sc.production()}")
    # print(f"Original optimisation time = {sc.optimisation_time():.2f}s")
    # perturbed, _ = SupplyChainPerturber.perturbe(sc=sc, type='normal', c_mu=0, u_mu=0, c_sigma=.1, u_sigma=.2)
    # print(f"Perturbed cost = {perturbed.cost()}")
    # print(f"Perturbed production = {perturbed.production()}")
    # print(f"Perturbed optimisation time = {perturbed.optimisation_time():.2f}s")
    
    sc = SupplyChain.tiny_pyramid(duration=20, verbose=True)
    sc_perturbed, _ = SupplyChainPerturber.perturbe(sc, type='binomial', verbose=True)
    

    # start = time.time()
    # sc.find_optimum()
    # end = time.time()
    # print(f"Computing optimum took {end - start:.2f}s.")

    # start = time.time()
    # perturbed = SupplyChainPerturber.perturbe(sc = sc)
    # end = time.time()
    # print(f'Generating one perturbation took {end - start:.2f}s.')

    # start = time.time()
    # for _ in range(100):
    #     perturbed = SupplyChainPerturber.perturbe(sc = sc)
    # end = time.time()
    # print(f'Generating 100 perturbations took {end - start:.2f}s.')



    pass
