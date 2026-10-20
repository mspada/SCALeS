import time
from typing import TYPE_CHECKING

import numpy as np

from ..verboser import Verboser

if TYPE_CHECKING:
    from ..model.supplychain import SupplyChain


"""
This class allows to perturbe a supply chain. 
No instance can be created, all methods are static. 

One method (perturbe) can be called. It takes a supply chain and return a new (unsolved) supply chain, obtained by perturbing it.
Its arguments are:
    - sc (SupplyChain): the SupplyChain to be perturbed;
    - type (str, 'normal' or 'binomial'): the type of perturbation to be performed.
    - p (float, 0 <= p <=1): when perturbing under 'binomial' mode, the probability of deleting each edge
    - mu (float): when perturbing under 'normal' mode, the expected value of the random normal noise added to the cost and capacity of each edge.
        Can be replaced by u_mu and c_mu (capacity and cost _mu defined differently) for finer control of the expected value of the capacity and cost respectively.
    - sigma (float): when perturbing under 'normal' mode, the standard deviation of the random normal noise added to the cost and capacity of each edge.
        Can be replaced by u_sigma and c_sigma (capacity and cost _sigma defined differently) for finer control of the expected value of the capacity and cost respectively.
    - verbose (bool): details the execution of each step when set to True. 
"""

class SupplyChainPerturber:

    @staticmethod
    def perturbe(sc: 'SupplyChain', **parameters) -> tuple['SupplyChain', dict]:
        """
        Public entry point to stochastically perturb a given Supply Chain network.
        Accepts 'type' ('binomial' or 'normal'), parameters (p, mu, sigma, c_mu, c_sigma, u_mu, u_sigma),
        and verbose options. Returns the perturbed SupplyChain instance along with details.
        """
        verbose = parameters.get('verbose', False)
        cur, tot = parameters.get('state', (1,1))

        p_type = parameters['type'] if 'type' in parameters and parameters['type'] in ['normal', 'binomial']\
                else 'normal'
        if p_type == 'binomial': 
            p = parameters['p'] if 'p' in parameters and isinstance(parameters['p'], (int, float, np.integer, np.floating)) and 0 <= parameters['p'] <= 1 else .1
            return SupplyChainPerturber._perturbe_binomial(sc=sc, p=p, verbose=verbose, cur=cur, rep=tot)
        # if none of the above applies, do normal perturbation
        c_mu = parameters['c_mu'] if 'c_mu' in parameters and isinstance(parameters['c_mu'], (int, float, np.integer, np.floating)) \
                else parameters['mu'] if 'mu' in parameters and isinstance(parameters['mu'], (int, float, np.integer, np.floating)) \
                else 0
        u_mu = parameters['u_mu'] if 'u_mu' in parameters and isinstance(parameters['u_mu'], (int, float, np.integer, np.floating)) \
                else parameters['mu'] if 'mu' in parameters and isinstance(parameters['mu'], (int, float, np.integer, np.floating)) \
                else 0
        c_sigma = parameters['c_sigma'] if 'c_sigma' in parameters and isinstance(parameters['c_sigma'], (int, float, np.integer, np.floating)) \
                else parameters['sigma'] if 'sigma' in parameters and isinstance(parameters['sigma'], (int, float, np.integer, np.floating)) \
                else .1
        u_sigma = parameters['u_sigma'] if 'u_sigma' in parameters and isinstance(parameters['u_sigma'], (int, float, np.integer, np.floating)) \
                else parameters['sigma'] if 'sigma' in parameters and isinstance(parameters['sigma'], (int, float, np.integer, np.floating)) \
                else .1
        return SupplyChainPerturber._perturbe_normal(sc=sc, c_mu=c_mu, u_mu=u_mu, c_sigma=c_sigma, u_sigma=u_sigma, verbose=verbose, cur=cur, rep=tot)



    @staticmethod
    def _perturbe_binomial(sc: 'SupplyChain', p: float = .1, verbose: bool = False, cur: int = 1, rep: int = 1) -> tuple['SupplyChain', dict]:
        """Stochastically deletes edges in the supply chain with independent probability p."""
        with Verboser(verbose, f"Perturbing Supply Chain... ({cur}/{rep})") as verboser:
            start = time.time()
            perturbation_details = { 'Delta': 0 }
            verboser.update(f"Perturbing Supply Chain... ({cur}/{rep})\n   -- Copying original Chain...")
            perturbed = sc.copy()
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
    def _perturbe_normal(sc: 'SupplyChain', c_mu: float = 0, c_sigma: float = 1, u_mu: float = 0, u_sigma: float = 1, verbose: bool = False, cur: int = 1, rep: int = 1) -> tuple['SupplyChain', dict]:
        """Stochastically perturbs edge costs and capacities by adding random normal noise."""
        with Verboser(verbose, f"Perturbing Supply Chain... ({cur}/{rep})") as verboser:
            start = time.time()
            perturbation_details = { 'Delta': 0 }
            verboser.update(f"Perturbing Supply Chain... ({cur}/{rep})\n   -- Copying original Chain...")
            perturbed = sc.copy()
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
