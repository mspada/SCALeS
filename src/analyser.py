import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import time
import json

from scb import SupplyChainBackbone
from chunglu_converter import ChungLuConverter
from supplychain import SupplyChain
from perturber import SupplyChainPerturber


"""
This class allows to analyse the perturbations of a supply chain. 
No instance can be created, all methods are static. 

Two methods methods --- analyse_perturbation and plot_from_data --- can be called. 
    - analyse_perturbation: takes a supply chain, sample some perturbations and measure the impact of perturbations.
        Arguments are:
        - filename (str): the file in which the results generated should be stored.
        - rep (int, > 0): the number of perturbations to generate.
        - other: any other parameter will be transmitted to SupplyChainPerturber.perturbe.
    - plot_from_data: plots the results from analyse_perturbation.
        Arguments are: 
        - filename (str): the file from which to read the results.
        - c_binwidth (float): the size of the bins when displaying the cost.
        - u_binwidth (float): the size of the bins when displahing the production.
"""


class SupplyChainAnalyser:

    @staticmethod
    def analyse_perturbation(filename: str, sc: SupplyChain, **parameters):
        rep = parameters['rep'] if 'rep' in parameters and type(parameters['rep']) == int else 100
        verbose = parameters.get('verbose', False)
                
        costs, productions, Deltas, opt_times, pert_times = [], [], [], [], []

        for i in range(rep): 
            perturbed, perturbation_details = SupplyChainPerturber.perturbe(sc, state=(i+1, rep), **parameters)
            perturbed.find_optimum(verbose=verbose, state=(i+1, rep))
            costs.append(perturbed.cost())
            productions.append(perturbed.production())
            Deltas.append(perturbation_details['Delta'])
            opt_times.append(perturbed.optimisation_time())
            pert_times.append(perturbation_details['time'])

        # TODO -- Think about adding size, number of edges, etc to the data
        data = { 
            'input': {
                'origin': 'analyse_perturbation',
                'rep': rep,
                'parameters': parameters
            },
            'output': {
                'og_cost': sc.cost(),
                'og_prod': sc.production(),
                'og_opt_time': sc.optimisation_time(),
                'costs': costs,
                'prods': productions, 
                'Deltas': Deltas,
                'opt_times': opt_times,
                'pert_times': pert_times
            }
        }

        with open(filename, 'w') as jsonfile: 
            json.dump(data, jsonfile, indent=4)

    @staticmethod
    def plot_from_data(filename: str, c_binwidth=.5, u_binwidth=.5): 
        with open(filename, 'r') as jsonfile:
            data = json.load(jsonfile)
        
        input = data['input']
        output = data['output']

        rep = input['rep']
        og_cost, og_prod, costs, prods, Deltas = output['og_cost'], output['og_prod'], output['costs'], output['prods'], output['Deltas']
        opt_times, pert_times = output['opt_times'], output['pert_times']

        c_start, c_end = 0, max(costs) + c_binwidth
        u_start, u_end = 0, max(prods) + u_binwidth
        c_bins = np.arange(c_start, c_end, c_binwidth)
        u_bins = np.arange(u_start, u_end, u_binwidth)

        plt.figure(figsize=(15, 9))
        plt.suptitle(f"Statistics under stochastic perturbation\n(size: TBD -- {rep} repetitions -- {np.average(opt_times):.2f}s for opt -- {np.average(pert_times):.2f}s for pert)", fontsize=14)

        plt.subplot(2,3,1)
        plt.title("Cost distribution")
        val, _, _ = plt.hist(costs, c_bins, label='costs')
        plt.vlines(x=og_cost, ymin=0, ymax=max(val), label='og cost', color='red')
        plt.vlines(x=np.average(costs), ymin=0, ymax=max(val), label='avg cost', color='orange')
        plt.legend()

        plt.subplot(2,3,2)
        plt.title("Production distribution")
        val, _, _ = plt.hist(prods, u_bins, label='prods')
        plt.vlines(x=og_prod, ymin=0, ymax=max(val), label='og prod', color='red')
        plt.vlines(x=np.average(prods), ymin=0, ymax=max(val), label='avg prod', color='orange')
        plt.legend()

        plt.subplot(2,3,3)
        pc_model = LinearRegression()
        pc_model.fit( np.array(prods).reshape((-1,1)), np.array(costs) )
        plt.title(fr"Cost against production $R^2=${pc_model.score(np.array(prods).reshape((-1,1)), np.array(costs)):.2f}")
        plt.scatter(prods, costs)
        plt.scatter( [og_prod], [og_cost], color='red')
        plt.plot( prods, pc_model.predict(np.array(prods).reshape((-1,1))), color='orange' )

        plt.subplot(2,3,4)
        plt.title('Cost variation against total variation')
        X = Deltas
        # Y = np.array(costs) - self.supply_chain.cost()
        Y = np.abs(np.array(costs) - og_cost)
        plt.scatter(X, Y)

        plt.subplot(2,3,5)
        plt.title('Production variation against total variation')
        X = Deltas
        # Y = np.array(prods) - self.supply_chain.production()
        Y = np.abs(np.array(prods) - og_prod)
        plt.scatter(X, Y)
        plt.show()

if __name__ == "__main__":
    filename = 'results/perturbation_data_output.json'    
    cl = SupplyChainBackbone.medium_example(prune=True, integrity='pertial', vratio=.8)
    clc = ChungLuConverter(cl, verbose=True)
    clc.to_json(filename='temp/temp.json', duration=50, demand=1e9, supply=1e9, verbose=True)
    sc = SupplyChain('temp/temp.json', verbose=True)
    SupplyChainAnalyser.analyse_perturbation(filename, sc, type='normal', p=.1, rep=100, verbose=True)
    SupplyChainAnalyser.plot_from_data(filename, c_binwidth=10, u_binwidth=1)
