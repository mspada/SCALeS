import json
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from .. import config
from .perturber import SupplyChainPerturber

if TYPE_CHECKING:
    from ..model.supplychain import SupplyChain


"""
This class allows to analyse the perturbations of a supply chain. 
No instance can be created, all methods are static. 

Two methods methods --- analyse_perturbation and plot_from_data --- can be called. 
    - analyse_perturbation: takes a supply chain, sample some perturbations and measure the impact of perturbations.
        Arguments are:
        - filename (str): the file in which the results generated should be stored.
            None uses results/perturbation/perturbation_data.json.
        - rep (int, > 0): the number of perturbations to generate.
        - other: any other parameter will be transmitted to SupplyChainPerturber.perturbe.
    - plot_from_data: plots the results from analyse_perturbation.
        Arguments are: 
        - filename (str): the file from which to read the results.
        - plot_filename (str): where to save the plot. Default is results/perturbation/perturbation_analysis.png.
        - c_binwidth (float): the size of the bins when displaying the cost.
        - u_binwidth (float): the size of the bins when displahing the production.
"""


class SupplyChainAnalyser:

    @staticmethod
    def analyse_perturbation(filename: str, sc: 'SupplyChain', **parameters) -> str:
        """
        Samples a specified number of stochastic perturbations and calculates the impact on flow optimization.
        Writes all output data (costs, productions, variation metrics, and runtimes) to filename in JSON format.
        Returns the path of the written file.
        """
        if filename is None:
            filename = str(config.results_dir("perturbation") / "perturbation_data.json")
        rep = parameters['rep'] if 'rep' in parameters and isinstance(parameters['rep'], (int, np.integer)) else 100
        verbose = parameters.get('verbose', False)
                
        costs, productions, Deltas, opt_times, pert_times = [], [], [], [], []

        for i in range(rep): 
            parameters['state'] = (i+1, rep)
            parameters['verbose'] = verbose
            perturbed, perturbation_details = SupplyChainPerturber.perturbe(sc, **parameters)
            perturbed._state = (i+1, rep)
            perturbed.find_optimum()
            costs.append(perturbed.cost())
            productions.append(perturbed.production())
            Deltas.append(perturbation_details['Delta'])
            opt_times.append(perturbed.optimisation_time())
            pert_times.append(perturbation_details['time'])

        data = { 
            'input': {
                'origin': 'analyse_perturbation',
                'rep': rep,
                'parameters': parameters
            },
            'output': {
                'size': sc.size(),
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

        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as jsonfile: 
            json.dump(data, jsonfile, indent=4)
        return filename

    @staticmethod
    def plot_from_data(filename: str, c_binwidth: float = .5, u_binwidth: float = .5, plot_filename: str = None) -> None: 
        """
        Loads perturbation data from filename and saves clean multi-subplot distribution and variation charts.
        Saves headless-friendly output directly to plot_filename (or results/perturbation/perturbation_analysis.png).
        """
        with open(filename, 'r') as jsonfile:
            data = json.load(jsonfile)
        
        input = data['input']
        output = data['output']

        rep = input['rep']
        size = output.get('size', 'TBD')
        og_cost, og_prod, costs, prods, Deltas = output['og_cost'], output['og_prod'], output['costs'], output['prods'], output['Deltas']
        opt_times, pert_times = output['opt_times'], output['pert_times']

        c_start, c_end = 0, max(costs) + c_binwidth
        u_start, u_end = 0, max(prods) + u_binwidth
        c_bins = np.arange(c_start, c_end, c_binwidth)
        u_bins = np.arange(u_start, u_end, u_binwidth)

        plt.figure(figsize=(15, 9))
        plt.suptitle(f"Statistics under stochastic perturbation\n(size: {size} -- {rep} repetitions -- {np.average(opt_times):.2f}s for opt -- {np.average(pert_times):.2f}s for pert)", fontsize=14)

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
        slope, intercept, r2 = _linear_fit(prods, costs)
        plt.title(fr"Cost against production $R^2=${r2:.2f}")
        plt.scatter(prods, costs)
        plt.scatter( [og_prod], [og_cost], color='red')
        plt.plot( prods, slope * np.array(prods) + intercept, color='orange' )

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

        if plot_filename:
            if os.path.dirname(plot_filename):
                os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
            plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(config.results_dir("perturbation") / "perturbation_analysis.png", format='png', dpi=300, bbox_inches='tight')
        plt.close()


def _linear_fit(x, y) -> tuple[float, float, float]:
    """Least-squares line y = slope * x + intercept and its coefficient of determination R^2."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.ptp(x) == 0:
        return 0.0, float(np.mean(y)), 0.0
    slope, intercept = np.polyfit(x, y, 1)
    ss_res = np.sum((y - (slope * x + intercept)) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), float(r2)
