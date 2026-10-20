import numpy as np

from scales import SupplyChainPerturber, find_bottlenecks
from scales.analysis.analyser import _linear_fit


def test_binomial_perturbation_leaves_original_untouched(supply_chain):
    edges_before = len(supply_chain._network.edges)
    np.random.seed(0)
    perturbed, details = SupplyChainPerturber.perturbe(supply_chain, type='binomial', p=.2)
    assert len(supply_chain._network.edges) == edges_before
    assert len(perturbed._network.edges) == edges_before - details['Delta']
    assert perturbed._optimum_found is False


def test_normal_perturbation_keeps_costs_and_capacities_non_negative(supply_chain):
    perturbed, _ = SupplyChainPerturber.perturbe(supply_chain, type='normal', sigma=50)
    for u, v, d in perturbed._network.edges(data=True):
        assert d['cost'] >= 0 and d['cap'] >= 0


def test_linear_fit():
    slope, intercept, r2 = _linear_fit([0, 1, 2, 3], [1, 3, 5, 7])
    assert np.isclose(slope, 2) and np.isclose(intercept, 1) and np.isclose(r2, 1)


def test_find_bottlenecks_sorted_by_implied_production(supply_chain):
    bottlenecks = find_bottlenecks(supply_chain, top_n=5, supply_limit=1000)
    implied = [b['implied_max_oem'] for b in bottlenecks]
    assert implied == sorted(implied)
