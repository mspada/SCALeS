"""
Stage 4: resilience analysis. Perturbs a supply chain many times and records cost and production.

Output: results/perturbation/
Needs `glpsol` (GLPK).
"""

from pathlib import Path

from scales import SupplyChain, SupplyChainAnalyser, SupplyChainPerturber, config

config.set_output_root(Path(__file__).resolve().parents[1])

sc = SupplyChain.tiny_pyramid(duration=20)

# A single perturbation: 'normal' adds noise to costs and capacities (c_mu/c_sigma, u_mu/u_sigma for finer control),
# 'binomial' deletes every arc with probability p.
perturbed, details = SupplyChainPerturber.perturbe(sc, type='binomial', p=.1)
print(f"Original:  cost={sc.cost():.2f}  production={sc.production():.2f}")
print(f"Perturbed: cost={perturbed.cost():.2f}  production={perturbed.production():.2f}  (arcs removed: {details['Delta']})")

# Repeated perturbations, stored as JSON and plotted.
data_file = SupplyChainAnalyser.analyse_perturbation(None, sc, rep=10, type='binomial', p=.1, verbose=True)
SupplyChainAnalyser.plot_from_data(data_file, c_binwidth=5, u_binwidth=3)
print("Results in", Path(data_file).parent)
