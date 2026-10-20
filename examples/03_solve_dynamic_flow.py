"""
Stage 3: expand a supply chain over time into a dynamic network flow (paper §3) and solve it with GLPK.

Input:  supply_chains/example_mid.json   (run 02_convert_to_supply_chain.py first)
Output: networks/dynamic/, networks/diagnostics/dynamic/, results/lp/, results/diagnostics/
Needs `glpsol` (GLPK); `draw_dynamic` additionally needs `neato` (Graphviz). See scales/config.py.
"""

from pathlib import Path

from scales import SupplyChain, config, print_bottlenecks

config.set_output_root(Path(__file__).resolve().parents[1])

json_path = config.supply_chains_dir() / "example_mid.json"
if not json_path.exists():
    raise SystemExit(f"{json_path} not found: run examples/02_convert_to_supply_chain.py first.")

# Building the dynamic network does not solve it, so it can be inspected or exported first.
sc = SupplyChain(JSONfilename=str(json_path), duration=60, demand=[10, 20], supply=1000, save_temp=True)
print("Static nodes:", sc.size(), "| dynamic nodes:", len(sc._network.nodes))
print("Minimum duration needed to reach the OEM:", sc.calculate_critical_propagation_delay())

sc.find_optimum()                             # keeps the AMPL data in results/lp/model.dat
print(f"Cost: {sc.cost():.2f}  Production: {sc.production():.2f}  ({sc.optimisation_time():.2f}s)")

# Diagnostics, all written below results/diagnostics/ and networks/diagnostics/
sc.layer_flow_input(layer=3)
sc.layer_flow_output(layer=3)
sc.diagnostic_backorder()
sc.diagnostic_generation()
sc.diagnostic_absorption()
sc.draw_node_diagnostic(label='B13', t=10)
sc.lp_node_diagnostic(label='B13', t=10)
print_bottlenecks(sc, top_n=10, supply_limit=1000)

# Change parameters and solve again: set_* rebuilds the network and discards the previous solution.
transport = next(e for e in sc.edges if '(O' in e['from'] and '(I' in e['to'])
print("Transport", transport['from'], "->", transport['to'], sc.get_transport(False, transport['from'], transport['to']))
sc.set_transport(False, transport['from'], transport['to'], transport_time=2, transport_cost=30.0, transport_capacity=500)
print(f"Cost after the change: {sc.cost():.2f}")

# Needs Graphviz:
# sc.draw_dynamic(show=False, filetype='svg', expanding_factor=30)   # networks/dynamic/network.svg

# Stages 1-3 in one call, without writing JSON: SupplyChain.tiny_pyramid(), .small_pyramid(), ...
