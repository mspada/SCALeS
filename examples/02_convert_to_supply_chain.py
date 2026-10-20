"""
Stages 1-2: turn a backbone into a static, quantitative supply chain network (paper §3.1-3.5) and save it as JSON.

Output: supply_chains/   (the JSON instance, input for stage 3)
        networks/supply_chains/, networks/diagnostics/static/
No solver needed. Stop here if you only need supply chain instances, e.g. for your own solution method.
"""

from pathlib import Path

from scales import SupplyChainBackbone, ScbConverter, config

config.set_output_root(Path(__file__).resolve().parents[1])

backbone = SupplyChainBackbone(size=[10, 20, 10, 1], prune=True, w=0.1, betawin=3, betawout=10, beta={(2, 3): 10}, seed=43)

# All §3.5 distribution parameters, shown with their default values.
converter = ScbConverter(
    backbone, seed=42,
    a=2.0,                                    # §3.5.1 Zipf exponent of the bills of materials
    sigma=0.2, c_base=100.0,                  # §3.5.2 production capacity
    pdelay_c=0.5, pdelay_l=1.0, pdelay_e=10.0,  # §3.5.3 production delay
    pcost_base=100.0, pcost_sigma=0.10,       # §3.5.4 production cost
    storage_buffer=4.0, storage_sigma=0.20,   # §3.5.5 local storage
    w_vol_mult=500.0, w_sigma=0.30,           # §3.5.5 warehouses
    t_layer_penalty=3.0, t_hub_base=10.0,     # §3.5.6 transport delay
    t_over_prov=1.25, t_cap_sigma=0.30,       # §3.5.7 transport capacity
    t_cost_base=50.0, t_cost_sigma=0.10,      # §3.5.8 transport cost
)

path = converter.to_json(str(config.supply_chains_dir() / "example_mid.json"))
print("Supply chain written to", path)

converter.draw()                              # networks/supply_chains/supply_chain.png
for label in ["A1", "B13", "D0"]:
    converter.draw_node_diagnostic(label)     # networks/diagnostics/static/diagnostic_<label>.png
