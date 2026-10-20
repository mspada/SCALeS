"""
Stage 1 only: generate a Supply Chain Backbone (paper §2) and draw it.

Output: networks/backbones/
No solver needed.
"""

from pathlib import Path

from scales import SupplyChainBackbone, config

config.set_output_root(Path(__file__).resolve().parents[1])

# Four layers (raw materials -> ... -> one OEM), pruned, with 10% warehouses.
# betawin/betawout make warehouses collect from few producers and distribute to many consumers (§2.3.1),
# beta={(2, 3): 10} makes the single top-layer OEM very unlikely to be pruned (§2.4 one company model).
backbone = SupplyChainBackbone(
    size=[10, 20, 10, 1],
    prune=True,
    w=0.1, betawin=3, betawout=10,
    beta={(2, 3): 10},
    seed=43,
)

print("Layer sizes before / after pruning:", backbone.original_size(), backbone.size())
print("Edges between layers:", backbone.edge_number())
warehouses = [n for n, d in backbone.network.nodes(data=True) if d['type'] == 'W']
print("Warehouse vertices:", len(warehouses))

node = backbone.get_node_from_label("B3")
print("Vertex B3:", node, backbone.network.nodes[node])

backbone.draw()                                   # networks/backbones/backbone.png
backbone.draw(filename=str(config.networks_dir("backbones") / "backbone_seed43"))

# Other ways to choose the layer sizes and weights:
#   SupplyChainBackbone(size=1000, prune=True, integrity='partial', vratio=.4)   # default pyramid of ~1000 vertices
#   SupplyChainBackbone(size=[3, 5, 3], tau=[2.2, 2.3, 2.4])                      # one power-law exponent per layer
#   SupplyChainBackbone(size=[5, 12, 6, 2], kc=[3, 4, 3, 1])                      # explicit truncation cut-offs
#   SupplyChainBackbone(weights=[[1, 2, 1], [2, 3, 2, 3, 2], [2, 1, 2]])          # explicit weights
#   SupplyChainBackbone.tiny_example(), .small_example(), .medium_example()       # presets
