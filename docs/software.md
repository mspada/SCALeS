# SCALeS software reference

This document is the software supplement of the paper (§5 "Software supplement"). It describes how the reference
implementation is organised and which data structures it uses. The theory behind each step is described in the
paper; the section numbers below refer to it.

## Pipeline and package layout

The code follows the two steps of the paper (backbone, then quantitative supply chain), followed by optimisation
and analysis. Every stage can be used on its own and writes its default output to its own folder:

| Stage | Class | Paper | Package | Default output |
| --- | --- | --- | --- | --- |
| 1. Backbone | `SupplyChainBackbone` | §2, Algorithms 1-3 | `scales/backbone/` | `networks/backbones/` |
| 2. Conversion | `ScbConverter` | §3.1-3.5 | `scales/conversion/` | `supply_chains/` (JSON), `networks/supply_chains/` |
| 3. Dynamic flow model | `SupplyChain` | §3 | `scales/model/` | `networks/dynamic/`, `results/lp/`, `results/diagnostics/` |
| 4. Analysis | `SupplyChainPerturber`, `SupplyChainAnalyser` | §4 | `scales/analysis/` | `results/perturbation/` |

```text
src/scales/
├── config.py               binary discovery (glpsol, neato) and output folders
├── backbone/
│   ├── generator.py        SupplyChainBackbone: layered Chung-Lu with truncated power law (Alg. 1), pruning (Alg. 2)
│   └── warehouse.py        warehouse injection with split β (Alg. 3, §2.3)
├── conversion/
│   ├── converter.py        ScbConverter: backbone → static network (Fig. 5, 8) → JSON
│   └── parameters.py       one function per distribution of §3.5
├── model/
│   ├── supplychain.py      SupplyChain: time expansion (Fig. 6, 7), parameter getters/setters, cost/production
│   ├── ampl.py             export of the dynamic network as AMPL data
│   ├── solver.py           running glpsol and reading the solution
│   └── lp/supplychain-lp-model.mod   the LP model
├── analysis/
│   ├── perturber.py        binomial arc deletion, normal noise on costs and capacities
│   ├── analyser.py         repeated perturbation experiments and plots
│   ├── bottlenecks.py      stoichiometric bottleneck ranking
│   └── diagnostics.py      LP constraint reports, flow and slack-variable tables
└── viz/
    ├── visualiser.py       NetworkVisualizer (png, svg, dot, window)
    └── drawing.py          drawings of the static and dynamic SupplyChain networks
```

`SupplyChain` combines `model/ampl.py`, `analysis/diagnostics.py` and `viz/drawing.py` as mixins. All their
methods are therefore available directly on a `SupplyChain` instance.

## Configuration

`glpsol` (GLPK) is needed to solve the LP and `neato` (Graphviz) only for `SupplyChain.draw_dynamic()`. They are
looked up on the `PATH`. To use other locations, set the environment variables `SCALES_GLPSOL` / `SCALES_NEATO` or
call `scales.config.set_glpsol(path)` / `set_neato(path)`.

Default outputs go below the current working directory. To change this, set `SCALES_OUTPUT_DIR` or call
`scales.config.set_output_root(path)`. Any explicit `filename` argument is used exactly as given.

## Data structures

### Backbone (`SupplyChainBackbone.network`, a `networkx.DiGraph`)

- **Vertex**: a tuple `(X, n)`, where `X` is the layer index (0 = raw materials, L-1 = final producers/OEM) and `n` is
  the index within the layer. Vertices are labelled `A0`, `B12`, … in drawings (layer letter + index). Use
  `get_node_from_label("B12")` to get the tuple.
- **Vertex attributes**: `layer`, `weight`, `top_layer`, `bot_layer`, `type` (`'P'` production or `'W'` warehouse)
  and, for warehouses, `w_id`.
- **Edges** `((X, n), (Y, m))` always point from the lower to the higher layer (or stay within a layer) and carry no
  attributes.

To add attributes to a vertex or an edge:
```python
backbone.network.nodes[(X, n)]['new_attribute'] = value
backbone.network.edges[(X, n), (Y, m)]['time_delay'] = 5
```

### Static supply chain (`ScbConverter`, JSON file)

Every production vertex `P` becomes the nodes `P(I0)…P(Ik)` (one input storage per incoming edge), `P(PI)`
(production input, which carries the production rule), `P(PO)` (production output) and `P(O0)…P(Om)` (one output
storage per consumer). A warehouse vertex `W` becomes a single node `W(W)`.

The JSON file has two keys:
- `nodes`: `{name: {"type", "rule", "layer", "storage_cap"?, "backbone_id"}}`, where `type` is one of
  `MATERIAL` (input storage in layer 0, receives supply), `OEM` (output storage in the top layer, receives demand),
  `LOCAL_STORAGE`, `PRODUCTION_INPUT`, `PRODUCTION_OUTPUT` or `WAREHOUSE`. `rule` maps every input storage of a
  `PRODUCTION_INPUT` node to the number of units needed per output unit (§3.1).
- `edges`: a list of `{"from", "to", "cost", "cap", "time", "backbone_id"}`. Arcs between storages of different
  vertices are transport arcs (§3.5.6-3.5.8). The `PI → PO` arc is the production arc (§3.5.2-3.5.4).

### Dynamic network (`SupplyChain._network`)

Every static node `name` is replicated for each time step: `(name, t)` for `t = 0 … duration-1`. An arc with delay
`d` connects `(u, t)` to `(v, t+d)`. Storage nodes additionally connect `(u, t)` to `(u, t+1)`, with the node's
`storage_cap` as capacity. After `find_optimum()`, arcs carry `flow`, and nodes carry `generation`, `absorption`
and `backorder`.

## Extending the model

- **New backbone features**: add keyword arguments in `SupplyChainBackbone._parse_kwargs` and the corresponding
  step in `_generate` or in a new module next to `warehouse.py`.
- **Different parameter distributions**: change or add functions in `conversion/parameters.py`; the converter
  passes its own random generator, so seeded runs stay reproducible.
- **Other solvers**: `SupplyChain.to_ampl_data()` returns the complete AMPL data for `model/lp/supplychain-lp-model.mod`,
  which any AMPL-compatible solver can read. To plug in a different solver, replace `model/solver.py`.
