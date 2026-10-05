# Large-Scale Supply Chain Generation & Simulation

Reference implementation accompanying the working paper *"Simple generation, modeling and simulation of realistic very large scale supply chain instances"* (Hruz, Larcher, Ding, Weingart, Spada, Scherrer — ETH Zurich / ZHAW, 2026).

This project generates **realistic, large-scale synthetic supply chain instances** and simulates their behaviour — including production, transport, storage, and cost optimization — using dynamic network flows. It is designed to give researchers access to non-trivial supply chain test instances, since real-world supply chain data is almost always a closely guarded company secret.

> **Status:** Work in progress / research code. APIs, parameter names, and file layout may still change.

## How it works

The pipeline has two conceptual stages, mirroring the structure of the paper:

1. **Backbone generation** — a layered Chung-Lu random graph models the hierarchical producer/consumer structure of a supply chain (raw material producers → intermediate manufacturers → OEM), with degree distributions following a truncated power law and a configurable warehouse layer injected on top.
2. **Quantitative supply chain construction** — the backbone is expanded into a full dynamic network flow model with production rules, transport arcs, local storage nodes, costs, capacities, and time delays, which can then be solved for an optimal cost allocation using an LP/MILP solver.

On top of this, a perturbation/analysis layer allows sampling random disruptions (edge removals, cost/capacity shocks) to study supply chain resilience.

## Module overview

| File | Class | Role |
|---|---|---|
| `src/scb.py` | `SupplyChainBackbone` | Generates the layered Chung-Lu backbone graph (vertices = production/warehouse sites, edges = supplier relationships). Handles weight sampling, pruning, and warehouse injection. |
| `src/chunglu_converter.py` | `ChungLuConverter` | Converts a `SupplyChainBackbone` instance into a full, timeless supply chain network (production nodes, local storage, transport arcs) and serializes it to JSON. |
| `src/supplychain.py` | `SupplyChain` | Loads a supply chain from JSON (or copies another instance), expands it into a dynamic (time-indexed) network flow problem, exports it to AMPL, solves it via `glpsol`, and exposes cost/production metrics. |
| `src/perturber.py` | `SupplyChainPerturber` | Generates perturbed copies of a `SupplyChain` (binomial edge deletion or normal noise on cost/capacity) to model disruptions. |
| `src/analyser.py` | `SupplyChainAnalyser` | Runs repeated perturbation experiments, stores results to JSON, and plots cost/production distributions under stochastic perturbation. |
| `src/verboser.py` | `Verboser` | Small helper for nice spinner-based progress output (built on `rich`), used throughout the other modules. |
| `src/local.py.template` | — | Template for machine-local paths (`glpsol`, `neato` binaries). Copy to `src/local.py` and edit before running anything that solves or draws via Graphviz. |
| `src/example.py` | — | Walkthrough script (mostly annotated, commented-out snippets) demonstrating how the three core classes — `SupplyChainBackbone`, `ChungLuConverter`, `SupplyChain` — fit together. |

### Data model notes

- Backbone vertices are `(layer_index, node_id)` tuples; layer 0 is the bottom (raw materials), the last layer is the top (final producers/OEM).
- After conversion, supply chain nodes become `(layer, node_id, subtype)` tuples, where subtype is one of `(I{k})` (input/local storage), `(P)` (production), or `(O)` (output/local storage).
- The dynamic network further indexes every node by a discrete time step, so a fully expanded supply chain node is `((layer, node_id, subtype), t)`.
- See Section 5.2 of the paper for the precise NetworkX attribute conventions used internally.

## Getting started

### Requirements

- Python 3.9+ (tested with 3.9.6)
- A GLPK installation (`glpsol`) for solving the LP/MILP model
- Graphviz (`neato`) if you want rendered `.dot`/`.pdf` network diagrams

### Virtual environment setup

Create and activate a virtual environment in the repository root, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The `.venv/` folder is git-ignored, so each user creates their own. To leave the environment, run `deactivate`. In Positron or VS Code, select the `.venv` interpreter from the interpreter picker.

### Local configuration

Several modules shell out to external binaries (`glpsol` for optimization, `neato` for graph layout). Paths are **machine-specific** and therefore not committed to the repository. Before running anything that solves or draws a network:

```bash
cp src/local.py.template src/local.py
# then edit src/local.py with the correct paths for your machine
```

### Minimal example

```python
from scb import SupplyChainBackbone
from chunglu_converter import ChungLuConverter
from supplychain import SupplyChain

# 1. Generate a small layered backbone graph
backbone = SupplyChainBackbone.tiny_example(prune=True, integrity="full")

# 2. Convert it into a full supply chain and write it to JSON
converter = ChungLuConverter(backbone)
converter.to_json("supply_chains/example.json", duration=10, supply=1e9, demand=1)

# 3. Load it as a dynamic flow network and solve for the optimal allocation
sc = SupplyChain(JSONfilename="supply_chains/example.json")
sc.find_optimum(verbose=True)
print(f"Total cost: {sc.cost():.2f}")
print(f"Total production: {sc.production():.2f}")
```

See `src/example.py` for a more complete tour of available parameters (weight distributions, layer sizes, pruning behaviour, warehouse injection, etc.).

### Running a perturbation study

```python
from analyser import SupplyChainAnalyser

SupplyChainAnalyser.analyse_perturbation(
    "results/perturbation_data.json", sc, type="normal", sigma=0.1, rep=100, verbose=True
)
SupplyChainAnalyser.plot_from_data("results/perturbation_data.json", c_binwidth=10, u_binwidth=1)
```

## Repository layout

```
.
├── src/                  # all source modules
│   ├── scb.py
│   ├── chunglu_converter.py
│   ├── supplychain.py
│   ├── perturber.py
│   ├── analyser.py
│   ├── verboser.py
│   ├── example.py
│   └── local.py.template
├── models/               # AMPL model file(s) used by SupplyChain.toAMPL() / find_optimum()
├── supply_chains/        # generated supply chain JSON instances
├── networks/             # rendered network diagrams (.png / .dot)
├── results/              # perturbation analysis output (.json) and plots
├── temp/                 # scratch files used during conversion/solving
├── requirements.txt
└── README.md
```

## Known gaps / TODO

- `models/stoichiometric-lp-model.mod` (the AMPL model referenced by `SupplyChain.toAMPL()`) is not yet included in this repository and needs to be added.
- `SupplyChain.size()` is a stub and currently always returns `0`.
- The "bootstrap phase" (ramp-up of production means, assembly lines coming online) described in the paper's introduction is explicitly out of scope for the current model, which only covers steady-state production.
- Network-size reduction for the dynamic flow expansion (Section 3.4 of the paper) is not yet implemented; the reference implementation currently replicates the full static network across all time steps.
- Several `# TODO` markers exist in-code (e.g. in `analyser.py` around tracking instance size/edge count alongside results, and in `supplychain.py` around simplifying the AMPL generation).

## Citing

If you use this code, please cite the associated working paper:

> T. Hruz, M. Larcher, J. Ding, J. Weingart, M. Spada, M. Scherrer. *Simple generation, modeling and simulation of realistic very large scale supply chain instances.* Work in progress, May 2026.

## License

TBD.
