# SCALeS: Large-Scale Supply Chain Generation & Simulation

Reference implementation accompanying the working paper *"Simple generation, modeling and simulation of realistic very large scale supply chain instances"* (Hruz, Larcher, Ding, Weingart, Spada, Scherrer, ETH Zurich / ZHAW, 2026).

This project generates **realistic, large-scale synthetic supply chain instances** and simulates their behaviour (production, transport, storage and cost optimisation) using dynamic network flows. It gives researchers access to non-trivial supply chain test instances, because real-world supply chain data is almost always a closely guarded company secret.

> **Status:** Work in progress / research code. APIs, parameter names, and file layout may still change.

## How it works

The pipeline has four stages. **You can stop after any stage.** Each stage writes its results into its own folder, so you can keep a backbone or a static supply chain instance without ever running the optimisation.

| Stage | What it does | Paper | Class | Default output |
| --- | --- | --- | --- | --- |
| 1. Backbone | Layered Chung-Lu random graph of producers (raw materials → … → OEM) with truncated power-law degrees, pruning and warehouses | §2 | `SupplyChainBackbone` | `networks/backbones/` |
| 2. Conversion | Expands the backbone into a static supply chain network: production rules, local storages, transport arcs, capacities, costs, delays | §3.1-3.5 | `ScbConverter` | `supply_chains/` (JSON), `networks/supply_chains/` |
| 3. Dynamic flow model | Expands the network over time and solves for the minimum-cost flow with an LP solver (GLPK) | §3 | `SupplyChain` | `networks/dynamic/`, `results/lp/`, `results/diagnostics/` |
| 4. Analysis | Perturbs the supply chain (arc deletions, cost/capacity noise) and measures cost and production | §4 | `SupplyChainAnalyser` | `results/perturbation/` |

Stages 1 and 2 need only Python. GLPK is needed from stage 3 onwards.

## Getting started

### Requirements

- Python 3.9+
- [GLPK](https://www.gnu.org/software/glpk/) (`glpsol`) for stage 3 and 4: `brew install glpk` or `apt-get install glpk-utils`
- Optional: [Graphviz](https://graphviz.org/) (`neato`) for `SupplyChain.draw_dynamic()`

### Installation

Create a virtual environment in the repository root and install the package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

`glpsol` and `neato` are found automatically if they are on your `PATH`. Otherwise, point SCALeS to them:

```bash
export SCALES_GLPSOL=/opt/glpk/bin/glpsol
export SCALES_NEATO=/usr/local/bin/neato
```

or in Python: `scales.config.set_glpsol("/opt/glpk/bin/glpsol")`.

> **macOS note:** if `import scales` fails after `pip install -e .` in a virtual environment inside `~/Documents`, macOS may have marked the `.pth` file hidden (Python then ignores it). Fix with `chflags nohidden .venv/lib/python3.*/site-packages/*.pth`, or keep the virtual environment outside synced folders.

### Minimal example

```python
from scales import SupplyChainBackbone, ScbConverter, SupplyChain

# 1. Backbone: 4 layers, pruned, reproducible
backbone = SupplyChainBackbone(size=[10, 20, 10, 1], prune=True, seed=43)
backbone.draw()                                   # networks/backbones/backbone.png

# 2. Static supply chain instance
converter = ScbConverter(backbone, seed=42)
path = converter.to_json()                        # supply_chains/supply_chain.json

# 3. Dynamic flow model, solved with GLPK
sc = SupplyChain(JSONfilename=path, duration=60, supply=1000, demand=10)
sc.find_optimum()
print(f"Total cost: {sc.cost():.2f}, total production: {sc.production():.2f}")
```

### Examples

`examples/` contains one runnable script per stage. Each script writes into the output folders of the repository:

| Script | Stages |
| --- | --- |
| `01_generate_backbone.py` | 1 |
| `02_convert_to_supply_chain.py` | 1-2, writes `supply_chains/example_mid.json` |
| `03_solve_dynamic_flow.py` | 3: solve, diagnostics, bottlenecks, changing parameters |
| `04_perturbation_analysis.py` | 4 |
| `paper_figure3_degree_distribution.py` | reproduces Figure 3 of the paper |

```bash
python examples/01_generate_backbone.py
```

## Repository layout

```text
.
├── src/scales/          the Python package (see docs/software.md)
│   ├── backbone/        stage 1
│   ├── conversion/      stage 2
│   ├── model/           stage 3, including the LP model (model/lp/)
│   ├── analysis/        stage 4 and diagnostics
│   ├── viz/             drawing helpers
│   └── config.py        solver paths and output folders
├── examples/            runnable scripts, one per stage
├── tests/               pytest suite (`pytest`)
├── docs/                software reference (architecture, data structures)
├── paper/               the working paper
├── networks/            output: drawings of backbones and supply chains
├── supply_chains/       output: static supply chain instances (JSON)
├── results/             output: LP data, diagnostics, perturbation results
└── temp/                output: scratch files used while solving
```

The output folders are part of the repository, but their contents are git-ignored. To write the outputs somewhere else, set `SCALES_OUTPUT_DIR` or call `scales.config.set_output_root(path)`. Any explicit `filename` argument is used exactly as given.

See [docs/software.md](docs/software.md) for the data structures (node naming, JSON format, time-expanded network) and for how to extend the model.

## Running the tests

```bash
pytest
```

Tests that need GLPK are skipped when `glpsol` is not available.

## Known gaps / TODO

- The "bootstrap phase" (ramp-up of production means, assembly lines coming online) described in the paper's introduction is out of scope; the model covers steady-state production only.
- Network-size reduction for the dynamic flow expansion (paper §3.6) is not yet implemented; the full static network is replicated for every time step.
- Linear costs only; concave/convex cost functions (paper §3.7) would need a MILP formulation.
- Minimum delivery contracts (lower bounds on output arcs, paper §3.1) are not part of the reference implementation.

## Citing

If you use this code, please cite the associated working paper:

> T. Hruz, M. Larcher, J. Ding, J. Weingart, M. Spada, M. Scherrer. *Simple generation, modeling and simulation of realistic very large scale supply chain instances.* Work in progress, May 2026.

See also [CITATION.cff](CITATION.cff).

## License

TBD.
