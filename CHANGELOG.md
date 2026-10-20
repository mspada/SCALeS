# Changelog

## 0.2.0

New version: massively updated, better documented and refactored.

### Model
- Warehouses with direction-specific multipliers `betawin` / `betawout` (paper §2.3).
- Truncated power-law weights with configurable cut-offs `kc` (paper §2.1).
- One input storage per incoming backbone edge; production rules, capacities, costs and delays drawn from the
  distributions of paper §3.5 (`scales/conversion/parameters.py`).
- LP model with `BACKORDER`, `GENERATION` and `ABSORPTION` slack variables, restricted per node type
  (`scales/model/lp/supplychain-lp-model.mod`).
- Time-varying and per-node supply and demand (scalar, 1D or 2D).
- Diagnostics: LP constraint reports (PDF), flow and slack-variable tables (CSV), bottleneck detection.

### Code structure
- Installable package `scales` (`pip install -e .`) with one sub-package per pipeline stage:
  `backbone`, `conversion`, `model`, `analysis`, plus `viz`.
- Each stage writes its default output into its own folder (`networks/`, `supply_chains/`, `results/`, `temp/`),
  configurable with `scales.config.set_output_root()` or `SCALES_OUTPUT_DIR`.
- `local.py` replaced by `scales.config`: `glpsol` and `neato` are found on the PATH, or via
  `SCALES_GLPSOL` / `SCALES_NEATO`.
- `SupplyChain(...)` no longer solves on construction; call `find_optimum()` (or pass `solve=True`).
  `cost()` and `production()` still solve on demand.
- New: `SupplyChain.from_converter()`, `SupplyChain.copy()`, `SupplyChain.to_ampl_data()`.
- `find_optimum()` raises a clear error when GLPK finds no optimal solution.
- Presets (`SupplyChain.tiny_pyramid()` and others) take `supply` / `demand` and default to `demand=10`.
  The previous `demand=1e9` made the LP numerically infeasible.
- Generated AMPL data is identical between runs.
- `scikit-learn` dependency removed. `reportlab` added (LP constraint reports).
- Removed the superseded `scb.py` and `chunglu_converter.py`.
- Tests in `tests/`, runnable examples per stage in `examples/`.
