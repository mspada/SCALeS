# How the SCALeS code works

The code turns a random graph into a realistic supply chain, then works out the cheapest way to run it over time. There are four steps, and each one feeds the next. Steps 1 to 3 correspond to §2 and §3 of the paper. The runnable scripts in `examples/01`–`04` follow the same order.

## 1. Backbone: who supplies whom (`SupplyChainBackbone`)

This step builds the skeleton of an industry as a directed graph (paper §2).

- **Vertices are producers**, arranged in layers. Layer 0 (A) is raw materials, the middle layers are component makers, and the top layer is the OEM (for example, a car maker).
- **Each vertex gets a random "size" (weight)** from a truncated power law. Most producers are small and a few are large hubs, as in real industries.
- **Edges are supplier relationships.** Two vertices connect with a probability that grows with both their weights and shrinks with the distance between their layers. Edges always point upward: raw materials go to components, components go to the final product.
- **Pruning** removes middle-layer producers that have no supplier or no customer, because a factory can't make something from nothing or produce for nobody.
- **Warehouses** are added on top. They take a product from a few producers and distribute it to many customers.

The result is only a structure, with no quantities yet. Drawings go to `networks/backbones/`.

## 2. Conversion: adding numbers (`ScbConverter`)

Each producer is expanded into small sub-nodes (paper §3.1, Figure 5):

```text
inputs (I0, I1, …)  →  production (PI → PO)  →  outputs (O0, O1, …)
  local storage          the factory itself        one per customer
```

Then every node and arc gets realistic random parameters (paper §3.5):

- **Production rule (bill of materials):** for example, "1 motor needs 1 of input A and 4 of input B."
- **Capacity:** how much a factory or transport route can handle.
- **Cost:** per unit, cheaper for big hubs (economies of scale).
- **Delay:** how many time steps production or transport takes.

The result is saved as a JSON file in `supply_chains/`. This is a complete, static supply chain instance. You can stop here if you only need test instances.

## 3. Over time and optimised (`SupplyChain`)

Real supply chains run over time, so the network is copied once per time step (t = 0, 1, …, duration−1), as in paper §3.2 and Figure 6:

- An arc with a delay of 3 connects a node at time `t` to a node at time `t+3`.
- Storage nodes connect to themselves at `t+1`, which models keeping stock.

Raw-material nodes receive **supply** and OEM nodes have **demand**. This becomes a **linear program (LP)**: find the flow on every arc that meets demand at minimum total cost, while respecting capacities and production rules. GLPK (`glpsol`) solves it.

Three slack variables keep the LP solvable even when the network can't fully meet demand:

- **BACKORDER:** demand not met in time, carried over to later steps, with a high penalty.
- **GENERATION:** goods appearing "out of nowhere" at storages, with a very high penalty. It shows where the network is starved.
- **ABSORPTION:** surplus raw material that gets thrown away.

After solving you can read the total cost and production, export tables (CSV) and constraint reports (PDF), draw the network, and list the bottlenecks that limit production. These outputs go to `networks/dynamic/`, `networks/diagnostics/` and `results/`.

## 4. Analysis: resilience (`SupplyChainAnalyser`)

This step tests how robust the supply chain is. It disturbs it many times, either by deleting random transport links or by adding random noise to costs and capacities, re-solves each time, and records how cost and production change. The results go to `results/perturbation/` as JSON and plots.

---

**In one sentence:** random graph → realistic supply chain with numbers → cheapest operation over time → how it reacts to disruptions.

For the package layout, data structures and how to extend the model, see [docs/software.md](../docs/software.md).
