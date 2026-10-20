"""
Generation of supply chain network parameters (paper §3.5).

Each function implements one subsection of §3.5 and draws from the numpy Generator `rng` it is given,
so the converter stays reproducible for a fixed seed. Default values match the paper.
"""

import numpy as np


def in_degree(scb, node) -> int:
    """
    Computes the indegree for a backbone node.
    Returns 1 for the bottom layer (zero layer nodes have no incoming backbone edges but receive external supply).
    """
    if node[0] == 0:
        return 1
    return scb.network.in_degree(node)


def out_degree(scb, node, out_edges) -> int:
    """
    Computes the outdegree for a backbone node.
    Returns 1 for the top layer nodes (OEMs) since they export final demand.
    """
    if node[0] == len(scb.size()) - 1:
        return 1
    return len(out_edges[node])


def production_rules(rng, scb, a: float = 2.0) -> dict:
    """
    §3.5.1 Production rules (bills of materials): Zipf distribution driven by the in-degree.
    Determines the number of inputs required from each predecessor to produce one output unit.
    - `a` (float): The parameter for the Zipf distribution (default=2.0).
    """
    prod_rules = {}
    for node in scb.network.nodes:
        d_in = in_degree(scb, node)
        if d_in > 0:
            prod_rules[node] = {r: rng.zipf(a=a) for r in range(d_in)}
        else:
            prod_rules[node] = {}
    return prod_rules


def production_capacity(rng, scb, node, out_edges, sigma: float = 0.2, c_base: float = 100.0) -> float:
    """
    §3.5.2 Production capacity: log-normal distribution driven by the out-degree.
    - `sigma` (float): The standard deviation of the log-normal distribution.
    - `c_base` (float): Base capacity required per outgoing connection.
    """
    d_out = out_degree(scb, node, out_edges)
    mu = np.log(max(1, d_out) * c_base)
    return rng.lognormal(mean=mu, sigma=sigma)


def production_delay(rng, scb, node, w: float, pdelay_c: float = 0.5, pdelay_l: float = 1.0, pdelay_e: float = 10.0) -> int:
    """
    §3.5.3 Production delay: Poisson distribution driven by layer depth and assembly complexity.
    - `w` (float): Node weight.
    - `pdelay_c` (float): Assembly complexity weight.
    - `pdelay_l` (float): Layer depth penalty.
    - `pdelay_e` (float): Efficiency scaling log base.
    """
    X, n = node
    d_in = in_degree(scb, node)

    # 1. Base time
    base_time = (pdelay_l * X) + (pdelay_c * d_in) + 1

    # 2. Efficiency factor
    efficiency_factor = np.log(max(pdelay_e, w)) / np.log(pdelay_e)

    # 3. Calculate expected delay
    expected_delay = max(1.0, base_time / efficiency_factor)

    # Draw from Poisson
    return max(1, int(rng.poisson(lam=expected_delay)))


def production_cost(rng, w: float, edge_time: float, pcost_base: float = 100.0, pcost_sigma: float = 0.10) -> float:
    """
    §3.5.4 Production cost: normal distribution coupled to production time and economy of scale.
    - `w` (float): Node weight.
    - `edge_time` (float): Generated production time (delay).
    - `pcost_base` (float): Base rate per time step.
    - `pcost_sigma` (float): Cost variance.
    """
    economy_of_scale = np.log10(max(10.0, w))
    base_cost = edge_time * pcost_base
    expected_cost = base_cost / economy_of_scale
    return max(0.1, float(rng.normal(loc=expected_cost, scale=expected_cost * pcost_sigma)))


def local_storage_capacity(rng, prod_cap: float, storage_buffer: float = 4.0, storage_sigma: float = 0.20) -> float:
    """
    §3.5.5 Local input/output storage capacity: log-normal around a target buffer of production throughput.
    - `prod_cap` (float): Linked production capacity of the node.
    - `storage_buffer` (float): Target time steps of production to buffer.
    - `storage_sigma` (float): Variance to allow right-skewed large bins.
    """
    mu = np.log(max(1.0, prod_cap) * storage_buffer)
    return float(rng.lognormal(mean=mu, sigma=storage_sigma))


def warehouse_capacity(rng, w: float, w_vol_mult: float = 500.0, w_sigma: float = 0.30) -> float:
    """
    §3.5.5 Warehouse capacity: log-normal driven by the topological weight.
    - `w` (float): Node weight.
    - `w_vol_mult` (float): Multiplier per unit of topological weight.
    - `w_sigma` (float): Capacity variance.
    """
    mu = np.log(max(1.0, w) * w_vol_mult)
    return float(rng.lognormal(mean=mu, sigma=w_sigma))


def transport_time(rng, X: int, Y: int, w_X: float, w_Y: float, t_layer_penalty: float = 3.0, t_hub_base: float = 10.0) -> int:
    """
    §3.5.6 Transport delay: Poisson driven by layer distance and hub efficiency.
    - `X`, `Y` (int): Source and destination layer indices.
    - `w_X`, `w_Y` (float): Weights of source and destination nodes.
    - `t_layer_penalty` (float): Penalty multiplier for crossing layers.
    - `t_hub_base` (float): Efficiency base logarithm modifier.
    """
    layer_distance = max(1, abs(X - Y))
    hub_efficiency = np.log10(max(t_hub_base, w_X * w_Y))
    expected_transport_time = max(1.0, (layer_distance * t_layer_penalty) / hub_efficiency)
    return max(1, int(rng.poisson(lam=expected_transport_time)))


def transport_capacity(rng, src_node, src_prod_cap: float, out_edges, t_over_prov: float = 1.25, t_cap_sigma: float = 0.30) -> float:
    """
    §3.5.7 Transport capacity: log-normal around a share of the source's production capacity.
    - `src_node` (tuple): The source node generating the transport.
    - `src_prod_cap` (float): Pre-calculated production capacity of the source node.
    - `t_over_prov` (float): Factor of excess capacity to over-provision.
    - `t_cap_sigma` (float): Route variance standard deviation.
    """
    d_out = max(1, len(out_edges[src_node]))
    base_route_cap = src_prod_cap / d_out
    mu = np.log(max(1.0, base_route_cap) * t_over_prov)
    return float(rng.lognormal(mean=mu, sigma=t_cap_sigma))


def transport_cost(rng, edge_time: float, transport_cap: float, t_cost_base: float = 50.0, t_cost_sigma: float = 0.10) -> float:
    """
    §3.5.8 Transport cost: normal distribution combining distance (time) and bulk capacity discounts.
    - `edge_time` (float): Transport edge time delay.
    - `transport_cap` (float): Evaluated capacity for the transport route.
    - `t_cost_base` (float): Distance base rate multiplier.
    - `t_cost_sigma` (float): Cost variance.
    """
    base_transport_cost = np.sqrt(edge_time) * t_cost_base
    bulk_discount = np.log10(max(10.0, transport_cap))
    expected_cost = base_transport_cost / bulk_discount
    return max(0.1, float(rng.normal(loc=expected_cost, scale=expected_cost * t_cost_sigma)))
