import networkx as nx

def find_bottlenecks(sc, top_n=15, supply_limit=None):
    """
    Systematically analyzes the static supply chain network to find absolute 
    stoichiometric bottlenecks. Returns the top bottlenecks limiting OEM production.
    """
    base_net = sc._baseNetwork
    
    try:
        topo = list(nx.topological_sort(base_net))
    except nx.NetworkXUnfeasible:
        print("Network contains cycles. Cannot compute stoichiometric multipliers.")
        return []

    mult = {n: 0.0 for n in base_net.nodes}
    
    # Identify OEM nodes
    oem_nodes = [n for n, d in base_net.nodes(data=True) if d.get('type') == 'OEM']
    if not oem_nodes:
        oem_nodes = [n for n in base_net.nodes if base_net.out_degree(n) == 0]
        
    for n in oem_nodes:
        mult[n] = 1.0

    edge_mults = {}

    for u in reversed(topo):
        rule = base_net.nodes[u].get('rule', {})
        is_pi = base_net.nodes[u].get('type') == 'PRODUCTION_INPUT'
        
        for p in base_net.predecessors(u):
            ratio = float(rule.get(p, 1.0)) if is_pi else 1.0
            e_mult = mult[u] * ratio
            edge_mults[(p, u)] = e_mult
            mult[p] += e_mult

    bottlenecks = []
    
    # 1. Check physical edges
    for (p, u), e_mult in edge_mults.items():
        if e_mult > 0:
            cap = float(base_net[p][u].get('cap', float('inf')))
            if cap < 1e8:
                bottlenecks.append({
                    'type': 'Edge',
                    'name': f"{p} -> {u}",
                    'cap': cap,
                    'multiplier': e_mult,
                    'implied_max_oem': cap / e_mult
                })

    # 2. Check material supply if provided
    if supply_limit is None and hasattr(sc, '_network'):
        for n, d in sc._network.nodes(data=True):
            if d.get('type') == 'MATERIAL' and 'supply' in d:
                supply_limit = d['supply']
                break

    if supply_limit is not None:
        for n, d in base_net.nodes(data=True):
            if d.get('type') == 'MATERIAL' and mult[n] > 0:
                bottlenecks.append({
                    'type': 'Material Supply',
                    'name': n,
                    'cap': float(supply_limit),
                    'multiplier': mult[n],
                    'implied_max_oem': float(supply_limit) / mult[n]
                })

    bottlenecks.sort(key=lambda x: x['implied_max_oem'])
    return bottlenecks[:top_n]

def print_bottlenecks(sc, top_n=15, supply_limit=None):
    print("\n=== SYSTEMATIC BOTTLENECK ANALYSIS ===")
    bottlenecks = find_bottlenecks(sc, top_n, supply_limit)
    if not bottlenecks:
        print("No bottlenecks found.")
        return

    print(f"{'Rank':<5} | {'Type':<17} | {'Component/Edge':<25} | {'Capacity':<10} | {'Multiplier':<10} | {'Max OEM Prod':<12}")
    print("-" * 87)
    for i, b in enumerate(bottlenecks):
        print(f"{i+1:<5} | {b['type']:<17} | {b['name']:<25} | {b['cap']:<10.2f} | {b['multiplier']:<10.2f} | {b['implied_max_oem']:<12.3f}")
    print("-" * 87)
    print()
