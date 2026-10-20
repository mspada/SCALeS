"""
Warehouse node injection (paper §2.3, Algorithm 3).

Warehouses are added as a fraction `w` of the already generated backbone, placed uniformly at random in the
layers 1..L-2 and connected with direction-specific multipliers betawin (incoming) and betawout (outgoing).
"""

import numpy as np
import networkx as nx


def inject_warehouses(scb) -> None:
    """Generates additional warehouse nodes in `scb.network` and connects them with direction-aware beta weights."""
    N_std = len(scb.network.nodes)
    N_wh = int(scb._w * N_std)
    if N_wh <= 0:
        return

    tau_wh = scb._tau[0]
    kc_wh = max(1.0, 1.2 * np.sqrt(N_wh))

    W_wh = scb._truncated_power_law(gamma=tau_wh, kc=kc_wh, size=N_wh)

    n_layers = len(scb._size)
    if n_layers < 3:
        return # Not enough layers to place warehouses between the lowest and highest layers

    L_sizes = scb.size()

    nodes_wh = []
    max_n = {X: -1 for X in range(n_layers)}
    for (X, n) in scb.network.nodes:
        if n > max_n[X]:
            max_n[X] = n

    for i in range(N_wh):
        l = scb._np_rng.integers(1, n_layers - 1)
        max_n[l] += 1
        n = max_n[l]
        weight = W_wh[i]
        # w_id is used for formatting the node name as W{w_id}
        nodes_wh.append( ((l, n), {'layer': l, 'weight': weight, 'top_layer': l==n_layers-1, 'bot_layer': l==0, 'type': 'W', 'w_id': i + 1}) )
        L_sizes[l] += 1

    scb.network.add_nodes_from(nodes_wh)

    edges_wh = []
    for X in range(n_layers):
        for Y in range(X, n_layers):
            nodesA = [n for n in scb.network.nodes if scb.network.nodes[n]['layer'] == X]
            nodesB = [n for n in scb.network.nodes if scb.network.nodes[n]['layer'] == Y]

            weightsA = np.array([scb.network.nodes[n]['weight'] for n in nodesA])
            weightsB = np.array([scb.network.nodes[n]['weight'] for n in nodesB])

            # Construct direction-aware beta matrix
            beta_matrix = np.zeros((len(nodesA), len(nodesB)))
            for i, u in enumerate(nodesA):
                u_is_w = scb.network.nodes[u].get('type') == 'W'
                for j, v in enumerate(nodesB):
                    v_is_w = scb.network.nodes[v].get('type') == 'W'
                    if u_is_w and not v_is_w:
                        # Warehouse to standard node (Outgoing)
                        beta_matrix[i, j] = scb._betawout
                    elif not u_is_w and v_is_w:
                        # Standard node to Warehouse (Incoming)
                        beta_matrix[i, j] = scb._betawin
                    elif u_is_w and v_is_w:
                        # Warehouse to Warehouse (Geometric mean fallback)
                        beta_matrix[i, j] = np.sqrt(scb._betawin * scb._betawout)
                    else:
                        # Standard to Standard
                        beta_matrix[i, j] = scb._beta[(X, Y)] if X != Y else scb._beta[(X, X)]

            if X != Y:
                denom = L_sizes[X] + L_sizes[Y]
                proba = scb._alpha[(X,Y)] * np.minimum(beta_matrix * np.outer(weightsA, weightsB) / denom, 1) if denom > 0 else np.zeros((len(weightsA), len(weightsB)))
            else:
                denom = 2 * L_sizes[X]
                proba = scb._alpha[(X,X)] * np.minimum(beta_matrix * np.outer(weightsA, weightsB) / denom, 1) if denom > 0 else np.zeros((len(weightsA), len(weightsB)))

            adj = scb._np_rng.binomial(n=1, p=proba, size=proba.shape)
            if X == Y:
                np.fill_diagonal(adj, 0)

            for i, u in enumerate(nodesA):
                for j, v in enumerate(nodesB):
                    if adj[i, j] == 1:
                        if scb.network.nodes[u].get('type') == 'W' or scb.network.nodes[v].get('type') == 'W':
                            edges_wh.append((u, v))

    scb.network.add_edges_from(edges_wh)


def prune_warehouses(network: nx.DiGraph, verbose: bool = False) -> None:
    """Recursively removes any warehouse node with an in-degree or out-degree of 0."""
    start_num_nodes, start_num_edges = len(network.nodes), len(network.edges)
    while True:
        nodes_to_remove = [
            node for node in network.nodes
            if network.nodes[node].get('type') == 'W' and
               (len(list(network.predecessors(node))) == 0 or len(list(network.successors(node))) == 0)
        ]
        if not nodes_to_remove:
            break
        network.remove_nodes_from(nodes_to_remove)

    end_num_nodes, end_num_edges = len(network.nodes), len(network.edges)
    if verbose:
        print(f'Pruned warehouses.\nKept {100*end_num_nodes/start_num_nodes:.1f}% of the nodes and {100*end_num_edges/start_num_edges:.1f}% of the edges.')
