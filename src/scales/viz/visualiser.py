import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import CSS4_COLORS
from networkx.drawing.nx_pydot import write_dot
import random
import numpy as np

class NetworkVisualizer:
    @staticmethod
    def draw(
        network,
        output='png',
        filename='networks/network_diagram',
        node_size=100,
        figsize=(8, 8),
        font_size=8,
        rng=None,
        pos=None,
        pos_shift=None,
        node_label_func=None,
        node_group_func=None,
        node_layer_func=None,
        is_warehouse_func=None,
        with_labels=True
    ):
        if output == 'none':
            return
            
        if rng is None:
            rng = random.Random()
            
        # Extract layers and assign colors
        layers = set()
        for n, d in network.nodes(data=True):
            if node_layer_func:
                layers.add(node_layer_func(n, d))
            else:
                layers.add(d.get('layer', n[0] if isinstance(n, tuple) else 0))
                
        # Convert layers to a list of tuples for sorting
        layer_list = [tuple(layer) if isinstance(layer, np.ndarray) else layer for layer in layers]
        try:
            # Sort primarily by the first element, then by the second
            sorted_layers = sorted(layer_list, key=lambda x: (x[0], x[1]) if isinstance(x, tuple) and len(x) > 1 else (x[0] if isinstance(x, tuple) else x, 0))
        except TypeError:
            # Fallback for non-sortable items (though the list comprehension should handle it)
            sorted_layers = layer_list
            
        col_layers = {layer: rng.choice(list(CSS4_COLORS.keys())) for layer in sorted_layers}
        
        if output in ('png', 'svg', 'window', 'save', 'show', 'save_and_show'):
            col = []
            for n, d in network.nodes(data=True):
                is_w = is_warehouse_func(n, d) if is_warehouse_func else d.get('type') == 'W'
                if is_w:
                    col.append('red')
                else:
                    layer = node_layer_func(n, d) if node_layer_func else d.get('layer', n[0] if isinstance(n, tuple) else 0)
                    col.append(col_layers[layer])
            
            lab = {}
            for n, d in network.nodes(data=True):
                if node_label_func:
                    lab[n] = node_label_func(n, d)
                else:
                    lab[n] = str(n)
                    
            if pos is None:
                has_pos = all('pos' in d for n, d in network.nodes(data=True))
                if has_pos:
                    pos = {n: d['pos'] for n, d in network.nodes(data=True)}
                else:
                    pos = nx.multipartite_layout(G=network, subset_key="layer")
                    if pos_shift:
                        pos = {n: pos_shift(n, p) for n, p in pos.items()}
            
            plt.figure(figsize=figsize)
            nx.draw(G=network, pos=pos, node_color=col, node_size=node_size, labels=lab, with_labels=with_labels, connectionstyle='arc3, rad=0.1', font_size=font_size)
            
            if output in ('png', 'svg', 'save', 'save_and_show'):
                if output == 'svg':
                    out_name = filename if filename.endswith('.svg') else f"{filename}.svg"
                    plt.savefig(out_name, format="svg", bbox_inches='tight')
                else:
                    # Note: savefig to png overrides filename to ensure .png extension unless the caller passed exact path.
                    # Since the caller might pass .png anyway, we just enforce it for 'png' output specifically, or just save it.
                    out_name = filename if filename.endswith('.png') else f"{filename}.png"
                    plt.savefig(out_name, format="png", dpi=300, bbox_inches='tight')
            
            if output in ('window', 'show', 'save_and_show'):
                plt.show()
                
            if output in ('png', 'svg', 'save', 'save_and_show', 'none'):
                plt.close()
            
        elif output == 'dot':
            for n, d in network.nodes(data=True):
                if node_label_func:
                    d['label'] = node_label_func(n, d)
                else:
                    d['label'] = str(n)
                    
                is_w = is_warehouse_func(n, d) if is_warehouse_func else d.get('type') == 'W'
                if is_w:
                    d['fillcolor'] = 'red'
                else:
                    layer = node_layer_func(n, d) if node_layer_func else d.get('layer', n[0] if isinstance(n, tuple) else 0)
                    d['fillcolor'] = col_layers[layer]
                    
                d['style'] = "filled"
                if node_group_func:
                    d['group'] = str(node_group_func(n, d))
                else:
                    layer = node_layer_func(n, d) if node_layer_func else d.get('layer', n[0] if isinstance(n, tuple) else 0)
                    d['group'] = str(layer)
                    
            out_name = filename if filename.endswith('.dot') else f"{filename}.dot"
            write_dot(network, out_name)
