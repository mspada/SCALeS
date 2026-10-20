"""
Drawing methods of SupplyChain (static base network, dynamic time-expanded network, node diagnostics).
"""

import os
import random
import subprocess
import sys
from collections import defaultdict, deque
from itertools import product

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .. import config
from .._utils import natural_sort_key


class DrawingMixin:
    """Methods of SupplyChain that render `self._baseNetwork` and `self._network`."""

    def draw_dynamic(self, show=True, save=True, filename=None, filetype='pdf', expanding_factor=6):
        """
        Renders the time-expanded network with Graphviz `neato`. Default filename is networks/dynamic/network.
        - `save` (bool): keeps the intermediate .dot file when True.
        - `show` (bool): opens the rendered file once created.
        """
        if filename is None:
            filename = str(config.networks_dir("dynamic") / "network")
        self._ensure_positions()
        temp_network = self._network.copy()
        for node in temp_network.nodes:
            if 'rule' in temp_network.nodes[node]:
                del temp_network.nodes[node]['rule']
            if 'supp_pred' in temp_network.nodes[node]:
                del temp_network.nodes[node]['supp_pred']
            if 'prod_pred' in temp_network.nodes[node]:
                del temp_network.nodes[node]['prod_pred']

        for (node, t) in self._network.nodes:
            x, y = self._network.nodes[(node, t)]['pos']
            temp_network.nodes[(node, t)]['pos'] = f"{expanding_factor*x:.3f}, {expanding_factor*y:.3f}!"
            temp_network.nodes[(node, t)]['label'] = node + str(t)

        for x in temp_network.nodes:
            if 'generation' in temp_network.nodes[x] and temp_network.nodes[x]['generation'] > 0:
                temp_network.nodes[x]['color']='red'
                temp_network.nodes[x]['fontcolor'] = 'red'
                temp_network.nodes[x]['fontname'] = 'Arial Bold'
                temp_network.nodes[x]['penwidth'] = 3

        for x, y in temp_network.edges:
            temp_network[x][y]['decorate'] = 'true'
            if self._optimum_found and 'flow' in temp_network[x][y]:
                flow_val = temp_network[x][y]['flow']
                temp_network[x][y]['label'] = f'<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0"><TR><TD>f:{flow_val:.2f}</TD></TR></TABLE>>'
                if flow_val > 0:
                    temp_network[x][y]['penwidth'] = 3
                    temp_network[x][y]['color'] = 'blue'
                    temp_network[x][y]['fontcolor'] = 'blue'
                    temp_network[x][y]['fontname'] = 'Arial Bold'
            else:
                temp_network[x][y]['label'] = '<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0"><TR><TD>f:na</TD></TR></TABLE>>'

        dot_file = f"{filename}.dot"
        out_file = f"{filename}.{filetype}"

        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        nx.drawing.nx_pydot.write_dot(temp_network, dot_file)

        command = [config.neato(), "-T", filetype, dot_file, "-o", out_file]
        subprocess.run(command, check=True, text=True, capture_output=True)

        if not save:
            os.remove(dot_file)

        if show:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.run([opener, out_file], check=True, text=True, capture_output=True)

    def draw_static(self, filename=None, output=None):
        """
        Uses `NetworkVisualizer` to draw the static baseline topology.
        - `filename` (str): Output filename. Default is networks/supply_chains/base_network.
        - `output` (str or None): Output format ('png', 'pdf', 'svg', etc.). Defaults to self._output.
        """
        from ..conversion import ScbConverter
        if filename is None:
            filename = str(config.networks_dir("supply_chains") / "base_network")
        self._ensure_positions()
        out = output if output is not None else self._output
        ScbConverter.draw_static_graph(
            supply_chain=self._baseNetwork,
            output=out,
            filename=filename,
            rng=random.Random(42)  # Use a fixed or simple seeded RNG if needed
        )

    def draw_node_diagnostic(self, label: str, t: int, filename=None, output=None):
        """
        Draws the dynamic flow network environment around the given backbone node
        at time t.
        - `label` (str): Label of the node.
        - `t` (int): Time step.
        - `filename` (str or None): Output filename. Default is networks/diagnostics/dynamic/diagnostic_<label>_t<t>.
        - `output` (str or None): Output format. Defaults to self._output.
        """
        subgraph_nodes, _, _ = self._node_time_neighbourhood(label, t)
        G = self._network.subgraph(subgraph_nodes)

        # Slightly larger canvas for high density
        plt.figure(figsize=(20, 12))

        # 1. Group nodes by their logical time and structural band
        nodes_by_time_type = defaultdict(list)
        for node in G.nodes():
            n_str, n_t = node
            paren_idx = n_str.find('(')
            node_lbl = n_str[:paren_idx] if paren_idx != -1 else n_str
            subtype = n_str[paren_idx:] if paren_idx != -1 else ""

            # Determine base Y band
            if node_lbl == label:
                if subtype.startswith('(I'): base_y = 3
                elif subtype == '(PI)': base_y = 1
                elif subtype == '(PO)': base_y = -1
                elif subtype.startswith('(O'): base_y = -3
                elif subtype == '(W)': base_y = 1
                else: base_y = 1
            else:
                # Neighboring nodes (push them further out dynamically)
                if subtype.startswith('(O') or subtype == '(PO)': base_y = 5
                elif subtype.startswith('(I') or subtype == '(PI)': base_y = -5
                elif subtype == '(W)':
                    base_y = 5 if node[1] <= t else -5
                else: base_y = 6

            nodes_by_time_type[(n_t, base_y)].append(node)

        # 2. Assign positions with deterministic vertical spread
        pos = {}
        for (n_t, base_y), group in nodes_by_time_type.items():
            # Sort naturally for consistent sequential placement (e.g., I0, I1, I2, I10)
            group.sort(key=lambda x: natural_sort_key(x[0]))
            spread = len(group)

            # Center the group vertically around the base_y band
            y_start = base_y + (spread - 1) * 0.3
            for idx, node in enumerate(group):
                # Scale X by 2 to stretch the timeline horizontally
                pos[node] = (n_t * 2, y_start - idx * 0.6)

        node_labels = {}
        for node in G.nodes():
            n_str, n_t = node
            lbl = f"{n_str}\nt={n_t}"
            attrs = []
            if 'storage_cap' in G.nodes[node] and G.nodes[node]['storage_cap'] != float('inf'):
                attrs.append(f"c:{G.nodes[node]['storage_cap']:.0f}")
            if 'generation' in G.nodes[node] and G.nodes[node]['generation'] > 0:
                attrs.append(f"g:{G.nodes[node]['generation']:.0f}")
            if 'absorption' in G.nodes[node] and G.nodes[node]['absorption'] > 0:
                attrs.append(f"a:{G.nodes[node]['absorption']:.0f}")
            if 'backorder' in G.nodes[node] and G.nodes[node]['backorder'] > 0:
                attrs.append(f"b:{G.nodes[node]['backorder']:.0f}")

            if attrs:
                lbl += "\n" + "\n".join(attrs)
            node_labels[node] = lbl

        edge_labels = {}
        for u, v, d in G.edges(data=True):
            if self._optimum_found and 'flow' in d:
                edge_labels[(u, v)] = f"f:{d['flow']:.1f}"
            else:
                edge_labels[(u, v)] = "f:na"

        node_colors = []
        for node in G.nodes():
            n_str, n_t = node
            paren_idx = n_str.find('(')
            node_lbl = n_str[:paren_idx] if paren_idx != -1 else n_str
            subtype = n_str[paren_idx:] if paren_idx != -1 else ""

            if node_lbl == label:
                if subtype.startswith('(I'): node_colors.append('lightblue')
                elif subtype == '(PI)': node_colors.append('orange')
                elif subtype == '(PO)': node_colors.append('coral')
                elif subtype.startswith('(O'): node_colors.append('lightgreen')
                elif subtype == '(W)': node_colors.append('red')
                else: node_colors.append('grey')
            else:
                if subtype == '(W)': node_colors.append('lightcoral')
                else: node_colors.append('lightgrey')

        # Draw with drastically reduced sizes and added curvature
        nx.draw(G, pos, with_labels=True, labels=node_labels, node_color=node_colors,
                node_size=400, font_size=5, arrows=True, arrowsize=8,
                connectionstyle='arc3,rad=0.15')

        # Push edge labels slightly off-center (label_pos=0.3) to prevent overlapping with crossing edges
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=4, label_pos=0.3)

        plt.title(f"Dynamic Flow Diagnostic view for {label} at t={t}")

        out = output if output is not None else self._output
        if out in ('png', 'svg', 'save', 'save_and_show'):
            if not filename:
                filename = str(config.networks_dir("diagnostics", "dynamic") / f"diagnostic_{label}_t{t}")
            out_dir = os.path.dirname(filename)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            if out == 'svg':
                plt.savefig(filename if filename.endswith('.svg') else f"{filename}.svg", format="svg", bbox_inches='tight')
            else:
                plt.savefig(filename if filename.endswith('.png') else f"{filename}.png", dpi=300, bbox_inches='tight')

        if out in ('window', 'show', 'save_and_show'):
            plt.show()

        if out in ('png', 'save', 'none'):
            plt.close()

    def _ensure_positions(self):
        # positions are only needed for drawing, so they are computed on first use
        if not all('pos' in d for _, d in self._baseNetwork.nodes(data=True)):
            self._find_positions()

    def _find_positions(self):
        pos_base = nx.multipartite_layout(G=self._baseNetwork, subset_key="layer")
        pos_base = { node: (x +  self._baseNetwork.nodes[node]['layer'][0] /4,y) for node, (x,y) in pos_base.items() }
        _, y_pos_base = zip(*pos_base.values())
        ymin, ymax = min(y_pos_base), max(y_pos_base)
        for node in self._baseNetwork.nodes:
            x, y = pos_base[node]
            self._baseNetwork.nodes[node]['pos'] = x, y
            for t in range(self.duration):
                self._network.nodes[(node, t)]['pos'] = x+t/2, y - t * (ymax - ymin + 1/2)

    def _drawBaseNetwork(self, layout=None):
        # a method that quickly draws the internal representation of the base network (without time)
        # should mainly be used for debugging
        if layout:
            pos = layout(self._baseNetwork)
        else:
            pos = self._customDownTopLayoutBaseGraph()
        nx.draw(self._baseNetwork, pos=pos, with_labels=True)
        plt.show()

    def _drawNetwork(self, baseLayout=None):
        # a method that quickly draws the internal representation of the network
        # should mainly be used for debugging
        if baseLayout:
            basePos = baseLayout(self._baseNetwork)
        else:
            basePos = self._customDownTopLayoutBaseGraph()

        xcoord, _ = zip(*basePos.values())
        xmin, xmax = min(xcoord), max(xcoord)
        deltax = (xmax - xmin)

        pos = {}

        for n, t in product(self.nodes, range(self.duration)):
            x, y = basePos[n]
            pos[(n,t)] = ( x + (2*t - self.duration + 1) * deltax , y )

        nx.draw(self._network, pos=pos, with_labels=True)
        plt.show()

    def _customDownTopLayoutBaseGraph(self, dx=1, dy=1/3):
        # should only be used if it is absolutely certain that the graph is acyclic
        positions = {}

        lay = deque([])
        for node, deg in self._baseNetwork.out_degree:
            if deg == 0:
                lay.append(node)

        layers = []
        layer_of = {}
        cur = 0

        while lay:
            layers.append(set())
            width = len(lay)
            for _ in range(width):
                node = lay.popleft()
                if node in layer_of:
                    layers[layer_of[node]].remove(node)
                layers[cur].add(node)
                layer_of[node] = cur
                for prev in self._baseNetwork.predecessors(node):
                    lay.append(prev)
            cur += 1

        x, y = 0, 0
        for layer in layers:
            x = 0
            for node in layer:
                positions[node] = np.array([x,y])
                x,y = x-dx, y+dy
            y = y+3*dy

        return positions
