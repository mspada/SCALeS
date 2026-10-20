"""
Reproduces Figure 3 of the paper: the overall in-, out- and total degree distribution of a layered SCB graph
with ~5000 vertices, averaged over many generated instances.

Output: networks/total-deg-dist.png
Usage:  python examples/paper_figure3_degree_distribution.py [repetitions]   (the paper uses 100)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scales import SupplyChainBackbone, config

config.set_output_root(Path(__file__).resolve().parents[1])

repetitions = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def degree_histogram(degrees) -> np.ndarray:
    return np.bincount(np.fromiter(degrees, dtype=int))


def add(total: np.ndarray, hist: np.ndarray) -> np.ndarray:
    if len(hist) > len(total):
        total = np.pad(total, (0, len(hist) - len(total)))
    total[:len(hist)] += hist
    return total


p_in, p_out, p_tot = np.zeros(1), np.zeros(1), np.zeros(1)
for i in range(repetitions):
    g = SupplyChainBackbone(size=5000, prune=True, verbose=True, integrity='partial', vratio=0.4).network
    p_in = add(p_in, degree_histogram(d for _, d in g.in_degree()))
    p_out = add(p_out, degree_histogram(d for _, d in g.out_degree()))
    p_tot = add(p_tot, degree_histogram(d for _, d in g.degree()))

# averages over the instances; + 1 avoids log(0) and has no influence on the shape
plt.figure(figsize=(8, 6))
for hist, label, color, style in [(p_in, 'Input degree distribution', 'blue', '--'),
                                  (p_out, 'Output degree distribution', 'red', '--'),
                                  (p_tot, 'Total degree distribution', 'green', '-')]:
    x = np.log(np.arange(len(hist)) + 1)
    plt.plot(x, np.log(hist / repetitions + 1), label=label, color=color, linewidth=1, linestyle=style, marker='o')
plt.title('Multi-partition')
plt.xlabel('log k')
plt.ylabel('log P(k)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
out = config.networks_dir() / "total-deg-dist.png"
plt.savefig(out, format='png', dpi=300, bbox_inches='tight')
print("Saved", out)
