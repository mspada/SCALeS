"""
Running GLPK (`glpsol`) on the stoichiometric LP model and reading its solution back into the network.
"""

import os
import subprocess
from tempfile import NamedTemporaryFile

import networkx as nx

from .. import config


def run_glpsol(data: str, model_path, dat_path: str = None) -> str:
    """
    Solves the LP `model_path` with the AMPL data `data` and returns glpsol's standard output.
    If `dat_path` is given, the data file is kept there; otherwise a temporary file in temp/ is used and removed.
    """
    binary = config.glpsol()
    if dat_path:
        out_dir = os.path.dirname(dat_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(dat_path, 'w') as f:
            f.write(data)
        return _glpsol(binary, model_path, dat_path)

    with NamedTemporaryFile(mode='w', dir=config.temp_dir(), suffix='.dat', delete=False) as temp_dat:
        temp_dat.write(data)
    try:
        return _glpsol(binary, model_path, temp_dat.name)
    finally:
        os.remove(temp_dat.name)


def _glpsol(binary: str, model_path, dat_path: str) -> str:
    command = [binary, "-m", str(model_path), "-d", str(dat_path)]
    stdout = subprocess.run(command, check=True, text=True, capture_output=True).stdout
    if "OPTIMAL LP SOLUTION FOUND" not in stdout:
        status = next((l.strip() for l in stdout.split('\n') if 'SOLUTION' in l), "no solution status reported")
        raise RuntimeError(
            f"GLPK did not find an optimal solution ({status}). Very large supply or demand values (e.g. 1e9 on both) "
            f"make the LP numerically unstable; try a demand closer to the production capacities."
        )
    return stdout


def apply_solution(network: nx.DiGraph, stdout: str, progress=None) -> None:
    """
    Reads the solution printed by the LP model into `network`. The model prints two kinds of lines:
        node lines:  <node> <t> <GENERATION> <ABSORPTION> <BACKORDER>
        arc lines:   <node> <t> <node> <t> <FLOW>
    All other solver output is ignored. `progress`, if given, is called with the fraction of lines processed.
    """
    lines = stdout.split('\n')
    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 5:
            u, tu, a, b, c = parts
            try:
                tu = int(tu)
                try:
                    generation, absorption, backorder = float(a), float(b), float(c)
                except ValueError:
                    v, tv, flow = a, int(b), float(c)
                    if network.has_edge((u, tu), (v, tv)):
                        network[(u, tu)][(v, tv)]['flow'] = flow
                else:
                    if (u, tu) in network:
                        network.nodes[(u, tu)]['generation'] = generation
                        network.nodes[(u, tu)]['absorption'] = absorption
                        network.nodes[(u, tu)]['backorder'] = backorder
            except ValueError:
                pass
        if progress:
            progress((idx + 1) / len(lines))
