import networkx as nx
import pytest

from scales import SupplyChain, config
from scales._utils import natural_sort_key
from scales.model.solver import apply_solution


def test_building_does_not_solve(supply_chain):
    assert supply_chain._optimum_found is False


def test_time_expansion_replicates_every_node(supply_chain):
    assert len(supply_chain._network.nodes) == len(supply_chain.nodes) * supply_chain.duration


def test_edges_respect_their_delay(supply_chain):
    for (u, tu), (v, tv) in supply_chain._network.edges:
        if u == v:
            assert tv == tu + 1  # storage carried over to the next step
        else:
            assert tv - tu == supply_chain._baseNetwork[u][v]['time']


def test_supply_and_demand_configurations(converter):
    # 1D: per time step, padded with the last value; 2D: rows are OEM nodes in natural order, columns time steps
    sc = SupplyChain.from_converter(converter, duration=5, supply=[1, 2, 3], demand=[[4, 5], [6, 7]])
    materials = [n for n, d in sc.nodes.items() if d['type'] == 'MATERIAL']
    assert [sc._network.nodes[(materials[0], t)]['supply'] for t in range(5)] == [1, 2, 3, 3, 3]
    oems = sorted((n for n, d in sc.nodes.items() if d['type'] == 'OEM'), key=natural_sort_key)
    assert [sc._network.nodes[(oems[0], t)]['demand'] for t in range(3)] == [4, 5, 5]
    assert [sc._network.nodes[(oems[-1], t)]['demand'] for t in range(3)] == [6, 7, 7]


def test_ampl_data_is_deterministic(supply_chain):
    assert supply_chain.to_ampl_data() == supply_chain.copy().to_ampl_data()


def test_set_transport_rebuilds_network(supply_chain):
    transport = next(e for e in supply_chain.edges if '(O' in e['from'] and '(I' in e['to'])
    supply_chain.set_transport(False, transport['from'], transport['to'], 2, 1.0, 123.0)
    assert supply_chain.get_transport(False, transport['from'], transport['to']) == (2, 1.0, 123.0)
    assert supply_chain._network[(transport['from'], 0)][(transport['to'], 2)]['cap'] == 123.0


def test_apply_solution_reads_nodes_and_arcs():
    g = nx.DiGraph()
    g.add_edge(('A0(O0)', 0), ('B0(I0)', 2))
    stdout = "GLPSOL log line\n    A0(O0)    0 0.000000 1.500000 0.000000\n    A0(O0)    0     B0(I0)    2 7.250000\n"
    apply_solution(g, stdout)
    assert g[('A0(O0)', 0)][('B0(I0)', 2)]['flow'] == 7.25
    assert g.nodes[('A0(O0)', 0)]['absorption'] == 1.5


def test_missing_glpsol_gives_clear_error(monkeypatch):
    monkeypatch.setenv("SCALES_GLPSOL", "/nonexistent/glpsol")
    monkeypatch.setenv("PATH", "")
    with pytest.raises(RuntimeError, match="SCALES_GLPSOL"):
        config.glpsol()


@pytest.mark.skipif(not config.has_glpsol(), reason="glpsol not installed")
def test_solve_end_to_end(supply_chain, output_root):
    supply_chain._save_temp = True
    supply_chain.find_optimum()
    assert supply_chain._optimum_found
    assert supply_chain.production() >= 0
    assert supply_chain.cost() >= 0
    assert (output_root / "results" / "lp" / "model.dat").exists()


def test_non_optimal_lp_raises_clear_error(supply_chain, tmp_path, monkeypatch):
    fake_glpsol = tmp_path / "glpsol"
    fake_glpsol.write_text("#!/bin/sh\necho 'LP HAS NO PRIMAL FEASIBLE SOLUTION'\n")
    fake_glpsol.chmod(0o755)
    monkeypatch.setattr(config, "_glpsol", str(fake_glpsol))
    with pytest.raises(RuntimeError, match="NO PRIMAL FEASIBLE SOLUTION"):
        supply_chain.find_optimum()
    assert not list((tmp_path / "temp").iterdir())  # temporary .dat removed even on failure
