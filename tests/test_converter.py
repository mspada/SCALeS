import json

from scales import ScbConverter


def test_to_json_defaults_to_supply_chains_folder(converter, output_root):
    path = converter.to_json()
    assert path == str(output_root / "supply_chains" / "supply_chain.json")
    data = json.loads(open(path).read())
    assert set(data) >= {"nodes", "edges"}


def test_same_seed_gives_identical_json(backbone):
    assert ScbConverter(backbone, seed=5).to_json_str() == ScbConverter(backbone, seed=5).to_json_str()


def test_production_rules_have_one_entry_per_input_storage(converter):
    # §3.1: every input storage I_k of a production node carries a positive integer coefficient
    nodes = json.loads(converter.to_json_str())["nodes"]
    for name, node in nodes.items():
        if node["type"] != "PRODUCTION_INPUT":
            continue
        label = name[:name.index("(")]
        inputs = {n for n in nodes if n.startswith(label + "(I")}
        assert set(node["rule"]) == inputs
        assert all(isinstance(k, int) and k >= 1 for k in node["rule"].values())


def test_boundary_layers_get_material_and_oem_nodes(converter, backbone):
    nodes = json.loads(converter.to_json_str())["nodes"]
    types = [node["type"] for node in nodes.values()]
    assert "MATERIAL" in types and "OEM" in types
    top = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[len(backbone.original_size()) - 1]
    assert all(name.startswith("A") for name, n in nodes.items() if n["type"] == "MATERIAL")
    assert all(name.startswith(top) for name, n in nodes.items() if n["type"] == "OEM")


def test_edge_parameters_are_positive(converter):
    edges = json.loads(converter.to_json_str())["edges"]
    for e in edges:
        assert e["cap"] > 0
        assert e["cost"] >= 0
        assert e["time"] >= 0
