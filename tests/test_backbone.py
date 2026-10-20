import pytest

from scales import SupplyChainBackbone


def test_seed_makes_generation_reproducible():
    a = SupplyChainBackbone(size=[5, 10, 5], prune=True, seed=3)
    b = SupplyChainBackbone(size=[5, 10, 5], prune=True, seed=3)
    assert sorted(a.network.nodes) == sorted(b.network.nodes)
    assert sorted(a.network.edges) == sorted(b.network.edges)


def test_edges_never_point_to_a_lower_layer(backbone):
    for u, v in backbone.network.edges:
        assert backbone.network.nodes[u]['layer'] <= backbone.network.nodes[v]['layer']


def test_pruning_leaves_no_dangling_intermediate_vertices(backbone):
    # Algorithm 2: in-degree >= 1 except in the bottom layer, out-degree >= 1 except in the top layer
    g = backbone.network
    for node, attrs in g.nodes(data=True):
        if not attrs['bot_layer']:
            assert g.in_degree(node) > 0, node
        if not attrs['top_layer']:
            assert g.out_degree(node) > 0, node


def test_warehouses_only_in_intermediate_layers(backbone):
    # Algorithm 3: warehouses are placed in layers 1..L-2
    n_layers = len(backbone.original_size())
    warehouses = [n for n, d in backbone.network.nodes(data=True) if d['type'] == 'W']
    assert warehouses, "fixture should contain at least one warehouse"
    for node in warehouses:
        assert 1 <= backbone.network.nodes[node]['layer'] <= n_layers - 2


def test_integrity_full_keeps_every_vertex():
    backbone = SupplyChainBackbone(size=[3, 6, 3], prune=True, integrity='full', beta=6, w=0, seed=1)
    assert backbone.size() == backbone.original_size()


def test_kc_with_wrong_length_raises():
    with pytest.raises(ValueError):
        SupplyChainBackbone(size=[5, 12, 6, 2], kc=[3, 4, 2])


def test_draw_writes_to_networks_backbones(backbone, output_root):
    backbone.draw()
    assert (output_root / "networks" / "backbones" / "backbone.png").exists()
