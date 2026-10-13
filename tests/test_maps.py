import numpy as np
import pytest
from helpers import H, W, open_map_text

from inkarena.sim import MapError, list_maps, load_map, parse_map, validate_map


def test_bundled_maps_load_and_validate():
    names = list_maps()
    assert len(names) >= 2
    for name in names:
        m = load_map(name)
        assert (m.width, m.height) == (32, 20)
        assert m.walls[0].all() and m.walls[-1].all()
        assert m.walls[:, 0].all() and m.walls[:, -1].all()


def test_spawns_are_rotated_for_team_b():
    m = load_map("duel")
    for (ax, ay), (bx, by) in zip(m.spawns[0], m.spawns[1], strict=True):
        assert (bx, by) == (m.width - ax, m.height - ay)
        assert m.base[int(ay), int(ax)] == 1
        assert m.base[int(by), int(bx)] == -1


def test_base_and_scorable_cells():
    m = parse_map(open_map_text())
    assert (m.base == 1).sum() == (m.base == -1).sum() == 8
    assert not m.scorable[m.walls].any()
    assert not m.scorable[m.base != 0].any()


def test_asymmetric_walls_rejected():
    rows = open_map_text().splitlines()
    rows[5] = rows[5][:5] + "#" + rows[5][6:]
    with pytest.raises(MapError, match="symmetric"):
        parse_map("\n".join(rows))


def test_asymmetric_base_rejected():
    rows = open_map_text().splitlines()
    rows[1] = rows[1].replace("b", ".", 1)
    with pytest.raises(MapError, match="symmetric"):
        parse_map("\n".join(rows))


def test_unreachable_cells_rejected():
    # Wall off a pocket around (10, 7) and its mirror image.
    pocket = [(9, 6), (10, 6), (11, 6), (9, 7), (11, 7), (9, 8), (10, 8), (11, 8)]
    with pytest.raises(MapError, match="unreachable"):
        parse_map(open_map_text(pocket))


def test_spawn_must_be_unique():
    text = open_map_text().replace("a", "1", 1)
    with pytest.raises(MapError, match="spawn"):
        parse_map(text)


def test_wrong_size_rejected():
    m = parse_map(open_map_text())
    with pytest.raises(MapError, match="size"):
        validate_map(m, size=(32, 20))
    assert m.walls.shape == (H, W)


def test_unknown_symbol_rejected():
    with pytest.raises(MapError, match="unknown"):
        parse_map(open_map_text().replace(".", "?", 1))


def test_bundled_maps_have_open_center():
    for name in list_maps():
        m = load_map(name)
        cx, cy = int(m.center[0]), int(m.center[1])
        assert not m.walls[cy - 1 : cy + 1, cx - 1 : cx + 1].any()
        assert np.array_equal(m.walls, m.walls[::-1, ::-1])
