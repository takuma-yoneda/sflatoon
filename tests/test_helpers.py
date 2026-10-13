import numpy as np

from inkarena.bots import helpers as h
from inkarena.sim import load_map


def walls_with_bar():
    w = np.zeros((7, 9), dtype=bool)
    w[0, :] = w[-1, :] = w[:, 0] = w[:, -1] = True
    w[1:5, 4] = True  # vertical bar, open at the bottom (y=5)
    return w


def test_path_goes_around_walls():
    w = walls_with_bar()
    p = h.path(w, (2.5, 2.5), (6.5, 2.5))
    assert p[-1] == (6.5, 2.5)
    assert (4.5, 5.5) in p  # through the gap
    assert all(not w[int(y), int(x)] for x, y in p)
    steps = [(2.5, 2.5)] + p
    assert all(
        abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(steps, steps[1:], strict=False)
    )
    assert len(p) == 2 + 3 + 3 + 3 + 1 - 2  # down 3, across 4, up 3


def test_path_unreachable_and_same_cell():
    w = walls_with_bar()
    w[5, 4] = True
    assert h.path(w, (2.5, 2.5), (6.5, 2.5)) == []
    assert h.path(w, (2.2, 2.7), (2.9, 2.1)) == []


def test_los():
    w = walls_with_bar()
    assert not h.los(w, (2.5, 2.5), (6.5, 2.5))
    assert h.los(w, (2.5, 5.5), (6.5, 5.5))


def test_nearest_walking_vs_straight_line():
    w = walls_with_bar()
    mask = np.zeros_like(w)
    mask[2, 5] = True  # just across the bar
    mask[5, 1] = True  # farther in a straight line, closer on foot? no: compare
    assert h.nearest(mask, (3.5, 2.5)) == (5, 2)
    assert h.nearest(mask, (3.5, 2.5), w) == (1, 5)
    assert h.nearest(np.zeros_like(w), (3.5, 2.5)) is None


def test_frontier():
    ink = np.zeros((5, 5), dtype=np.int8)
    ink[2, 2] = 1
    ink[2, 3] = -1
    f = h.frontier(ink)
    assert f.sum() == 4 and f[2, 3] and f[1, 2] and not f[2, 2]


def test_regions():
    mask = np.zeros((5, 6), dtype=bool)
    mask[0:2, 0:2] = True
    mask[3:5, 3:6] = True
    mask[0, 5] = True
    labels, n = h.regions(mask)
    assert n == 3
    assert len({labels[0, 0], labels[4, 5], labels[0, 5]}) == 3
    assert (labels[~mask] == 0).all()


def test_distance_field_on_real_map():
    m = load_map("duel")
    d = h.distance_field(m.walls, [(15, 18)])
    assert d[18, 15] == 0
    assert (d[~m.walls] < h.UNREACHABLE).all()
    assert (d[m.walls] == h.UNREACHABLE).all()
