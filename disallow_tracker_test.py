import numpy as np

from disallow_tracker import DisallowTracker


def test_disallow_tracker():
    sizes = [5, 8, 9]
    total = np.prod(sizes)
    d_tracker = DisallowTracker(sizes)
    s = set()
    for i in range(total):
        res = d_tracker.sample()
        s.add(tuple(res))
    assert len(s) == total

def test_disallow_throw():
    sizes = [5, 8, 9]
    total = np.prod(sizes)
    d_tracker = DisallowTracker(sizes)
    s = set()
    try:
        for i in range(total+1):
            res = d_tracker.sample()
            s.add(tuple(res))
    except ValueError:
        pass
    assert len(s) == total