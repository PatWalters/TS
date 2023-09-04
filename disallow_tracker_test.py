import numpy as np
import pytest

from disallow_tracker import DisallowTracker

To_Fill = DisallowTracker.To_Fill
Empty = DisallowTracker.Empty

def test_disallow_tracker_complete():
    sizes = [5, 8, 9]
    total = np.prod(sizes)
    d_tracker = DisallowTracker(sizes)
    s = set()
    for i in range(total):
        res = d_tracker.sample()
        s.add(tuple(res))
    assert len(s) == total

def test_disallow_throws_when_full():
    sizes = [5, 8, 9]
    total = np.prod(sizes)
    d_tracker = DisallowTracker(sizes)
    s = set()
    for i in range(total):
        res = d_tracker.sample()
        s.add(tuple(res))
    assert len(s) == total
    with pytest.raises(ValueError):
        d_tracker.sample()

def test_disallow_simple():
    sizes = [3, 4, 5]

    d_tracker = DisallowTracker(sizes)
    assert d_tracker.get_disallowed_selection_mask([To_Fill, 2, 3]) == set()
    assert d_tracker.get_disallowed_selection_mask([1, To_Fill, 3]) == set()
    assert d_tracker.get_disallowed_selection_mask([1, 2, To_Fill]) == set()

    d_tracker.update([1, 2, 3])

    assert d_tracker.get_disallowed_selection_mask([To_Fill, 2, 3]) == set([1])
    assert d_tracker.get_disallowed_selection_mask([1, To_Fill, 3]) == set([2])
    assert d_tracker.get_disallowed_selection_mask([1, 2, To_Fill]) == set([3])

    d_tracker.update([0, 2, 3])

    assert d_tracker.get_disallowed_selection_mask([To_Fill, 2, 3]) == set([0, 1])
    assert d_tracker.get_disallowed_selection_mask([1, To_Fill, 3]) == set([2])
    assert d_tracker.get_disallowed_selection_mask([1, 2, To_Fill]) == set([3])
    assert d_tracker.get_disallowed_selection_mask([0, To_Fill, 3]) == set([2])
    assert d_tracker.get_disallowed_selection_mask([0, 2, To_Fill]) == set([3])

def test_disallow_reagent_exhausted():
    sizes = [3, 4, 5]

    d_tracker = DisallowTracker(sizes)

    # This will fully exhaust reagent position 0 for the [To_Fill, 1, 1] case
    d_tracker.update([0, 1, 1])
    d_tracker.update([1, 1, 1])
    d_tracker.update([2, 1, 1])

    # The important tests that we propogated to the Empty with To_Fill cases for reagent 0
    assert d_tracker.get_disallowed_selection_mask([Empty, To_Fill, 1]) == set([1])
    assert d_tracker.get_disallowed_selection_mask([Empty, 1, To_Fill]) == set([1])
    # Shouldn't really get here in practice (because of the above check), but good to double check.
    assert d_tracker.get_disallowed_selection_mask([To_Fill, 1, 1]) == set([0, 1, 2])

    # If we select the 0th reagent first, this is just the regular exclusion
    for reagent_0 in range(3):
        assert d_tracker.get_disallowed_selection_mask([reagent_0, 1, To_Fill]) == set([1])
        assert d_tracker.get_disallowed_selection_mask([reagent_0, To_Fill, 1]) == set([1])
        # Nothing propagated to the other cases
        assert d_tracker.get_disallowed_selection_mask([reagent_0, Empty, To_Fill]) == set([])
        assert d_tracker.get_disallowed_selection_mask([reagent_0, To_Fill, Empty]) == set([])
