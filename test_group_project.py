from app1 import build_lists
from Feature_01 import return_even
from Feature_02 import return_odd


def test_return_even():
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert return_even(values) == [0, 2, 4, 6, 8]


def test_return_odd():
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert return_odd(values) == [1, 3, 5, 7, 9]


def test_build_lists():
    even, odd = build_lists(list(range(10)))
    assert even == [0, 2, 4, 6, 8]
    assert odd == [1, 3, 5, 7, 9]
