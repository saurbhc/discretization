from ..discretization import Discretization


def test_discretization():
    assert (Discretization().execute()) == 5.5
