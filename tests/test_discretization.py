from ..discretization import Discretization


def test_discretization():
    assert (Discretization(dataset_file_name="dataset.csv").execute()) == 10.0


def test_discretization2():
    assert (Discretization(dataset_file_name="dataset2.csv").execute()) == 5.5
