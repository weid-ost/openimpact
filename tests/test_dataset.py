import pytest
from torch_geometric.data import InMemoryDataset


def test_dataset_initialization(kelmarsh_dataset):
    assert isinstance(kelmarsh_dataset, InMemoryDataset)


def test_dataset_loading(kelmarsh_dataset):
    kelmarsh_dataset = kelmarsh_dataset.shuffle()
    assert len(kelmarsh_dataset) > 0  # Dataset should not be empty


def test_dataset_features_and_labels(kelmarsh_dataset):
    # Get the first graph object in your dataset
    data = kelmarsh_dataset[0]

    # Check for feature matrix existence
    assert hasattr(data, "x")
    assert data.x.dim() == 2  # Features should be a 2D matrix

    # Check for edge index existence
    assert hasattr(data, "edge_index")
    assert data.edge_index.dim() == 2  # Edges should have 2 dimensions

    # Check for labels
    assert hasattr(data, "y")
    # Add any other dataset-specific checks below
