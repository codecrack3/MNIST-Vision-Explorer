import numpy as np

from src.model.inference import extract_activations
from src.model.network import ConvNet
from src.visualization.filters import get_filter_payload
from src.visualization.weights import get_weight_payload


def test_activation_shapes_match_architecture() -> None:
    model = ConvNet()
    image = np.zeros((28, 28), dtype=np.float32)

    result = extract_activations(model, image)

    assert result.conv1["shape"] == [16, 13, 13]
    assert result.conv2["shape"] == [32, 5, 5]
    assert result.fc1["shape"] == [128]
    assert len(result.output_probs) == 10


def test_filter_payload_shapes() -> None:
    model = ConvNet()
    payload = get_filter_payload(model)

    assert payload["conv1"]["shape"] == [16, 1, 3, 3]
    assert payload["conv2_summary"]["shape"] == [32, 16, 3, 3]
    assert payload["conv2_summary"]["reduced_shape"] == [32, 3, 3]


def test_weight_payload_histogram_consistency() -> None:
    model = ConvNet()
    payload = get_weight_payload(model)

    assert len(payload["layers"]) >= 3
    for layer in payload["layers"]:
        bins = layer["hist"]["bins"]
        counts = layer["hist"]["counts"]
        assert len(bins) == len(counts) + 1
        assert sum(counts) == layer["count"]
