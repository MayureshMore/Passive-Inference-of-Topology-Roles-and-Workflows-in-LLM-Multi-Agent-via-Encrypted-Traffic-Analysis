"""Unit tests for per-flow feature extraction (determinism + correctness)."""
import numpy as np

from features.per_flow import compute_per_flow, PerFlowFeatures


# synthetic packets: (timestamp, wire_size, direction)  direction 1=out, -1=in
PKTS = [
    (0.00, 100, 1),
    (0.05, 1400, -1),
    (0.10, 1400, -1),
    (0.20, 80, 1),
    (0.25, 600, -1),
]


def test_per_flow_directional_counts():
    feat = compute_per_flow("flowA", bursts=[], all_packets=PKTS)
    assert feat.n_packets_out == 2
    assert feat.n_packets_in == 3
    assert feat.total_bytes_out == 180
    assert feat.total_bytes_in == 3400


def test_per_flow_vector_is_35dim():
    feat = compute_per_flow("flowA", bursts=[], all_packets=PKTS)
    v = feat.to_vector()
    assert v.shape == (35,)
    assert v.dtype.kind == "f"


def test_per_flow_is_deterministic():
    v1 = compute_per_flow("flowA", bursts=[], all_packets=PKTS).to_vector()
    v2 = compute_per_flow("flowA", bursts=[], all_packets=list(PKTS)).to_vector()
    assert np.array_equal(v1, v2)


def test_per_flow_empty_is_safe():
    feat = compute_per_flow("empty", bursts=[], all_packets=[])
    v = feat.to_vector()
    assert v.shape == (35,)
    assert np.all(np.isfinite(v))


def test_per_flow_vector_has_no_nans():
    v = compute_per_flow("flowA", bursts=[], all_packets=PKTS).to_vector()
    assert np.all(np.isfinite(v))
