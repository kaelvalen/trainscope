from trainscope.layer_matrix import build_layer_matrix, natural_key


def _ramp(n, onset=None):
    return [
        (
            s,
            1.0 + 0.01 * (s % 3) + (0.5 * (s - onset) if onset is not None and s >= onset else 0.0),
        )
        for s in range(n)
    ]


def test_rows_are_naturally_sorted():
    m = build_layer_matrix({"blocks.10.w": _ramp(4), "blocks.2.w": _ramp(4), "emb": _ramp(4)})
    assert m["layers"] == ["blocks.2.w", "blocks.10.w", "emb"]
    assert sorted(["b.10", "b.2"], key=natural_key) == ["b.2", "b.10"]


def test_one_bucket_per_step_when_short():
    assert build_layer_matrix({"a": _ramp(5)}, buckets=600)["bucket_starts"] == [0, 1, 2, 3, 4]


def test_bucket_takes_max_and_respects_range():
    m = build_layer_matrix({"a": [(s, float(s)) for s in range(100)]}, buckets=10, start=20, end=59)
    assert m["bucket_starts"][0] == 20 and len(m["bucket_starts"]) == 10
    assert m["values"][0][0] == 23.0


def test_baseline_is_pre_spike_and_z_flags_onset():
    m = build_layer_matrix({"a": _ramp(60, onset=40)}, buckets=60, first_spike=45)
    assert m["baseline"] == {"start": 0, "end": 44}
    assert m["z"][0][10] < 3
    assert m["z"][0][50] == 6.0


def test_no_spike_uses_first_half():
    assert build_layer_matrix({"a": _ramp(40)}, buckets=40)["baseline"] == {"start": 0, "end": 19}


def test_nonfinite_values_become_gaps():
    m = build_layer_matrix({"a": [(0, float("nan")), (1, None), (2, float("inf"))], "b": _ramp(3)})
    assert m["values"][0] == [None, None, None] and m["z"][0] == [None, None, None]


def test_empty_input():
    assert build_layer_matrix({})["layers"] == []
