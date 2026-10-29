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


def test_steps_land_in_the_bucket_their_start_labels():
    m = build_layer_matrix({"a": [(s, float(s)) for s in range(10)]}, buckets=3)
    starts = m["bucket_starts"]
    assert starts == [0, 4, 7]
    # each bucket's max is the largest step that is >= its start and < the next start
    assert m["values"][0] == [3.0, 6.0, 9.0]
    for span in range(1, 40):
        for k in range(1, span + 1):
            mm = build_layer_matrix({"a": [(s, float(s)) for s in range(span)]}, buckets=k)
            b = mm["bucket_starts"]
            for i, v in enumerate(mm["values"][0]):
                end = b[i + 1] if i + 1 < len(b) else span
                assert b[i] <= v < end


def test_column_reader_returns_only_requested_columns(tmp_path):
    import pyarrow as pa
    import pyarrow.ipc as ipc

    from trainscope.io.writer import read_arrow_columns_sync

    path = tmp_path / "layer.arrow"
    table = pa.table(
        {"step": [0, 1, 2], "grad_l2_norm": [1.0, 2.0, 3.0], "hist_counts": [[1.0]] * 3}
    )
    for writer_factory in (ipc.new_file, ipc.new_stream):
        with pa.OSFile(str(path), "wb") as sink, writer_factory(sink, table.schema) as w:
            w.write_table(table)
        cols = read_arrow_columns_sync(path, ["step", "grad_l2_norm"])
        assert cols == {"step": [0, 1, 2], "grad_l2_norm": [1.0, 2.0, 3.0]}
    assert read_arrow_columns_sync(tmp_path / "missing.arrow", ["step"]) == {}
