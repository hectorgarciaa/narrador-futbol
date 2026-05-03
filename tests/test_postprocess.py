"""Test postprocess_emit_block: edges -> events por bloque emitido, no ultimo edge."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.actions.postprocess import postprocess_emit_block, reset_postprocess_state


def test_edge_A_10_frames_emitted():
    reset_postprocess_state()
    edges = [
        {"frame_id": 100 + i, "canonical_src": "5", "canonical_dst": "8",
         "edge_src": "home_5", "edge_dst": "home_8"}
        for i in range(10)
    ]
    actions = postprocess_emit_block(edges, total_emit_frames=40)
    assert len(actions) == 1
    a = actions[0]
    assert a["canonical_src"] == "5"
    assert a["event_type"] == "pase"
    assert a["support_frames"] == 10
    assert a["longest_consecutive_run"] == 10
    print("PASS test_edge_A_10_frames_emitted")


def test_noise_B_1_frame_ignored():
    reset_postprocess_state()
    edges = [
        {"frame_id": 100, "canonical_src": "2", "canonical_dst": "8",
         "edge_src": "home_2", "edge_dst": "home_8"},
    ]
    actions = postprocess_emit_block(edges, total_emit_frames=40)
    assert len(actions) == 0
    print("PASS test_noise_B_1_frame_ignored")


def test_edge_C_5_frames_borderline():
    reset_postprocess_state()
    edges = [
        {"frame_id": 200 + i, "canonical_src": "5", "canonical_dst": "8",
         "edge_src": "home_5", "edge_dst": "home_8"}
        for i in range(5)
    ]
    actions = postprocess_emit_block(edges, total_emit_frames=30)
    assert len(actions) == 1
    a = actions[0]
    assert a["support_frames"] == 5
    print("PASS test_edge_C_5_frames_borderline")


def test_edge_D_4_frames_below_threshold():
    reset_postprocess_state()
    edges = [
        {"frame_id": 300 + i, "canonical_src": "5", "canonical_dst": "8",
         "edge_src": "home_5", "edge_dst": "home_8"}
        for i in range(4)
    ]
    actions = postprocess_emit_block(edges, total_emit_frames=40)
    assert len(actions) == 0
    print("PASS test_edge_D_4_frames_below_threshold")


def test_no_last_edge_only():
    reset_postprocess_state()
    edges = [
        *[
            {"frame_id": 100 + i, "canonical_src": "5", "canonical_dst": "8",
             "edge_src": "home_5", "edge_dst": "home_8"}
            for i in range(10)
        ],
        {"frame_id": 110, "canonical_src": "2", "canonical_dst": "8",
         "edge_src": "home_2", "edge_dst": "home_8"},
    ]
    actions = postprocess_emit_block(edges, total_emit_frames=40)
    assert len(actions) == 1
    a = actions[0]
    assert a["canonical_src"] == "5"
    print("PASS test_no_last_edge_only")


def test_consecutive_run_requirement():
    reset_postprocess_state()
    edges = [
        {"frame_id": 100, "canonical_src": "5", "canonical_dst": "8",
         "edge_src": "home_5", "edge_dst": "home_8"},
        {"frame_id": 102, "canonical_src": "5", "canonical_dst": "8",
         "edge_src": "home_5", "edge_dst": "home_8"},
        {"frame_id": 105, "canonical_src": "5", "canonical_dst": "8",
         "edge_src": "home_5", "edge_dst": "home_8"},
        {"frame_id": 110, "canonical_src": "5", "canonical_dst": "8",
         "edge_src": "home_5", "edge_dst": "home_8"},
        {"frame_id": 115, "canonical_src": "5", "canonical_dst": "8",
         "edge_src": "home_5", "edge_dst": "home_8"},
        {"frame_id": 120, "canonical_src": "5", "canonical_dst": "8",
         "edge_src": "home_5", "edge_dst": "home_8"},
    ]
    actions = postprocess_emit_block(edges, total_emit_frames=40)
    assert len(actions) == 0
    print("PASS test_consecutive_run_requirement")


def test_support_ratio_enforced():
    reset_postprocess_state()
    edges = [
        {"frame_id": i, "canonical_src": "5", "canonical_dst": "8",
         "edge_src": "home_5", "edge_dst": "home_8"}
        for i in range(5)
    ]
    actions = postprocess_emit_block(edges, total_emit_frames=100)
    assert len(actions) == 0
    print("PASS test_support_ratio_enforced")


def test_support_ratio_passes():
    reset_postprocess_state()
    edges = [
        {"frame_id": i, "canonical_src": "5", "canonical_dst": "8",
         "edge_src": "home_5", "edge_dst": "home_8"}
        for i in range(5)
    ]
    actions = postprocess_emit_block(edges, total_emit_frames=30)
    assert len(actions) == 1
    print("PASS test_support_ratio_passes")


def test_event_fields_have_start_end_frame():
    """Evento debe incluir start_frame, end_frame, source."""
    reset_postprocess_state()
    edges = [
        {"frame_id": 400 + i, "canonical_src": "5", "canonical_dst": "8",
         "edge_src": "home_5", "edge_dst": "home_8"}
        for i in range(8)
    ]
    actions = postprocess_emit_block(edges, total_emit_frames=40)
    assert len(actions) == 1
    a = actions[0]
    assert a["start_frame"] == 400
    assert a["end_frame"] == 407
    assert a["source"] == "postprocess_emit_block"
    assert a["event_type"] == "pase"
    assert a["canonical_src"] == "5"
    assert a["canonical_dst"] == "8"
    print("PASS test_event_fields_have_start_end_frame")


def test_latest_event_is_not_last_edge():
    """El evento consolidado debe ser A, no B aunque B este al final."""
    reset_postprocess_state()
    edges = [
        *[
            {"frame_id": 500 + i, "canonical_src": "5", "canonical_dst": "8",
             "edge_src": "home_5", "edge_dst": "home_8"}
            for i in range(10)
        ],
        {"frame_id": 510, "canonical_src": "2", "canonical_dst": "8",
         "edge_src": "home_2", "edge_dst": "home_8"},
    ]
    actions = postprocess_emit_block(edges, total_emit_frames=40)
    assert len(actions) == 1
    assert actions[0]["canonical_src"] == "5"
    print("PASS test_latest_event_is_not_last_edge")


if __name__ == "__main__":
    test_edge_A_10_frames_emitted()
    test_noise_B_1_frame_ignored()
    test_edge_C_5_frames_borderline()
    test_edge_D_4_frames_below_threshold()
    test_no_last_edge_only()
    test_consecutive_run_requirement()
    test_support_ratio_enforced()
    test_support_ratio_passes()
    test_event_fields_have_start_end_frame()
    test_latest_event_is_not_last_edge()
    print("\n=== 10/10 tests de postprocess pasaron ===")
