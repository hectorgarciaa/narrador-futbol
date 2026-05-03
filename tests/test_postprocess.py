"""Test postprocess_emit_block: edges → events por bloque emitido, no ultimo edge."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.actions.postprocess import postprocess_emit_block, reset_postprocess_state


def test_edge_A_10_frames_emitted():
    """Edge A con 10 frames consecutivos debe ser emitido."""
    reset_postprocess_state()
    edges = [
        {"frame_id": 100 + i, "canonical_src": "5", "canonical_dst": "8",
         "edge_src": "home_5", "edge_dst": "home_8"}
        for i in range(10)
    ]
    actions = postprocess_emit_block(edges, total_emit_frames=40)
    assert len(actions) == 1, f"Esperado 1 action, obtenido {len(actions)}"
    a = actions[0]
    assert a["canonical_src"] == "5", f"src={a['canonical_src']}"
    assert a["canonical_dst"] == "8", f"dst={a['canonical_dst']}"
    assert a["event_type"] == "pase", f"event={a['event_type']}"
    assert a["support_frames"] == 10, f"support={a['support_frames']}"
    assert a["longest_consecutive_run"] == 10, f"consecutive={a['longest_consecutive_run']}"
    print("PASS test_edge_A_10_frames_emitted")


def test_noise_B_1_frame_ignored():
    """Edge B con solo 1 frame debe ser ignorado (min_support=5)."""
    reset_postprocess_state()
    edges = [
        {"frame_id": 100, "canonical_src": "2", "canonical_dst": "8",
         "edge_src": "home_2", "edge_dst": "home_8"},
    ]
    actions = postprocess_emit_block(edges, total_emit_frames=40)
    assert len(actions) == 0, f"Esperado 0 actions (ruido), obtenido {len(actions)}"
    print("PASS test_noise_B_1_frame_ignored")


def test_edge_C_5_frames_borderline():
    """Edge C con 5 frames y emit_frames=30: ratio=5/30=0.167 > 0.15, se emite."""
    reset_postprocess_state()
    edges = [
        {"frame_id": 200 + i, "canonical_src": "5", "canonical_dst": "8",
         "edge_src": "home_5", "edge_dst": "home_8"}
        for i in range(5)
    ]
    actions = postprocess_emit_block(edges, total_emit_frames=30)
    assert len(actions) == 1, f"Esperado 1 action (5 frames, ratio=0.167 > 0.15), obtenido {len(actions)}"
    a = actions[0]
    assert a["support_frames"] == 5
    print("PASS test_edge_C_5_frames_borderline")


def test_edge_D_4_frames_below_threshold():
    """Edge con 4 frames debe ser ignorado (< min_support=5)."""
    reset_postprocess_state()
    edges = [
        {"frame_id": 300 + i, "canonical_src": "5", "canonical_dst": "8",
         "edge_src": "home_5", "edge_dst": "home_8"}
        for i in range(4)
    ]
    actions = postprocess_emit_block(edges, total_emit_frames=40)
    assert len(actions) == 0, f"Esperado 0 actions (<5 frames), obtenido {len(actions)}"
    print("PASS test_edge_D_4_frames_below_threshold")


def test_no_last_edge_only():
    """Verificar que NO se usa solo el ultimo edge: si hay A(10 frames) y ruido B(1 frame),
    A debe ganar, no el ultimo edge del bloque."""
    reset_postprocess_state()
    edges = [
        # Edge A: 10 frames
        *[
            {"frame_id": 100 + i, "canonical_src": "5", "canonical_dst": "8",
             "edge_src": "home_5", "edge_dst": "home_8"}
            for i in range(10)
        ],
        # Ruido B: 1 frame al final del bloque
        {"frame_id": 110, "canonical_src": "2", "canonical_dst": "8",
         "edge_src": "home_2", "edge_dst": "home_8"},
    ]
    actions = postprocess_emit_block(edges, total_emit_frames=40)
    assert len(actions) == 1, f"Esperado 1 action (A gana a B), obtenido {len(actions)}"
    a = actions[0]
    assert a["canonical_src"] == "5", f"Gano el edge equivocado: src={a['canonical_src']}"
    print("PASS test_no_last_edge_only")


def test_consecutive_run_requirement():
    """Edge con 6 frames no consecutivos pero suficiente soporte: requiere consecutivos >=3."""
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
    # support_frames=6 (>=5), longest_consecutive=1 (<3), debe ignorarse
    actions = postprocess_emit_block(edges, total_emit_frames=40)
    assert len(actions) == 0, f"Esperado 0 actions (consecutive_run=1 < 3), obtenido {len(actions)}"
    print("PASS test_consecutive_run_requirement")


def test_support_ratio_enforced():
    """Edge con 5 frames sobre emit_frames=100: ratio=5/100=0.05 < 0.15, ignorado."""
    reset_postprocess_state()
    edges = [
        {"frame_id": i, "canonical_src": "5", "canonical_dst": "8",
         "edge_src": "home_5", "edge_dst": "home_8"}
        for i in range(5)
    ]
    actions = postprocess_emit_block(edges, total_emit_frames=100)  # ratio = 5/100 = 0.05
    assert len(actions) == 0, f"Esperado 0 actions (ratio 0.05 < 0.15), obtenido {len(actions)}"
    print("PASS test_support_ratio_enforced")


def test_support_ratio_passes():
    """Edge con 5 frames sobre emit_frames=30: ratio=5/30=0.167 > 0.15, emitido."""
    reset_postprocess_state()
    edges = [
        {"frame_id": i, "canonical_src": "5", "canonical_dst": "8",
         "edge_src": "home_5", "edge_dst": "home_8"}
        for i in range(5)
    ]
    actions = postprocess_emit_block(edges, total_emit_frames=30)
    assert len(actions) == 1
    print("PASS test_support_ratio_passes")


if __name__ == "__main__":
    test_edge_A_10_frames_emitted()
    test_noise_B_1_frame_ignored()
    test_edge_C_5_frames_borderline()
    test_edge_D_4_frames_below_threshold()
    test_no_last_edge_only()
    test_consecutive_run_requirement()
    test_support_ratio_enforced()
    print("\n=== Todos los tests de postprocess pasaron ===")
