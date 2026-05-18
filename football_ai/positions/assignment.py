from __future__ import annotations

import itertools
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _scipy_linear_sum_assignment
except Exception:  # pragma: no cover
    _scipy_linear_sum_assignment = None

DEFAULT_RATIO_PRIORITY_MIN_COUNT = 200
DEFAULT_RATIO_PRIORITY_MIN_CUMULATIVE_RATIO = 0.40
DEFAULT_RATIO_PRIORITY_MIN_FINAL_RATIO = 0.40
SUPPORTED_EXPECTED_ROLES_ASSIGNMENT_METHODS = ("hungarian", "ratio_priority")


def normalize_role_token(value: Any) -> str:
    token = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    return "_".join(part for part in token.split("_") if part)


def normalize_expected_roles_assignment_method(value: Any, default: str = "hungarian") -> str:
    normalized = str(value or default).strip().lower()
    if normalized not in SUPPORTED_EXPECTED_ROLES_ASSIGNMENT_METHODS:
        return str(default)
    return normalized


def _solve_cost_matrix(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if _scipy_linear_sum_assignment is not None:
        return _scipy_linear_sum_assignment(cost_matrix)

    num_rows, num_cols = cost_matrix.shape
    best_cost = None
    best_pairs = None
    for chosen_cols in itertools.permutations(range(num_cols), min(num_rows, num_cols)):
        total_cost = 0.0
        pairs = []
        for row_idx, col_idx in enumerate(chosen_cols):
            total_cost += float(cost_matrix[row_idx, col_idx])
            pairs.append((row_idx, col_idx))
        if best_cost is None or total_cost < best_cost:
            best_cost = total_cost
            best_pairs = pairs
    if not best_pairs:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    return (
        np.asarray([pair[0] for pair in best_pairs], dtype=np.int64),
        np.asarray([pair[1] for pair in best_pairs], dtype=np.int64),
    )


def build_state_from_player_prediction_row(row: Mapping[str, Any], label_names: Sequence[str]) -> dict[str, Any]:
    observations = max(1, int(row.get("frames_seen", 1) or 1))
    predicted_role = normalize_role_token(row.get("predicted_role"))
    confidence = float(row.get("predicted_role_confidence", 0.0) or 0.0)
    state = {
        "team_id": str(row.get("team_id")),
        "player_id": int(row.get("player_id")),
        "observations": int(observations),
        "role_counts": {},
        "confidence_sums": {},
        "prob_sums": {},
    }
    if predicted_role:
        state["role_counts"][predicted_role] = int(observations)
        state["confidence_sums"][predicted_role] = float(confidence) * float(observations)
    for label_name in label_names:
        role_label = normalize_role_token(label_name)
        if not role_label:
            continue
        prob_value = float(row.get(f"prob_{label_name}", 0.0) or 0.0)
        state["prob_sums"][role_label] = float(prob_value) * float(observations)
    return state


def _select_stable_role_from_counts(role_counts: Mapping[str, Any], confidence_sums: Mapping[str, Any]) -> str | None:
    best_role = None
    best_priority = None
    for role_name, count in role_counts.items():
        confidence_sum = float(confidence_sums.get(role_name, 0.0))
        mean_confidence = confidence_sum / max(1, int(count))
        priority = (int(count), mean_confidence, str(role_name))
        if best_priority is None or priority > best_priority:
            best_priority = priority
            best_role = str(role_name)
    return best_role


def _ordered_role_labels_from_keys(role_names: Sequence[Any]) -> list[str]:
    normalized: list[str] = []
    for raw_role in role_names:
        role = normalize_role_token(raw_role)
        if not role or role in normalized or role == "POR":
            continue
        normalized.append(role)
    return normalized


def _rank_roles_for_state(state: Mapping[str, Any], role_labels: Sequence[str]) -> list[dict[str, Any]]:
    observations = max(1, int(state.get("observations", 0)))
    ranked_roles = []
    for normalized_role in role_labels:
        count = int(state.get("role_counts", {}).get(normalized_role, 0))
        if count <= 0:
            continue
        ratio = float(count) / float(observations)
        mean_prob = float(state.get("prob_sums", {}).get(normalized_role, 0.0)) / float(observations)
        confidence_sum = float(state.get("confidence_sums", {}).get(normalized_role, 0.0))
        mean_confidence = confidence_sum / float(count) if count > 0 else 0.0
        ranked_roles.append(
            {
                "role": str(normalized_role),
                "ratio": float(ratio),
                "count": int(count),
                "mean_prob": float(mean_prob),
                "mean_confidence": float(mean_confidence),
                "confidence_sum": float(confidence_sum),
            }
        )
    ranked_roles.sort(
        key=lambda item: (
            item["ratio"],
            item["count"],
            item["mean_prob"],
            item["mean_confidence"],
            item["role"],
        ),
        reverse=True,
    )
    return ranked_roles


def _build_valid_role_ranking_with_invalid_transfer(
    state: Mapping[str, Any],
    role_labels: Sequence[str],
    valid_roles: Sequence[str],
    available_roles: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    ranked_roles = _rank_roles_for_state(state, role_labels)
    valid_role_set = {normalize_role_token(role) for role in valid_roles}
    available_counter = (
        Counter(normalize_role_token(role) for role in available_roles)
        if available_roles is not None
        else None
    )
    transferred_ranking = []
    carry_ratio = 0.0
    carry_count = 0
    carry_prob = 0.0
    carry_confidence_sum = 0.0

    for item in ranked_roles:
        role = str(item["role"])
        if role not in valid_role_set:
            carry_ratio += float(item["ratio"])
            carry_count += int(item["count"])
            carry_prob += float(item["mean_prob"])
            carry_confidence_sum += float(item["confidence_sum"])
            continue

        if available_counter is not None and available_counter.get(role, 0) <= 0:
            carry_ratio += float(item["ratio"])
            carry_count += int(item["count"])
            carry_prob += float(item["mean_prob"])
            carry_confidence_sum += float(item["confidence_sum"])
            continue

        raw_count = int(item["count"])
        raw_confidence_sum = float(item["confidence_sum"])
        effective_count = int(raw_count + carry_count)
        effective_confidence_sum = float(raw_confidence_sum + carry_confidence_sum)
        transferred_ranking.append(
            {
                "role": role,
                "effective_ratio": float(item["ratio"] + carry_ratio),
                "raw_ratio": float(item["ratio"]),
                "effective_count": int(effective_count),
                "raw_count": int(raw_count),
                "effective_mean_prob": float(item["mean_prob"] + carry_prob),
                "raw_mean_prob": float(item["mean_prob"]),
                "effective_mean_confidence": (
                    float(effective_confidence_sum) / float(effective_count)
                    if effective_count > 0
                    else 0.0
                ),
                "raw_mean_confidence": float(item["mean_confidence"]),
                "transferred_ratio": float(carry_ratio),
            }
        )
        carry_ratio = 0.0
        carry_count = 0
        carry_prob = 0.0
        carry_confidence_sum = 0.0

    return transferred_ranking


def _build_available_role_assignment_items(
    state: Mapping[str, Any],
    role_labels: Sequence[str],
    valid_roles: Sequence[str],
    available_roles: Sequence[str],
) -> dict[str, dict[str, Any]]:
    ranking = _build_valid_role_ranking_with_invalid_transfer(
        state=state,
        role_labels=role_labels,
        valid_roles=valid_roles,
        available_roles=available_roles,
    )
    return {str(item["role"]): item for item in ranking}


def _build_direct_role_assignment_item(
    state: Mapping[str, Any],
    normalized_role: str,
    observations: int,
) -> dict[str, Any]:
    raw_count = int(state.get("role_counts", {}).get(normalized_role, 0))
    raw_ratio = float(raw_count) / float(max(1, observations))
    raw_mean_prob = float(state.get("prob_sums", {}).get(normalized_role, 0.0)) / float(max(1, observations))
    confidence_sum = float(state.get("confidence_sums", {}).get(normalized_role, 0.0))
    raw_mean_confidence = confidence_sum / float(raw_count) if raw_count > 0 else 0.0
    return {
        "role": str(normalized_role),
        "effective_ratio": float(raw_ratio),
        "raw_ratio": float(raw_ratio),
        "effective_count": int(raw_count),
        "raw_count": int(raw_count),
        "effective_mean_prob": float(raw_mean_prob),
        "raw_mean_prob": float(raw_mean_prob),
        "effective_mean_confidence": float(raw_mean_confidence),
        "raw_mean_confidence": float(raw_mean_confidence),
        "transferred_ratio": 0.0,
    }


def _role_assignment_score_from_item(item: Mapping[str, Any], observations: int) -> float:
    return (
        float(item.get("effective_ratio", 0.0))
        + 1e-3 * float(item.get("raw_ratio", 0.0))
        + 1e-6 * float(item.get("effective_mean_prob", 0.0))
        + 1e-9 * float(item.get("raw_mean_prob", 0.0))
        + 1e-12 * float(observations)
    )


def _resolve_remaining_snapshot_assignments(
    pending_candidates: Mapping[int, Mapping[str, Any]],
    available_roles: Sequence[str],
    role_labels: Sequence[str],
    valid_roles: Sequence[str],
    allow_zero_score: bool = False,
    phase: str = "remaining_optimal",
) -> tuple[list[dict[str, Any]], dict[int, Mapping[str, Any]], list[str]]:
    assignments = []
    pending = {int(track_id): state for track_id, state in pending_candidates.items()}
    free_roles = [str(role) for role in available_roles]
    large_cost = 1e6

    while pending and free_roles:
        player_ids = sorted(int(track_id) for track_id in pending.keys())
        role_instances = list(free_roles)
        cost_matrix = np.full((len(player_ids), len(role_instances)), large_cost, dtype=np.float64)
        pair_payload = {}

        for row_pos, track_id in enumerate(player_ids):
            state = pending[int(track_id)]
            observations = max(1, int(state.get("observations", 0)))
            items_by_role = _build_available_role_assignment_items(
                state=state,
                role_labels=role_labels,
                valid_roles=valid_roles,
                available_roles=role_instances,
            )
            for col_pos, raw_role in enumerate(role_instances):
                normalized_role = normalize_role_token(raw_role)
                item = items_by_role.get(normalized_role)
                if item is None and allow_zero_score:
                    item = _build_direct_role_assignment_item(
                        state=state,
                        normalized_role=str(normalized_role),
                        observations=int(observations),
                    )
                if item is None:
                    continue
                if not allow_zero_score and float(item.get("effective_ratio", 0.0)) <= 0.0:
                    continue
                score = _role_assignment_score_from_item(item, observations)
                cost_matrix[row_pos, col_pos] = -float(score)
                pair_payload[(row_pos, col_pos)] = {
                    "track_id": int(track_id),
                    "raw_role": str(raw_role),
                    "normalized_role": str(normalized_role),
                    "state": state,
                    "item": item,
                    "observations": int(observations),
                }

        if not pair_payload:
            break

        row_ind, col_ind = _solve_cost_matrix(cost_matrix)
        best_choice = None
        for row_pos, col_pos in zip(row_ind.tolist(), col_ind.tolist()):
            if cost_matrix[row_pos, col_pos] >= large_cost / 2.0:
                continue
            payload = pair_payload.get((row_pos, col_pos))
            if payload is None:
                continue
            item = payload["item"]
            priority = (
                float(item.get("effective_ratio", 0.0)),
                float(item.get("raw_ratio", 0.0)),
                int(item.get("effective_count", 0)),
                int(item.get("raw_count", 0)),
                float(item.get("effective_mean_prob", 0.0)),
                float(item.get("raw_mean_prob", 0.0)),
                float(item.get("effective_mean_confidence", 0.0)),
                float(item.get("raw_mean_confidence", 0.0)),
                int(payload["observations"]),
                -int(payload["track_id"]),
            )
            if best_choice is None or priority > best_choice[0]:
                best_choice = (priority, payload, list(free_roles))

        if best_choice is None:
            break

        _, payload, available_before = best_choice
        item = payload["item"]
        assignments.append(
            {
                "player_id": int(payload["track_id"]),
                "slot": str(payload["raw_role"]),
                "slot_normalized": str(payload["normalized_role"]),
                "effective_ratio": float(item.get("effective_ratio", 0.0)),
                "final_ratio": float(item.get("raw_ratio", 0.0)),
                "effective_count": int(item.get("effective_count", 0)),
                "count": int(item.get("raw_count", 0)),
                "observations": int(payload["observations"]),
                "available_before": list(available_before),
                "transferred_ratio": float(item.get("transferred_ratio", 0.0)),
                "phase": str(phase),
            }
        )
        pending.pop(int(payload["track_id"]), None)
        try:
            free_roles.remove(str(payload["raw_role"]))
        except ValueError:
            pass

    return assignments, pending, free_roles


def simulate_ratio_priority_snapshot_for_team(
    team_id: Any,
    states: Mapping[int, Mapping[str, Any]],
    expected_roles: Sequence[str],
    min_count: int = DEFAULT_RATIO_PRIORITY_MIN_COUNT,
    min_cumulative_ratio: float = DEFAULT_RATIO_PRIORITY_MIN_CUMULATIVE_RATIO,
    min_final_ratio: float = DEFAULT_RATIO_PRIORITY_MIN_FINAL_RATIO,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    team_key = str(team_id)
    available_roles = [str(role) for role in (expected_roles or []) if normalize_role_token(role) != "POR"]
    ranked_expected_roles = _ordered_role_labels_from_keys(expected_roles or [])
    role_labels = _ordered_role_labels_from_keys(
        set(ranked_expected_roles)
        | {
            role
            for state in states.values()
            if str(state.get("team_id")) == team_key
            for role in list(state.get("role_counts", {}).keys()) + list(state.get("prob_sums", {}).keys())
        }
    )

    eligible = {}
    low_observations = {}
    for player_id, state in states.items():
        if str(state.get("team_id")) != team_key:
            continue
        if int(state.get("observations", 0)) >= int(min_count):
            eligible[int(player_id)] = state
        else:
            low_observations[int(player_id)] = state

    pending = dict(eligible)
    steps = []
    while pending and available_roles:
        best_assignment = None
        for player_id, state in pending.items():
            observations = max(1, int(state.get("observations", 0)))
            valid_ranking = _build_valid_role_ranking_with_invalid_transfer(
                state=state,
                role_labels=role_labels,
                valid_roles=ranked_expected_roles,
                available_roles=available_roles,
            )
            chosen = None
            if valid_ranking:
                item = valid_ranking[0]
                normalized_role = str(item["role"])
                chosen = (
                    float(item["effective_ratio"]),
                    float(item["raw_ratio"]),
                    int(item["effective_count"]),
                    int(item["raw_count"]),
                    float(item["effective_mean_prob"]),
                    float(item["raw_mean_prob"]),
                    float(item["effective_mean_confidence"]),
                    float(item["raw_mean_confidence"]),
                    float(item["transferred_ratio"]),
                    str(normalized_role),
                )

            if chosen is None:
                continue

            (
                effective_ratio,
                final_ratio,
                effective_count,
                final_count,
                effective_prob,
                final_prob,
                effective_conf,
                final_conf,
                transferred_ratio,
                normalized_role,
            ) = chosen
            if effective_ratio < float(min_cumulative_ratio) or final_ratio < float(min_final_ratio):
                continue

            raw_role = next(
                (
                    str(role)
                    for role in available_roles
                    if normalize_role_token(role) == normalized_role
                ),
                None,
            )
            if raw_role is None:
                continue

            candidate = (
                float(effective_ratio),
                float(final_ratio),
                int(effective_count),
                int(final_count),
                float(effective_prob),
                float(final_prob),
                float(effective_conf),
                float(final_conf),
                int(observations),
                -int(player_id),
            )
            if best_assignment is None or candidate > best_assignment[0]:
                best_assignment = (
                    candidate,
                    int(player_id),
                    str(raw_role),
                    str(normalized_role),
                    state,
                    list(available_roles),
                    float(transferred_ratio),
                )

        if best_assignment is None:
            break

        (
            candidate,
            player_id,
            raw_role,
            normalized_role,
            state,
            available_before,
            transferred_ratio,
        ) = best_assignment
        observations = max(1, int(state.get("observations", 0)))
        steps.append(
            {
                "team_id": team_key,
                "step_idx": len(steps),
                "player_id": int(player_id),
                "slot": str(raw_role),
                "slot_normalized": str(normalized_role),
                "effective_ratio": float(candidate[0]),
                "final_ratio": float(candidate[1]),
                "effective_count": int(candidate[2]),
                "count": int(candidate[3]),
                "observations": int(observations),
                "available_before": list(available_before),
                "transferred_ratio": float(transferred_ratio),
                "phase": "threshold",
            }
        )
        pending.pop(int(player_id), None)
        try:
            available_roles.remove(str(raw_role))
        except ValueError:
            pass

    residual_assignments, pending, available_roles = _resolve_remaining_snapshot_assignments(
        pending_candidates=pending,
        available_roles=available_roles,
        role_labels=role_labels,
        valid_roles=ranked_expected_roles,
    )
    for assignment in residual_assignments:
        steps.append(
            {
                "team_id": team_key,
                "step_idx": len(steps),
                "player_id": int(assignment["player_id"]),
                "slot": str(assignment["slot"]),
                "slot_normalized": str(assignment["slot_normalized"]),
                "effective_ratio": float(assignment["effective_ratio"]),
                "final_ratio": float(assignment["final_ratio"]),
                "effective_count": int(assignment["effective_count"]),
                "count": int(assignment["count"]),
                "observations": int(assignment["observations"]),
                "available_before": list(assignment["available_before"]),
                "transferred_ratio": float(assignment["transferred_ratio"]),
                "phase": str(assignment.get("phase", "remaining_optimal")),
            }
        )

    final_fill_candidates = dict(pending)
    final_fill_candidates.update(low_observations)
    final_fill_assignments, final_fill_candidates, available_roles = _resolve_remaining_snapshot_assignments(
        pending_candidates=final_fill_candidates,
        available_roles=available_roles,
        role_labels=role_labels,
        valid_roles=ranked_expected_roles,
        allow_zero_score=True,
        phase="fill_remaining",
    )
    for assignment in final_fill_assignments:
        steps.append(
            {
                "team_id": team_key,
                "step_idx": len(steps),
                "player_id": int(assignment["player_id"]),
                "slot": str(assignment["slot"]),
                "slot_normalized": str(assignment["slot_normalized"]),
                "effective_ratio": float(assignment["effective_ratio"]),
                "final_ratio": float(assignment["final_ratio"]),
                "effective_count": int(assignment["effective_count"]),
                "count": int(assignment["count"]),
                "observations": int(assignment["observations"]),
                "available_before": list(assignment["available_before"]),
                "transferred_ratio": float(assignment["transferred_ratio"]),
                "phase": str(assignment.get("phase", "fill_remaining")),
            }
        )
    pending = {
        int(player_id): state
        for player_id, state in pending.items()
        if int(player_id) in final_fill_candidates
    }
    low_observations = {
        int(player_id): state
        for player_id, state in low_observations.items()
        if int(player_id) in final_fill_candidates
    }

    unresolved = []
    remaining_counter = Counter(normalize_role_token(role) for role in available_roles)
    for category, candidates in (("threshold", pending), ("low_observations", low_observations)):
        for player_id, state in sorted(candidates.items()):
            observations = max(1, int(state.get("observations", 0)))
            valid_ranking = _build_valid_role_ranking_with_invalid_transfer(
                state=state,
                role_labels=role_labels,
                valid_roles=ranked_expected_roles,
                available_roles=available_roles,
            )
            best_available_role = None
            best_available_count = 0
            best_available_ratio = 0.0
            best_available_effective_ratio = 0.0
            transferred_ratio = 0.0
            if valid_ranking:
                item = valid_ranking[0]
                normalized_role = str(item["role"])
                if remaining_counter.get(normalized_role, 0) > 0:
                    best_available_role = str(normalized_role)
                    best_available_count = int(item["raw_count"])
                    best_available_ratio = float(item["raw_ratio"])
                    best_available_effective_ratio = float(item["effective_ratio"])
                    transferred_ratio = float(item["transferred_ratio"])

            dominant_role = _select_stable_role_from_counts(
                state.get("role_counts", {}),
                state.get("confidence_sums", {}),
            )
            dominant_count = int(state.get("role_counts", {}).get(dominant_role, 0))
            unresolved.append(
                {
                    "player_id": int(player_id),
                    "category": str(category),
                    "observations": int(observations),
                    "dominant_role": dominant_role,
                    "dominant_ratio": float(dominant_count) / float(observations),
                    "best_available_role": best_available_role,
                    "best_available_count": int(best_available_count),
                    "best_available_ratio": float(best_available_ratio),
                    "best_available_effective_ratio": (
                        float(best_available_effective_ratio) if best_available_role is not None else 0.0
                    ),
                    "best_available_transferred_ratio": float(transferred_ratio),
                }
            )

    return steps, unresolved, list(available_roles)
