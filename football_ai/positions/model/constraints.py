from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .config import ExpectedRoleSlot


ROLE_SLOT_ALIASES: dict[str, tuple[str, ...]] = {
    "GK": ("POR",),
    "GOALKEEPER": ("POR",),
    "KEEPER": ("POR",),
    "DFIZQ": ("DFC_IZQ",),
    "DF_IZQ": ("DFC_IZQ",),
    "DFCIZQ": ("DFC_IZQ",),
    "DFDCHA": ("DFC_DER",),
    "DF_DCHA": ("DFC_DER",),
    "DFC_DCHA": ("DFC_DER",),
    "DFDER": ("DFC_DER",),
    "DF_DER": ("DFC_DER",),
    "DFCENT": ("DFC_CENT",),
    "DFC_CENTRAL": ("DFC_CENT",),
    "DEL": ("DC",),
    "ST": ("DC",),
    "STRIKER": ("DC",),
}

LATERAL_ROLE_FAMILIES: dict[str, tuple[str, ...]] = {
    "LEFT": ("CI", "LI", "MI", "EI"),
    "RIGHT": ("CD", "LD", "MD", "ED"),
}
ROLE_TO_LATERAL_FAMILY = {
    role_label: family_name
    for family_name, role_labels in LATERAL_ROLE_FAMILIES.items()
    for role_label in role_labels
}
ROLE_SLOT_ANCHORS: dict[str, tuple[float, float]] = {
    "CI": (0.40, 0.12),
    "LI": (0.26, 0.12),
    "DFC_IZQ": (0.18, 0.34),
    "DFC_CENT": (0.16, 0.50),
    "DFC_DER": (0.18, 0.66),
    "LD": (0.26, 0.88),
    "CD": (0.40, 0.88),
    "MC": (0.52, 0.50),
    "MI": (0.60, 0.22),
    "MD": (0.60, 0.78),
    "EI": (0.80, 0.12),
    "ED": (0.80, 0.88),
    "DC": (0.84, 0.50),
}
SLOT_GEOMETRY_WEIGHT = 1.35
SLOT_LABEL_COMPAT_WEIGHT = 1.10
ANCHOR_DY_WEIGHT = 1.20


def constrain_player_predictions_with_expected_roles(
    player_predictions_df: pd.DataFrame,
    label_names: Sequence[str],
    expected_roles_by_team: Mapping[str, Sequence[str]] | None,
) -> pd.DataFrame:
    constrained = player_predictions_df.copy()
    constrained["predicted_role_unconstrained"] = constrained["predicted_role"].astype(str)
    constrained["predicted_role_confidence_unconstrained"] = pd.to_numeric(
        constrained["predicted_role_confidence"],
        errors="coerce",
    ).astype(np.float32)
    constrained["expected_role_slot"] = pd.Series([pd.NA] * len(constrained), dtype="string")
    constrained["assignment_method"] = pd.Series(["unconstrained"] * len(constrained), dtype="string")
    constrained["assignment_cost"] = np.nan
    constrained["matched_model_role"] = pd.Series([pd.NA] * len(constrained), dtype="string")
    if not expected_roles_by_team:
        return constrained

    available_team_ids = set(constrained["team_id"].astype(str).unique().tolist())
    epsilon = 1e-9
    prob_cols = [f"prob_{label}" for label in label_names]
    for raw_team_id, expected_roles in expected_roles_by_team.items():
        team_id = str(raw_team_id)
        if team_id not in available_team_ids:
            continue
        team_df = constrained.loc[constrained["team_id"].astype(str) == team_id].copy()
        slots = [_resolve_expected_role_slot(role_label, label_names) for role_label in expected_roles]
        goalkeeper_slots = [slot for slot in slots if "POR" in slot.allowed_labels]
        field_slots = [slot for slot in slots if "POR" not in slot.allowed_labels]
        _assign_goalkeepers(constrained, team_df, goalkeeper_slots)
        _assign_field_players(constrained, team_df, field_slots, epsilon)

    keep_cols = [
        "team_id",
        "player_id",
        "class_name",
        "predicted_role",
        "predicted_role_confidence",
        "predicted_role_unconstrained",
        "predicted_role_confidence_unconstrained",
        "matched_model_role",
        "expected_role_slot",
        "assignment_method",
        "assignment_cost",
        "frames_seen",
        "x",
        "y",
        "x_m",
        "y_m",
        *prob_cols,
    ]
    return constrained.loc[:, [col for col in keep_cols if col in constrained.columns]]


def _assign_goalkeepers(constrained: pd.DataFrame, team_df: pd.DataFrame, goalkeeper_slots: list[ExpectedRoleSlot]) -> None:
    goalkeeper_df = team_df[team_df["class_name"].astype(str) == "goalkeeper"].copy()
    if len(goalkeeper_df) > len(goalkeeper_slots):
        raise ValueError(
            f"El equipo {team_df['team_id'].iloc[0]!r} tiene {len(goalkeeper_df)} goalkeeper tracks y {len(goalkeeper_slots)} slots POR esperados."
        )
    goalkeeper_df = goalkeeper_df.sort_values(["frames_seen", "player_id"], ascending=[False, True])
    for (_, row), slot in zip(goalkeeper_df.iterrows(), goalkeeper_slots):
        constrained.at[row.name, "predicted_role"] = "POR"
        constrained.at[row.name, "predicted_role_confidence"] = 1.0
        constrained.at[row.name, "expected_role_slot"] = slot.slot_label
        constrained.at[row.name, "assignment_method"] = "hungarian_expected_roles"
        constrained.at[row.name, "assignment_cost"] = 0.0


def _assign_field_players(
    constrained: pd.DataFrame,
    team_df: pd.DataFrame,
    field_slots: list[ExpectedRoleSlot],
    epsilon: float,
) -> None:
    if not field_slots:
        return
    field_df = team_df[team_df["class_name"].astype(str) != "goalkeeper"].copy()
    field_indices = field_df.index.tolist()
    cost_matrix = np.zeros((len(field_indices), len(field_slots)), dtype=np.float64)
    best_labels_per_pair: list[list[tuple[str, float]]] = []
    for row_pos, row_idx in enumerate(field_indices):
        player_row = field_df.loc[row_idx]
        row_pairs: list[tuple[str, float]] = []
        for col_pos, slot in enumerate(field_slots):
            best_label, best_prob, total_cost = _best_label_for_slot(player_row, slot, epsilon)
            row_pairs.append((best_label, best_prob))
            cost_matrix[row_pos, col_pos] = float(total_cost)
        best_labels_per_pair.append(row_pairs)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assigned_row_positions = set(row_ind.tolist())
    for row_pos, col_pos in zip(row_ind.tolist(), col_ind.tolist()):
        row_idx = field_indices[row_pos]
        slot = field_slots[col_pos]
        best_label, best_prob = best_labels_per_pair[row_pos][col_pos]
        constrained.at[row_idx, "predicted_role"] = slot.slot_label
        constrained.at[row_idx, "predicted_role_confidence"] = float(best_prob)
        constrained.at[row_idx, "expected_role_slot"] = slot.slot_label
        constrained.at[row_idx, "assignment_method"] = "hungarian_expected_roles"
        constrained.at[row_idx, "assignment_cost"] = float(cost_matrix[row_pos, col_pos])
        constrained.at[row_idx, "matched_model_role"] = str(best_label)

    fallback_options = _fallback_options(field_slots)
    for row_pos, row_idx in enumerate(field_indices):
        if row_pos in assigned_row_positions:
            continue
        _assign_fallback(constrained, field_df.loc[row_idx], row_idx, fallback_options, epsilon)


def _fallback_options(field_slots: list[ExpectedRoleSlot]) -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for slot in field_slots:
        for label in slot.allowed_labels:
            if label == "POR":
                continue
            pair = (slot.slot_label, str(label))
            if pair not in seen:
                options.append(pair)
                seen.add(pair)
    return options


def _assign_fallback(
    constrained: pd.DataFrame,
    player_row: pd.Series,
    row_idx: int,
    fallback_options: list[tuple[str, str]],
    epsilon: float,
) -> None:
    if not fallback_options:
        constrained.at[row_idx, "assignment_method"] = "expected_roles_unassigned"
        return
    best_slot_label = None
    best_label = None
    best_prob = -1.0
    best_cost = None
    for slot_label, label in fallback_options:
        prob_value = float(player_row.get(f"prob_{label}", 0.0))
        total_cost = -math.log(max(prob_value, epsilon))
        total_cost += _slot_label_compatibility_penalty(slot_label, label)
        total_cost += _slot_geometry_penalty(player_row, slot_label)
        if best_cost is None or total_cost < best_cost:
            best_slot_label = slot_label
            best_label = label
            best_prob = prob_value
            best_cost = float(total_cost)
    if best_label is None or best_slot_label is None:
        constrained.at[row_idx, "assignment_method"] = "expected_roles_unassigned"
        return
    constrained.at[row_idx, "predicted_role"] = best_slot_label
    constrained.at[row_idx, "predicted_role_confidence"] = float(best_prob)
    constrained.at[row_idx, "expected_role_slot"] = best_slot_label
    constrained.at[row_idx, "assignment_method"] = "expected_roles_fallback_best_allowed"
    constrained.at[row_idx, "assignment_cost"] = float(best_cost)
    constrained.at[row_idx, "matched_model_role"] = str(best_label)


def normalize_role_token(value: Any) -> str:
    token = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    return "_".join(part for part in token.split("_") if part)


def _resolve_expected_role_slot(role_label: Any, label_names: Sequence[str]) -> ExpectedRoleSlot:
    slot_label = normalize_role_token(role_label)
    known_labels = {str(label) for label in label_names}
    if slot_label in known_labels or slot_label == "POR":
        return ExpectedRoleSlot(
            input_label=str(role_label),
            slot_label=slot_label,
            allowed_labels=_expected_slot_allowed_labels(slot_label, known_labels),
        )
    allowed = ROLE_SLOT_ALIASES.get(slot_label)
    if allowed is None:
        supported = sorted(set(known_labels).union(ROLE_SLOT_ALIASES.keys(), {"POR"}))
        raise ValueError(f"Rol esperado no soportado: {role_label!r}. Usa labels del modelo o aliases soportados: {supported}")
    filtered = tuple(label for label in allowed if label == "POR" or label in known_labels)
    if not filtered:
        raise ValueError(f"El rol esperado {role_label!r} no es compatible con las clases del checkpoint.")
    return ExpectedRoleSlot(input_label=str(role_label), slot_label=slot_label, allowed_labels=filtered)


def _expected_slot_allowed_labels(slot_label: str, known_labels: set[str]) -> tuple[str, ...]:
    family_name = ROLE_TO_LATERAL_FAMILY.get(slot_label)
    if family_name is None:
        return (slot_label,)
    family_allowed = tuple(role_label for role_label in LATERAL_ROLE_FAMILIES[family_name] if role_label in known_labels)
    return family_allowed or (slot_label,)


def _best_label_for_slot(player_row: pd.Series, slot: ExpectedRoleSlot, epsilon: float) -> tuple[str, float, float]:
    geometry_penalty = _slot_geometry_penalty(player_row, slot.slot_label)
    best_label = None
    best_prob = 0.0
    best_cost = None
    for label in slot.allowed_labels:
        if label == "POR":
            continue
        prob_value = float(player_row.get(f"prob_{label}", 0.0))
        total_cost = -math.log(max(prob_value, epsilon))
        total_cost += _slot_label_compatibility_penalty(slot.slot_label, str(label))
        total_cost += geometry_penalty
        if best_cost is None or total_cost < best_cost:
            best_label = str(label)
            best_prob = float(prob_value)
            best_cost = float(total_cost)
    if best_label is None or best_cost is None:
        raise ValueError(f"El slot {slot.slot_label!r} no tiene labels válidos para la fila actual.")
    return best_label, best_prob, float(best_cost)


def _slot_label_compatibility_penalty(slot_label: str, candidate_label: str) -> float:
    if str(slot_label) == str(candidate_label):
        return 0.0
    slot_family = ROLE_TO_LATERAL_FAMILY.get(str(slot_label))
    candidate_family = ROLE_TO_LATERAL_FAMILY.get(str(candidate_label))
    if slot_family is None or candidate_family is None:
        return 0.0
    if slot_family != candidate_family:
        return 1_000.0
    slot_anchor = ROLE_SLOT_ANCHORS.get(str(slot_label))
    candidate_anchor = ROLE_SLOT_ANCHORS.get(str(candidate_label))
    if slot_anchor is None or candidate_anchor is None:
        return 0.35
    return SLOT_LABEL_COMPAT_WEIGHT * _weighted_anchor_distance(slot_anchor, candidate_anchor)


def _slot_geometry_penalty(player_row: pd.Series, slot_label: str) -> float:
    slot_anchor = ROLE_SLOT_ANCHORS.get(str(slot_label))
    if slot_anchor is None:
        return 0.0
    player_x = pd.to_numeric(player_row.get("x", np.nan), errors="coerce")
    player_y = pd.to_numeric(player_row.get("y", np.nan), errors="coerce")
    if pd.isna(player_x) or pd.isna(player_y):
        return 0.0
    return SLOT_GEOMETRY_WEIGHT * _weighted_anchor_distance((float(player_x), float(player_y)), slot_anchor)


def _weighted_anchor_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    dx = float(point_a[0]) - float(point_b[0])
    dy = float(point_a[1]) - float(point_b[1])
    return float(np.sqrt((dx * dx) + ((dy * ANCHOR_DY_WEIGHT) ** 2)))
