from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd

from football_ai.positions.data import (
    POSITIONS_DATASET_DIR,
    POSITIONS_LABELS_DIR,
    ROLE_LABELS_V1,
    FeatureSpec,
    add_velocity_features,
    apply_periodic_role_labels_propagated,
    build_observations_from_tracks,
    build_periodic_label_schedule,
    build_periodic_role_label_template,
    build_role_samples,
    find_project_root,
    get_video_fps,
    infer_attack_direction_by_team,
    list_position_videos,
    load_role_label_entries_json,
    load_tracks_json,
    render_frame_with_player_ids,
    resolve_tracks_path_for_video,
    save_role_label_template_json,
    upsert_role_labels_in_template_json,
    upsert_run_to_common_dataset,
    validate_periodic_role_labels,
)


DEFAULT_PLAYER_IDS = tuple(range(1, 27))
_ACTIVE_LABELER: "PeriodicRoleLabelerUI | None" = None


@dataclass(frozen=True)
class WorkflowConfig:
    project_root: Path | None = None
    video_filename: str | None = None
    field_length_m: float = 106.0
    field_width_m: float = 68.0
    max_teammates: int = 10
    label_every_seconds: float = 6.0
    label_start_seconds: float = 0.0
    label_start_frame: int | None = None
    label_min_players: int = 18
    label_window_frames: int = 20
    labels_dir: Path | None = None
    role_labels: tuple[str, ...] = tuple(ROLE_LABELS_V1)


@dataclass(frozen=True)
class WorkflowContext:
    config: WorkflowConfig
    project_root: Path
    video_path: Path
    tracks_path: Path
    match_id: str
    labels_json_path: Path
    obs_df: pd.DataFrame
    video_fps: float
    auto_label_start_frame: int
    selected_label_start_frame: int
    effective_label_start_seconds: float
    label_schedule_df: pd.DataFrame

    @property
    def schedule_frames(self) -> list[int]:
        if self.label_schedule_df.empty:
            return []
        return (
            self.label_schedule_df["frame_id"]
            .astype(int)
            .drop_duplicates()
            .sort_values()
            .tolist()
        )


def _choose_video_path(project_root: Path, video_filename: str | None) -> Path:
    videos = list_position_videos(project_root)
    if not videos:
        raise FileNotFoundError("No se encontraron vídeos en data/partidosPosiciones.")

    if video_filename is None:
        return videos[0]

    selected = [path for path in videos if path.name == video_filename]
    if not selected:
        available = "\n".join(f" - {path.name}" for path in videos)
        raise FileNotFoundError(
            f"Vídeo no encontrado: {video_filename}\n"
            f"Disponibles en data/partidosPosiciones:\n{available}"
        )
    return selected[0]


def _auto_start_frame_first_20(obs_df: pd.DataFrame) -> int:
    if obs_df.empty:
        raise ValueError("No hay observaciones para seleccionar frame inicial.")

    first_20_obs = obs_df[obs_df["frame_id"].between(0, 19, inclusive="both")]
    if first_20_obs.empty:
        return int(obs_df["frame_id"].min())

    first_20_counts = first_20_obs.groupby("frame_id").size().sort_index()
    return int(first_20_counts.idxmax())


def _sync_schedule_metadata_if_needed(
    labels_json_path: Path,
    context: WorkflowContext,
    schedule_template: Mapping[str, Any],
) -> bool:
    if not labels_json_path.exists():
        return False

    with labels_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, Mapping):
        return False

    existing_schedule = payload.get("schedule", [])
    target_schedule = schedule_template.get("schedule", [])
    if existing_schedule == target_schedule:
        return False

    updated_payload = dict(payload)
    updated_payload["schedule"] = target_schedule
    updated_payload["fps"] = float(context.video_fps)
    updated_payload["label_every_seconds"] = float(context.config.label_every_seconds)
    updated_payload["label_start_seconds"] = float(context.effective_label_start_seconds)
    updated_payload["min_players_per_label_frame"] = int(context.config.label_min_players)

    with labels_json_path.open("w", encoding="utf-8") as f:
        json.dump(updated_payload, f, ensure_ascii=False, indent=2)
    return True


def prepare_periodic_role_workflow(config: WorkflowConfig) -> tuple[WorkflowContext, dict[str, Any]]:
    project_root = find_project_root(config.project_root)
    video_path = _choose_video_path(project_root, config.video_filename)
    tracks_path = resolve_tracks_path_for_video(project_root, video_path)
    if not tracks_path.exists():
        raise FileNotFoundError(
            f"No existe tracks JSON para el vídeo. Esperado: {tracks_path}"
        )

    match_id = video_path.stem.replace(" ", "_")
    labels_dir = config.labels_dir or (project_root / POSITIONS_LABELS_DIR)
    labels_json_path = labels_dir / f"{match_id}_labels_every_{int(config.label_every_seconds)}s.json"

    tracks = load_tracks_json(tracks_path)
    obs_df = build_observations_from_tracks(
        tracks=tracks,
        match_id=match_id,
        field_length_m=float(config.field_length_m),
        field_width_m=float(config.field_width_m),
        tracked_classes=("player", "goalkeeper"),
    )
    obs_df = add_velocity_features(obs_df)
    if obs_df.empty:
        raise ValueError(
            "No hay observaciones player/goalkeeper con field_position_m en tracks.json."
        )

    video_fps = get_video_fps(video_path, default_fps=25.0)
    auto_label_start_frame = _auto_start_frame_first_20(obs_df)
    selected_label_start_frame = (
        auto_label_start_frame
        if config.label_start_frame is None
        else int(config.label_start_frame)
    )

    effective_label_start_seconds = max(
        float(config.label_start_seconds),
        float(selected_label_start_frame) / float(video_fps),
    )

    label_schedule_df = build_periodic_label_schedule(
        observations=obs_df,
        fps=video_fps,
        every_seconds=float(config.label_every_seconds),
        start_seconds=effective_label_start_seconds,
        min_players=int(config.label_min_players),
        window_frames=int(config.label_window_frames),
    )
    if label_schedule_df.empty:
        raise ValueError("No se pudo construir una planificación de etiquetado temporal.")

    label_template = build_periodic_role_label_template(
        observations=obs_df,
        match_id=match_id,
        video_path=video_path,
        fps=video_fps,
        every_seconds=float(config.label_every_seconds),
        start_seconds=effective_label_start_seconds,
        min_players=int(config.label_min_players),
        window_frames=int(config.label_window_frames),
    )

    created_template = not labels_json_path.exists()
    save_role_label_template_json(label_template, labels_json_path, overwrite=False)

    context = WorkflowContext(
        config=config,
        project_root=project_root,
        video_path=video_path,
        tracks_path=tracks_path,
        match_id=match_id,
        labels_json_path=labels_json_path,
        obs_df=obs_df,
        video_fps=float(video_fps),
        auto_label_start_frame=auto_label_start_frame,
        selected_label_start_frame=selected_label_start_frame,
        effective_label_start_seconds=float(effective_label_start_seconds),
        label_schedule_df=label_schedule_df,
    )

    synced_existing_schedule = False
    if not created_template:
        synced_existing_schedule = _sync_schedule_metadata_if_needed(
            labels_json_path=labels_json_path,
            context=context,
            schedule_template=label_template,
        )

    summary = {
        "project_root": str(project_root),
        "video_path": str(video_path),
        "tracks_path": str(tracks_path),
        "match_id": match_id,
        "labels_json_path": str(labels_json_path),
        "fps": float(video_fps),
        "auto_label_start_frame": int(auto_label_start_frame),
        "selected_label_start_frame": int(selected_label_start_frame),
        "effective_label_start_seconds": float(effective_label_start_seconds),
        "label_window_frames": int(config.label_window_frames),
        "num_schedule_frames": int(len(label_schedule_df)),
        "created_template": bool(created_template),
        "synced_existing_schedule": bool(synced_existing_schedule),
        "teams": sorted(obs_df["team_id"].astype(str).unique().tolist()),
        "num_observations": int(len(obs_df)),
        "num_observation_frames": int(obs_df["frame_id"].nunique()),
    }
    return context, summary


def load_labels_and_validation(context: WorkflowContext) -> tuple[pd.DataFrame, dict[str, Any]]:
    label_entries_df = load_role_label_entries_json(context.labels_json_path)
    validation = validate_periodic_role_labels(
        observations=context.obs_df,
        label_entries=label_entries_df,
        allowed_labels=context.config.role_labels,
    )
    return label_entries_df, validation


def build_labeled_observations(
    context: WorkflowContext,
    label_entries_df: pd.DataFrame,
) -> pd.DataFrame:
    return apply_periodic_role_labels_propagated(
        observations=context.obs_df,
        label_entries=label_entries_df,
        schedule_frames=context.schedule_frames,
    )


def build_samples_from_labeled_observations(
    context: WorkflowContext,
    labeled_obs_df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, FeatureSpec, dict[str, int], pd.DataFrame]:
    attack_direction_by_team, attack_details = infer_attack_direction_by_team(labeled_obs_df)

    samples_df, teammates_tensor, teammate_mask, feature_spec = build_role_samples(
        observations_with_roles=labeled_obs_df,
        attack_direction_by_team=attack_direction_by_team,
        max_teammates=int(context.config.max_teammates),
        drop_unlabeled=True,
    )
    return (
        samples_df,
        teammates_tensor,
        teammate_mask,
        feature_spec,
        attack_direction_by_team,
        attack_details,
    )


def export_position_dataset_run(
    context: WorkflowContext,
    labeled_obs_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    teammates_tensor: np.ndarray,
    teammate_mask: np.ndarray,
    feature_spec: FeatureSpec,
    attack_direction_by_team: Mapping[str, int],
) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = context.project_root / POSITIONS_DATASET_DIR / f"{context.match_id}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_table_path = out_dir / "base_table.csv"
    samples_csv_path = out_dir / "samples_metadata_and_obj_features.csv"
    samples_npz_path = out_dir / "samples_teammates.npz"
    meta_json_path = out_dir / "dataset_meta.json"

    labeled_obs_df.to_csv(base_table_path, index=False)
    samples_df.to_csv(samples_csv_path, index=False)
    np.savez_compressed(
        samples_npz_path,
        teammates_tensor=np.asarray(teammates_tensor, dtype=np.float32),
        teammate_mask=np.asarray(teammate_mask, dtype=np.uint8),
    )

    meta = {
        "created_at": datetime.now().isoformat(),
        "match_id": context.match_id,
        "video_path": str(context.video_path),
        "tracks_path": str(context.tracks_path),
        "labels_json_path": str(context.labels_json_path),
        "label_every_seconds": float(context.config.label_every_seconds),
        "label_start_seconds": float(context.effective_label_start_seconds),
        "label_window_frames": int(context.config.label_window_frames),
        "field_length_m": float(context.config.field_length_m),
        "field_width_m": float(context.config.field_width_m),
        "max_teammates": int(context.config.max_teammates),
        "role_labels": list(context.config.role_labels),
        "objective_feature_names": list(feature_spec.objective_feature_names),
        "teammate_feature_names": list(feature_spec.teammate_feature_names),
        "attack_direction_by_team": {
            str(team): int(direction) for team, direction in attack_direction_by_team.items()
        },
        "num_base_rows": int(len(labeled_obs_df)),
        "num_labeled_rows": int(labeled_obs_df["role_label"].notna().sum()),
        "num_samples": int(len(samples_df)),
    }
    with meta_json_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    append_summary = upsert_run_to_common_dataset(
        project_root=context.project_root,
        labeled_obs_df=labeled_obs_df,
        samples_df=samples_df,
        teammates_tensor=teammates_tensor,
        teammate_mask=teammate_mask,
        feature_spec=feature_spec,
        source_info={
            "run_id": run_id,
            "match_id": context.match_id,
            "video_path": str(context.video_path),
            "tracks_path": str(context.tracks_path),
            "labels_json_path": str(context.labels_json_path),
            "num_samples": int(len(samples_df)),
        },
    )

    return {
        "run_id": run_id,
        "out_dir": out_dir,
        "base_table_path": base_table_path,
        "samples_csv_path": samples_csv_path,
        "samples_npz_path": samples_npz_path,
        "meta_json_path": meta_json_path,
        "append_summary": append_summary,
    }


class PeriodicRoleLabelerUI:
    """Editor interactivo de etiquetas periódicas por frame.

    Pensado para usarse desde Jupyter. Muestra una sola imagen: el frame actual.
    """

    def __init__(
        self,
        context: WorkflowContext,
        player_ids: Sequence[int] = DEFAULT_PLAYER_IDS,
        frame_figsize: tuple[int, int] = (18, 10),
        show_player_id: bool = True,
        show_team_id: bool = False,
    ) -> None:
        try:
            import ipywidgets as widgets
            from IPython.display import clear_output, display
        except ImportError as exc:
            raise ImportError(
                "ipywidgets no está disponible. Instálalo con `pip install ipywidgets`."
            ) from exc

        self.context = context
        self.player_ids = [int(pid) for pid in player_ids]
        self.frame_figsize = frame_figsize
        self.show_player_id = bool(show_player_id)
        self.show_team_id = bool(show_team_id)

        self._widgets = widgets
        self._display = display
        self._clear_output = clear_output

        self.schedule_frames = context.schedule_frames
        if not self.schedule_frames:
            raise ValueError("No hay frames en el schedule para etiquetar.")

        self.team_options = sorted(
            context.obs_df["team_id"].dropna().astype(str).unique().tolist()
        )
        if not self.team_options:
            raise ValueError("No se detectaron equipos en observaciones.")

        self.role_options = [""] + list(context.config.role_labels)
        self.default_team_by_player = self._compute_default_team_by_player()

        self.label_entries_df, self.validation = load_labels_and_validation(context)
        self._carry_selection_by_pid: dict[int, tuple[str, str]] | None = None

        self.frame_dropdown = widgets.Dropdown(
            options=[(str(fid), int(fid)) for fid in self.schedule_frames],
            value=int(self.schedule_frames[0]),
            description="Frame:",
            layout=widgets.Layout(width="220px"),
        )
        self.prev_btn = widgets.Button(description="⟵ Anterior", button_style="")
        self.next_btn = widgets.Button(description="Siguiente ⟶", button_style="")
        self.save_btn = widgets.Button(description="Guardar frame", button_style="success")
        self.save_next_btn = widgets.Button(
            description="Guardar y siguiente", button_style="info"
        )
        self.reload_btn = widgets.Button(description="Recargar frame", button_style="")

        self.frame_preview_title = widgets.HTML(
            value="<b>Frame preview</b>",
            layout=widgets.Layout(margin="0 0 6px 0"),
        )
        self.frame_preview_image = widgets.Image(
            value=b"",
            format="png",
            layout=widgets.Layout(width="100%", height="auto"),
        )
        self.frame_preview_box = widgets.VBox(
            [self.frame_preview_title, self.frame_preview_image],
            layout=widgets.Layout(border="1px solid #ddd", padding="8px"),
        )
        self.frame_info_out = widgets.Output(
            layout=widgets.Layout(
                border="1px solid #ddd",
                padding="8px",
                max_height="260px",
                overflow_y="auto",
            )
        )
        self.status_out = widgets.Output(
            layout=widgets.Layout(border="1px solid #ddd", padding="8px")
        )

        self.row_widgets: dict[int, dict[str, Any]] = {}
        self._build_rows()
        self._wire_callbacks()

        controls = widgets.HBox(
            [
                self.frame_dropdown,
                self.prev_btn,
                self.next_btn,
                self.save_btn,
                self.save_next_btn,
                self.reload_btn,
            ],
            layout=widgets.Layout(flex_flow="row wrap", gap="8px", align_items="center"),
        )

        self.ui = widgets.VBox(
            [controls, self.frame_preview_box, self.rows_box, self.frame_info_out, self.status_out]
        )

        self._populate_widgets_for_frame(int(self.frame_dropdown.value))
        with self.status_out:
            print("Listo. Etiqueta y pulsa 'Guardar frame' o 'Guardar y siguiente'.")
            print("Solo se muestra la imagen del frame actual.")

    def _compute_default_team_by_player(self) -> dict[int, str]:
        tmp_obs = self.context.obs_df[["player_id", "team_id"]].dropna().copy()
        tmp_obs["team_id"] = tmp_obs["team_id"].astype(str)
        defaults: dict[int, str] = {}
        for pid, group in tmp_obs.groupby("player_id"):
            defaults[int(pid)] = str(group["team_id"].mode().iloc[0])
        return defaults

    def _build_rows(self) -> None:
        widgets = self._widgets

        header = widgets.HBox(
            [
                widgets.HTML("<b>ID</b>", layout=widgets.Layout(width="60px")),
                widgets.HTML("<b>Presente</b>", layout=widgets.Layout(width="90px")),
                widgets.HTML("<b>Equipo</b>", layout=widgets.Layout(width="210px")),
                widgets.HTML("<b>Posición</b>", layout=widgets.Layout(width="220px")),
            ]
        )

        rows = []
        for pid in self.player_ids:
            id_label = widgets.HTML(f"<b>{pid}</b>", layout=widgets.Layout(width="60px"))
            present_label = widgets.HTML("", layout=widgets.Layout(width="90px"))

            default_team = self.default_team_by_player.get(pid, self.team_options[0])
            if default_team not in self.team_options:
                default_team = self.team_options[0]

            team_dd = widgets.Dropdown(
                options=self.team_options,
                value=default_team,
                layout=widgets.Layout(width="200px"),
            )
            role_dd = widgets.Dropdown(
                options=self.role_options,
                value="",
                layout=widgets.Layout(width="210px"),
            )

            self.row_widgets[pid] = {
                "present": present_label,
                "team": team_dd,
                "role": role_dd,
            }
            rows.append(
                widgets.HBox(
                    [id_label, present_label, team_dd, role_dd],
                    layout=widgets.Layout(margin="2px 0"),
                )
            )

        rows_scroll = widgets.VBox(
            rows,
            layout=widgets.Layout(
                max_height="62vh",
                overflow_y="auto",
                overflow_x="hidden",
                padding="2px 0",
            ),
        )

        self.rows_box = widgets.VBox(
            [header, rows_scroll],
            layout=widgets.Layout(border="1px solid #ddd", padding="8px"),
        )

    def _wire_callbacks(self) -> None:
        self.frame_dropdown.observe(self._on_frame_change, names="value")
        self.prev_btn.on_click(self._go_prev_frame)
        self.next_btn.on_click(self._go_next_frame)
        self.save_btn.on_click(self._on_save_click)
        self.save_next_btn.on_click(self._on_save_next_click)
        self.reload_btn.on_click(self._on_reload_click)

    def display(self):
        self._display(self.ui)
        return self.ui

    def close(self) -> None:
        # Cierra la UI previa para no acumular widgets si se re-ejecuta la celda.
        try:
            self.ui.close()
        except Exception:
            pass

    def _load_labels_for_frame(self, frame_id: int) -> tuple[dict[int, tuple[str, str]], pd.DataFrame]:
        labels_df_local = load_role_label_entries_json(self.context.labels_json_path)
        frame_labels = labels_df_local[
            (labels_df_local["frame_id"] == int(frame_id))
            & (labels_df_local["role_label"].notna())
        ].copy()
        if frame_labels.empty:
            return {}, labels_df_local

        frame_labels["team_id"] = frame_labels["team_id"].astype(str)
        by_pid: dict[int, tuple[str, str]] = {}
        for _, row in frame_labels.sort_values(["player_id", "team_id"]).iterrows():
            pid = int(row["player_id"])
            if pid not in by_pid:
                by_pid[pid] = (str(row["team_id"]), str(row["role_label"]))
        return by_pid, labels_df_local

    def _snapshot_current_selection(self) -> dict[int, tuple[str, str]]:
        snapshot: dict[int, tuple[str, str]] = {}
        for pid in self.player_ids:
            team_id = str(self.row_widgets[pid]["team"].value)
            role_label = str(self.row_widgets[pid]["role"].value)
            snapshot[int(pid)] = (team_id, role_label)
        return snapshot

    def _render_frame_preview(self, frame_id: int, frame_obs: pd.DataFrame) -> None:
        frame_players = frame_obs.sort_values(["team_id", "player_id"]).copy()
        if frame_players.empty:
            self.frame_preview_title.value = (
                f"<b>Frame {frame_id}</b> | sin jugadores detectados para previsualizar."
            )
            self.frame_preview_image.value = b""
            return

        try:
            preview_bgr = render_frame_with_player_ids(
                self.context.video_path,
                frame_players,
                int(frame_id),
                show_team_id=self.show_team_id,
                show_player_id=self.show_player_id,
            )
        except Exception as exc:
            self.frame_preview_title.value = f"<b>Frame {frame_id}</b> | error: {exc}"
            self.frame_preview_image.value = b""
            return

        ok, encoded = cv2.imencode(".png", preview_bgr)
        if not ok:
            self.frame_preview_title.value = (
                f"<b>Frame {frame_id}</b> | error al codificar preview."
            )
            self.frame_preview_image.value = b""
            return

        self.frame_preview_title.value = (
            f"<b>Frame {frame_id}</b> | jugadores+porteros detectados: {len(frame_players)}"
        )
        self.frame_preview_image.value = encoded.tobytes()

    def _populate_widgets_for_frame(
        self,
        frame_id: int,
        carry_defaults: dict[int, tuple[str, str]] | None = None,
    ) -> None:
        frame_obs = self.context.obs_df[self.context.obs_df["frame_id"] == int(frame_id)].copy()
        frame_obs["team_id"] = frame_obs["team_id"].astype(str)

        present_ids = set(frame_obs["player_id"].astype(int).tolist())
        frame_team_by_pid: dict[int, str] = {}
        for pid, group in frame_obs.groupby("player_id"):
            frame_team_by_pid[int(pid)] = str(group["team_id"].mode().iloc[0])

        labels_by_pid, labels_df_local = self._load_labels_for_frame(int(frame_id))

        for pid in self.player_ids:
            widget_row = self.row_widgets[pid]
            widget_row["present"].value = "✅" if pid in present_ids else ""

            base_team = frame_team_by_pid.get(
                pid,
                self.default_team_by_player.get(pid, self.team_options[0]),
            )
            base_role = ""

            if carry_defaults is not None and pid in carry_defaults:
                carry_team, carry_role = carry_defaults[pid]
                if carry_team in self.team_options:
                    base_team = carry_team
                if carry_role in self.role_options:
                    base_role = carry_role

            selected_team, selected_role = labels_by_pid.get(pid, (base_team, base_role))
            if selected_team not in self.team_options:
                selected_team = base_team
            if selected_role not in self.role_options:
                selected_role = base_role

            widget_row["team"].value = selected_team
            widget_row["role"].value = selected_role

        self._render_frame_preview(int(frame_id), frame_obs)

        with self.frame_info_out:
            self._clear_output(wait=True)
            print(f"Frame {frame_id} | jugadores detectados: {len(frame_obs)}")
            if not frame_obs.empty:
                self._display(
                    frame_obs[["team_id", "player_id", "class_name", "x", "y"]]
                    .sort_values(["team_id", "player_id"])
                    .reset_index(drop=True)
                )

        self.label_entries_df = labels_df_local
        self.validation = validate_periodic_role_labels(
            self.context.obs_df,
            self.label_entries_df,
            allowed_labels=self.context.config.role_labels,
        )

    def _build_frame_role_map(self, frame_id: int) -> dict[str, dict[int, str | None]]:
        labels_df_local = load_role_label_entries_json(self.context.labels_json_path)
        frame_existing = labels_df_local[labels_df_local["frame_id"] == int(frame_id)].copy()

        frame_map: dict[str, dict[int, str | None]] = {}

        for _, row in frame_existing.iterrows():
            team_id = str(row["team_id"])
            player_id = int(row["player_id"])
            frame_map.setdefault(team_id, {})[player_id] = None

        for pid in self.player_ids:
            team_id = str(self.row_widgets[pid]["team"].value)
            role_label = str(self.row_widgets[pid]["role"].value).strip()
            if role_label:
                frame_map.setdefault(team_id, {})[int(pid)] = role_label

        return frame_map

    def save_current_frame(self, go_next: bool = False) -> dict[str, Any]:
        frame_id = int(self.frame_dropdown.value)
        frame_map = self._build_frame_role_map(frame_id)

        summary = upsert_role_labels_in_template_json(
            label_json_path=self.context.labels_json_path,
            frame_role_maps={frame_id: frame_map},
        )

        self.label_entries_df, self.validation = load_labels_and_validation(self.context)

        with self.status_out:
            self._clear_output(wait=True)
            print(f"Guardado frame: {frame_id}")
            print(summary)
            print("invalid_labels:", self.validation["invalid_labels"])
            print("labeled_entries:", self.validation["labeled_entries"])

        if go_next:
            self._go_next_frame()

        return summary

    def _go_prev_frame(self, *_):
        current = int(self.frame_dropdown.value)
        idx = self.schedule_frames.index(current)
        if idx > 0:
            self.frame_dropdown.value = int(self.schedule_frames[idx - 1])

    def _go_next_frame(self, *_):
        self._carry_selection_by_pid = self._snapshot_current_selection()
        current = int(self.frame_dropdown.value)
        idx = self.schedule_frames.index(current)
        if idx < len(self.schedule_frames) - 1:
            self.frame_dropdown.value = int(self.schedule_frames[idx + 1])

    def _on_frame_change(self, change):
        if change.get("name") == "value" and change.get("new") is not None:
            carry = self._carry_selection_by_pid
            self._carry_selection_by_pid = None
            self._populate_widgets_for_frame(int(change["new"]), carry_defaults=carry)

    def _on_save_click(self, _):
        self.save_current_frame(go_next=False)

    def _on_save_next_click(self, _):
        self.save_current_frame(go_next=True)

    def _on_reload_click(self, _):
        self._populate_widgets_for_frame(int(self.frame_dropdown.value))

    def compute_labeled_observations(self) -> pd.DataFrame:
        return build_labeled_observations(self.context, self.label_entries_df)

    def export_current_labels_to_dataset(self) -> dict[str, Any]:
        labeled_obs_df = self.compute_labeled_observations()
        if labeled_obs_df["role_label"].notna().sum() == 0:
            raise ValueError("No hay etiquetas en role_label. Etiqueta y guarda antes de exportar.")

        (
            samples_df,
            teammates_tensor,
            teammate_mask,
            feature_spec,
            attack_direction_by_team,
            _attack_details,
        ) = build_samples_from_labeled_observations(self.context, labeled_obs_df)

        return export_position_dataset_run(
            context=self.context,
            labeled_obs_df=labeled_obs_df,
            samples_df=samples_df,
            teammates_tensor=teammates_tensor,
            teammate_mask=teammate_mask,
            feature_spec=feature_spec,
            attack_direction_by_team=attack_direction_by_team,
        )


def create_labeler_ui(
    config: WorkflowConfig,
    player_ids: Sequence[int] = DEFAULT_PLAYER_IDS,
    frame_figsize: tuple[int, int] = (18, 10),
) -> tuple[WorkflowContext, PeriodicRoleLabelerUI, dict[str, Any]]:
    global _ACTIVE_LABELER
    if _ACTIVE_LABELER is not None:
        _ACTIVE_LABELER.close()

    context, summary = prepare_periodic_role_workflow(config)
    ui = PeriodicRoleLabelerUI(
        context=context,
        player_ids=player_ids,
        frame_figsize=frame_figsize,
        show_player_id=True,
        show_team_id=False,
    )
    _ACTIVE_LABELER = ui
    return context, ui, summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Workflow de dataset de posiciones. "
            "Subcomando prepare para preparar schedule/JSON, "
            "subcomando export para exportar dataset usando labels guardadas."
        )
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--project-root", type=Path, default=None)
        p.add_argument("--video-filename", type=str, default=None)
        p.add_argument("--label-every-seconds", type=float, default=6.0)
        p.add_argument("--label-start-seconds", type=float, default=0.0)
        p.add_argument("--label-start-frame", type=int, default=None)
        p.add_argument("--label-min-players", type=int, default=18)
        p.add_argument("--label-window-frames", type=int, default=20)
        p.add_argument("--max-teammates", type=int, default=10)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Prepara observaciones, schedule y plantilla de labels JSON.",
    )
    add_common_args(prepare_parser)

    export_parser = subparsers.add_parser(
        "export",
        help="Exporta dataset (run + common) desde labels guardadas.",
    )
    add_common_args(export_parser)

    return parser.parse_args(argv)


def _build_config_from_args(args: argparse.Namespace) -> WorkflowConfig:
    return WorkflowConfig(
        project_root=args.project_root,
        video_filename=args.video_filename,
        max_teammates=int(args.max_teammates),
        label_every_seconds=float(args.label_every_seconds),
        label_start_seconds=float(args.label_start_seconds),
        label_start_frame=args.label_start_frame,
        label_min_players=int(args.label_min_players),
        label_window_frames=int(args.label_window_frames),
    )


def _run_prepare(config: WorkflowConfig) -> int:
    _context, summary = prepare_periodic_role_workflow(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nSiguiente paso: en Jupyter crea la UI con create_labeler_ui(config).")
    return 0


def _run_export(config: WorkflowConfig) -> int:
    context, summary = prepare_periodic_role_workflow(config)
    print("Preparación:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    label_entries_df, validation = load_labels_and_validation(context)
    labeled_obs_df = build_labeled_observations(context, label_entries_df)
    if labeled_obs_df["role_label"].notna().sum() == 0:
        raise ValueError(
            "No hay etiquetas en role_label dentro del JSON. "
            "Etiqueta primero y vuelve a ejecutar export."
        )

    (
        samples_df,
        teammates_tensor,
        teammate_mask,
        feature_spec,
        attack_direction_by_team,
        _attack_details,
    ) = build_samples_from_labeled_observations(context, labeled_obs_df)

    export_summary = export_position_dataset_run(
        context=context,
        labeled_obs_df=labeled_obs_df,
        samples_df=samples_df,
        teammates_tensor=teammates_tensor,
        teammate_mask=teammate_mask,
        feature_spec=feature_spec,
        attack_direction_by_team=attack_direction_by_team,
    )
    print("\nExportación completada:")
    print(
        json.dumps(
            {
                "out_dir": str(export_summary["out_dir"]),
                "base_table_path": str(export_summary["base_table_path"]),
                "samples_csv_path": str(export_summary["samples_csv_path"]),
                "samples_npz_path": str(export_summary["samples_npz_path"]),
                "meta_json_path": str(export_summary["meta_json_path"]),
                "append_summary": {
                    k: (str(v) if isinstance(v, Path) else v)
                    for k, v in export_summary["append_summary"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("\nValidación etiquetas:")
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _build_config_from_args(args)
    if args.command == "prepare":
        return _run_prepare(config)
    if args.command == "export":
        return _run_export(config)
    raise ValueError(f"Comando no soportado: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
