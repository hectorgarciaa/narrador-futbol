from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .core import FeatureSpec, POSITIONS_COMMON_DIR


def upsert_run_to_common_dataset(
    project_root: Path,
    labeled_obs_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    teammates_tensor: np.ndarray,
    teammate_mask: np.ndarray,
    feature_spec: FeatureSpec,
    source_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if "match_id" not in labeled_obs_df.columns or "match_id" not in samples_df.columns:
        return append_run_to_common_dataset(project_root, labeled_obs_df, samples_df, teammates_tensor, teammate_mask, feature_spec, source_info)

    paths = _common_dataset_paths(project_root)
    new_matches = sorted({str(v) for v in samples_df["match_id"].dropna().astype(str).unique().tolist()})
    if not new_matches:
        return append_run_to_common_dataset(project_root, labeled_obs_df, samples_df, teammates_tensor, teammate_mask, feature_spec, source_info)

    existing_base = pd.read_csv(paths["base_table_path"]) if paths["base_table_path"].exists() else pd.DataFrame()
    if "match_id" in existing_base.columns:
        existing_base = existing_base[~existing_base["match_id"].astype(str).isin(new_matches)].copy()
    combined_base = pd.concat([existing_base, labeled_obs_df], ignore_index=True, sort=False)
    combined_base.to_csv(paths["base_table_path"], index=False)

    combined_samples, combined_tensor, combined_mask = _merge_sample_store(
        samples_path=paths["samples_csv_path"],
        npz_path=paths["samples_npz_path"],
        incoming_samples=samples_df,
        incoming_tensor=teammates_tensor,
        incoming_mask=teammate_mask,
        replace_matches=new_matches,
    )
    _append_source_record(paths["sources_log_path"], {**dict(source_info or {}), "upserted_at": datetime.now().isoformat(), "upsert_matches": new_matches})
    _write_dataset_meta(paths["meta_json_path"], feature_spec, combined_base, combined_samples, combined_tensor)
    return {
        **paths,
        "num_base_rows": int(len(combined_base)),
        "num_samples": int(len(combined_samples)),
        "upsert_matches": new_matches,
    }


def append_run_to_common_dataset(
    project_root: Path,
    labeled_obs_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    teammates_tensor: np.ndarray,
    teammate_mask: np.ndarray,
    feature_spec: FeatureSpec,
    source_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    paths = _common_dataset_paths(project_root)
    existing_base = pd.read_csv(paths["base_table_path"]) if paths["base_table_path"].exists() else pd.DataFrame()
    combined_base = pd.concat([existing_base, labeled_obs_df], ignore_index=True, sort=False)
    combined_base.to_csv(paths["base_table_path"], index=False)
    combined_samples, combined_tensor, _ = _merge_sample_store(
        samples_path=paths["samples_csv_path"],
        npz_path=paths["samples_npz_path"],
        incoming_samples=samples_df,
        incoming_tensor=teammates_tensor,
        incoming_mask=teammate_mask,
        replace_matches=None,
    )
    _append_source_record(paths["sources_log_path"], {**dict(source_info or {}), "appended_at": datetime.now().isoformat()})
    _write_dataset_meta(paths["meta_json_path"], feature_spec, combined_base, combined_samples, combined_tensor)
    return {
        **paths,
        "num_base_rows": int(len(combined_base)),
        "num_samples": int(len(combined_samples)),
    }


def _merge_sample_store(
    samples_path: Path,
    npz_path: Path,
    incoming_samples: pd.DataFrame,
    incoming_tensor: np.ndarray,
    incoming_mask: np.ndarray,
    replace_matches: list[str] | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    new_tensor = np.asarray(incoming_tensor, dtype=np.float32)
    new_mask = np.asarray(incoming_mask, dtype=np.uint8)
    if len(incoming_samples) != new_tensor.shape[0] or len(incoming_samples) != new_mask.shape[0]:
        raise ValueError("Muestras y tensores no tienen la misma longitud.")

    existing_samples = pd.read_csv(samples_path) if samples_path.exists() else pd.DataFrame()
    if npz_path.exists():
        with np.load(npz_path) as old_npz:
            old_tensor = np.asarray(old_npz["teammates_tensor"], dtype=np.float32)
            old_mask = np.asarray(old_npz["teammate_mask"], dtype=np.uint8)
    else:
        old_tensor = np.zeros((0, new_tensor.shape[1], new_tensor.shape[2]), dtype=np.float32)
        old_mask = np.zeros((0, new_mask.shape[1]), dtype=np.uint8)

    if replace_matches and len(existing_samples) > 0 and "match_id" in existing_samples.columns:
        keep_mask = ~existing_samples["match_id"].astype(str).isin(replace_matches)
        kept_samples = existing_samples.loc[keep_mask].reset_index(drop=True)
        kept_tensor = old_tensor[np.asarray(keep_mask, dtype=bool)]
        kept_mask = old_mask[np.asarray(keep_mask, dtype=bool)]
    else:
        kept_samples = existing_samples
        kept_tensor = old_tensor
        kept_mask = old_mask

    combined_samples = pd.concat([kept_samples, incoming_samples], ignore_index=True, sort=False)
    combined_samples.to_csv(samples_path, index=False)
    combined_tensor = new_tensor if kept_tensor.shape[0] == 0 else np.concatenate([kept_tensor, new_tensor], axis=0)
    combined_mask = new_mask if kept_mask.shape[0] == 0 else np.concatenate([kept_mask, new_mask], axis=0)
    np.savez_compressed(npz_path, teammates_tensor=combined_tensor.astype(np.float32), teammate_mask=combined_mask.astype(np.uint8))
    return combined_samples, combined_tensor, combined_mask


def _write_dataset_meta(
    meta_json_path: Path,
    feature_spec: FeatureSpec,
    base_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    teammates_tensor: np.ndarray,
) -> None:
    previous_meta = {}
    if meta_json_path.exists():
        try:
            with meta_json_path.open("r", encoding="utf-8") as f:
                previous_meta = json.load(f)
        except Exception:
            previous_meta = {}
    meta = {
        "updated_at": datetime.now().isoformat(),
        "num_runs": int(previous_meta.get("num_runs", 0)) + 1,
        "num_base_rows": int(len(base_df)),
        "num_samples": int(len(samples_df)),
        "num_matches": int(samples_df["match_id"].nunique()) if "match_id" in samples_df.columns and len(samples_df) > 0 else 0,
        "max_teammates": int(teammates_tensor.shape[1]) if teammates_tensor.ndim == 3 else 0,
        "teammate_feature_dim": int(teammates_tensor.shape[2]) if teammates_tensor.ndim == 3 else 0,
        "objective_feature_names": list(feature_spec.objective_feature_names),
        "teammate_feature_names": list(feature_spec.teammate_feature_names),
        "sources_log_path": str(meta_json_path.with_name("sources.jsonl")),
    }
    with meta_json_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _append_source_record(sources_log_path: Path, payload: Mapping[str, Any]) -> None:
    with sources_log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(payload), ensure_ascii=False, default=str) + "\n")


def _common_dataset_paths(project_root: Path) -> dict[str, Path]:
    common_dir = project_root / POSITIONS_COMMON_DIR
    common_dir.mkdir(parents=True, exist_ok=True)
    return {
        "common_dir": common_dir,
        "base_table_path": common_dir / "base_table.csv",
        "samples_csv_path": common_dir / "samples_metadata_and_obj_features.csv",
        "samples_npz_path": common_dir / "samples_teammates.npz",
        "meta_json_path": common_dir / "dataset_meta.json",
        "sources_log_path": common_dir / "sources.jsonl",
    }
