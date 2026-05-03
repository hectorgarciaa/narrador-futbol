from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.actions_incremental.detector import ActionsDetector, ActionsDetectorConfig  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test sintetico para comprobar que fixed_lag_frames afecta realmente a la señal incremental.",
    )
    parser.add_argument("--lag-a", type=int, default=4, help="Primer lag a comparar.")
    parser.add_argument("--lag-b", type=int, default=16, help="Segundo lag a comparar.")
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Ruta opcional a un Markdown con el resumen del test.",
    )
    return parser


def synthetic_packet(frame_id: int) -> dict[str, object]:
    if frame_id < 10:
        home_x = 12.0
    elif frame_id < 20:
        home_x = 42.0
    else:
        home_x = 18.0
    return {
        "tracks_frame": {
            "player": {},
            "goalkeeper": {
                "1": {
                    "field_position_m": [home_x, 34.0],
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "team": "home",
                },
                "2": {
                    "field_position_m": [95.0, 34.0],
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "team": "away",
                },
            },
            "referee": {},
            "ball": {},
        },
        "possession": {},
    }


def run_detector_for_lag(lag_frames: int) -> pd.DataFrame:
    detector = ActionsDetector(
        ActionsDetectorConfig(
            fixed_lag_frames=int(lag_frames),
            smoothing_window=9,
            enable_local_savgol=False,
            alpha_beta_enabled=True,
            enable_measurement_ewm=True,
        )
    )
    for frame_id in range(30):
        detector.update(frame_id, synthetic_packet(frame_id))
    debug_df = detector.build_slot_motion_debug_dataframe()
    return debug_df[debug_df["slot"] == "home_1"].reset_index(drop=True)


def main() -> None:
    args = build_parser().parse_args()
    lag_a_df = run_detector_for_lag(args.lag_a)
    lag_b_df = run_detector_for_lag(args.lag_b)
    merged = lag_a_df[["frame_id", "x", "vx", "accel"]].merge(
        lag_b_df[["frame_id", "x", "vx", "accel"]],
        on="frame_id",
        suffixes=(f"_lag{args.lag_a}", f"_lag{args.lag_b}"),
    )
    merged["abs_x_diff"] = np.abs(merged[f"x_lag{args.lag_a}"] - merged[f"x_lag{args.lag_b}"])
    merged["abs_vx_diff"] = np.abs(merged[f"vx_lag{args.lag_a}"] - merged[f"vx_lag{args.lag_b}"])
    merged["abs_accel_diff"] = np.abs(merged[f"accel_lag{args.lag_a}"] - merged[f"accel_lag{args.lag_b}"])
    max_x_diff = float(merged["abs_x_diff"].max())
    max_vx_diff = float(merged["abs_vx_diff"].max())
    max_accel_diff = float(merged["abs_accel_diff"].max())
    changed = bool(max(max_x_diff, max_vx_diff, max_accel_diff) > 1e-6)
    payload = {
        "lag_a": int(args.lag_a),
        "lag_b": int(args.lag_b),
        "changed": changed,
        "max_abs_x_diff": max_x_diff,
        "max_abs_vx_diff": max_vx_diff,
        "max_abs_accel_diff": max_accel_diff,
        "focus_window": merged[(merged["frame_id"] >= 8) & (merged["frame_id"] <= 22)].to_dict(orient="records"),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_report is not None:
        lines = [
            "# Fixed lag debug report",
            "",
            f"- lag A: `{args.lag_a}`",
            f"- lag B: `{args.lag_b}`",
            f"- changed: `{changed}`",
            f"- max_abs_x_diff: `{max_x_diff:.6f}`",
            f"- max_abs_vx_diff: `{max_vx_diff:.6f}`",
            f"- max_abs_accel_diff: `{max_accel_diff:.6f}`",
            "",
            "## Ventana 8..22",
            "",
            "| frame | x_a | x_b | vx_a | vx_b | accel_a | accel_b |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        window_df = merged[(merged["frame_id"] >= 8) & (merged["frame_id"] <= 22)]
        for row in window_df.to_dict(orient="records"):
            lines.append(
                f"| {int(row['frame_id'])} | {row[f'x_lag{args.lag_a}']:.4f} | {row[f'x_lag{args.lag_b}']:.4f} | {row[f'vx_lag{args.lag_a}']:.4f} | {row[f'vx_lag{args.lag_b}']:.4f} | {row[f'accel_lag{args.lag_a}']:.4f} | {row[f'accel_lag{args.lag_b}']:.4f} |"
            )
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not changed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
