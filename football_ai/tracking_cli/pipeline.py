import json
from pathlib import Path

from football_ai.core import Logger, get_config, get_logger
from football_ai.evaluation import Evaluator
from football_ai.positions import (
    OnlineSpecialSeedRoleAssigner,
    build_role_artifacts_output_dir,
    build_role_predictions_output_paths,
    copy_output_artifact,
    save_dataframe_csv,
    save_role_visualizations,
)
from football_ai.tracking import Tracker
from football_ai.visualization import Drawer

from .paths import (
    build_output_video_path,
    build_tracking_metrics_output_paths,
    build_tracks_output_paths,
    resolve_video_path,
)
from .persistence import save_result, save_summary, upsert_tracking_metrics_dataset

def run_tracking_pipeline(args):
    # Load configuration
    config = get_config()

    # Configure logging
    Logger.setup_from_config(config)
    logger = get_logger(__name__)

    logger.info("Starting football tracking system")

    try:
        # Get paths and parameters from config
        model_path = str(config.get_path("paths", "models", "modelo_base"))
        video_path, video_source = resolve_video_path(config, args.video_shortcut)
        
        output = build_output_video_path(config, video_path)
        output_path_named, output_path_legacy = build_tracks_output_paths(config, video_path)
        summary_path, metrics_dataset_path = build_tracking_metrics_output_paths(config, video_path)
        
        role_artifacts_dir = build_role_artifacts_output_dir(config, video_path)
        role_frame_csv_path, role_player_csv_path, role_greedy_csv_path = build_role_predictions_output_paths(config, video_path, use_artifacts_dir=True)
        role_frame_csv_path_legacy, role_player_csv_path_legacy, role_greedy_csv_path_legacy = build_role_predictions_output_paths(config, video_path, use_artifacts_dir=False)

        # Configuration parameters
        visualization_conf = config.visualization
        show_kmeans = config.get("visualization", "show_kmeans")
        show_output = config.get("visualization", "show_output")
        four_panel_enabled = bool(visualization_conf.get("four_panel_enabled", False))

        # Tracking configuration
        detector_conf = config.detection
        team_detector_conf = config.team_detector
        projector_conf = config.projector
        bytetracker_conf = config.bytetracker
        ball_conf = config.ball

        tracker_conf = config.tracking

        logger.info(f"Model: {model_path}")
        logger.info(f"Video: {video_path}")
        logger.info(f"Video source: {video_source}")
        logger.info(f"Named tracks JSON output: {output_path_named}")
        logger.info(f"Legacy tracks JSON output: {output_path_legacy}")
        logger.info(f"Summary JSON output: {summary_path}")
        logger.info(f"Role artifacts directory: {role_artifacts_dir}")
        logger.info(f"Frame role CSV output: {role_frame_csv_path}")
        logger.info(f"Player role CSV output: {role_player_csv_path}")
        logger.info(f"Greedy role CSV output: {role_greedy_csv_path}")
        logger.info(f"Tracking metrics dataset CSV: {metrics_dataset_path}")
        logger.info(f"Tracking configuration: {tracker_conf}")
        logger.info(f"Field tracking configuration: {projector_conf}")
        logger.info(f"Team detector configuration: {team_detector_conf}")

        # Run tracking
        tracker = Tracker(model_path, detector_conf, team_detector_conf, bytetracker_conf,
                          ball_conf, tracker_conf, projector_conf, config.project_root)
        
        online_special_seed_role_assigner = OnlineSpecialSeedRoleAssigner(config, video_path, logger)

        logger.info("Extracting tracks from video...")
        
        tracks = tracker.get_tracks(
            video_path,
            show_kmeans,
            frame_hook=online_special_seed_role_assigner.on_frame,
            collect_visual_debug=four_panel_enabled,
        )
        
        role_postprocess_result = online_special_seed_role_assigner.summary() if online_special_seed_role_assigner.enabled else None
        if role_postprocess_result is not None:
            logger.info(
                "Online position-role assignment applied: %s player predictions, %s frame predictions",
                role_postprocess_result.get("position_role_player_predictions"),
                role_postprocess_result.get("position_role_frame_predictions"),
            )

        # Draw tracks
        colors = config.get_visualization_colors()
        drawer = Drawer(colors=colors, visualization_conf=visualization_conf)
        logger.info("Drawing tracks on video...")
        drawer.draw_tracks(
            tracks,
            video_path,
            output,
            show=show_output,
            four_panel=four_panel_enabled,
            debug_frames=(tracker.visualization_debug_frames if four_panel_enabled else None),
            expected_counts=tracker.max_tracks_per_class,
        )
        logger.info(f"Video with tracks saved to: {output}")

        # Evaluation
        logger.info("Evaluating tracks...")
        evaluator = Evaluator()
        evaluation = evaluator.evaluate(["player", "ball"], tracks)
        summary = dict(evaluation["player"]["summary"])
        ball_summary = evaluation.get("ball", {}).get("summary", {})
        summary["ball_coverage"] = float(ball_summary.get("mean_coverage", 0.0))
        summary["ball_tracks"] = int(ball_summary.get("num_tracks", 0))
        if role_postprocess_result is not None:
            summary["position_role_online_applied"] = True
            for key, value in role_postprocess_result.items():
                summary[f"special_seed_{key}"] = value

        # Save tracks JSON
        save_result(tracks, output_path_named, logger)
        if output_path_legacy != output_path_named:
            save_result(tracks, output_path_legacy, logger)
        if online_special_seed_role_assigner.enabled:
            role_frame_df, role_player_df, role_greedy_df = online_special_seed_role_assigner.build_role_export_dataframes(tracks)
            save_dataframe_csv(role_frame_df, role_frame_csv_path, logger, "Frame role predictions CSV")
            copy_output_artifact(role_frame_csv_path, role_frame_csv_path_legacy, logger, "Frame role predictions CSV")
            save_dataframe_csv(role_player_df, role_player_csv_path, logger, "Player role summary CSV")
            copy_output_artifact(role_player_csv_path, role_player_csv_path_legacy, logger, "Player role summary CSV")
            save_dataframe_csv(role_greedy_df, role_greedy_csv_path, logger, "Greedy role diagnostics CSV")
            copy_output_artifact(role_greedy_csv_path, role_greedy_csv_path_legacy, logger, "Greedy role diagnostics CSV")
            save_role_visualizations(
                frame_df=role_frame_df,
                player_df=role_player_df,
                config=config,
                video_path=video_path,
                output_dir=role_artifacts_dir,
                expected_roles_by_team=online_special_seed_role_assigner.expected_roles_by_team,
                assignment_method=online_special_seed_role_assigner.role_stabilization_expected_roles_assignment,
                min_count=online_special_seed_role_assigner.role_stabilization_expected_roles_min_count,
                min_cumulative_ratio=online_special_seed_role_assigner.role_stabilization_expected_roles_min_ratio,
                min_final_ratio=online_special_seed_role_assigner.role_stabilization_expected_roles_min_final_ratio,
                logger=logger,
                legacy_output_dir=Path(role_frame_csv_path_legacy).parent,
            )
        save_summary(summary, summary_path, logger)
        upsert_tracking_metrics_dataset(metrics_dataset_path, video_path, video_source, output_path_named, summary_path, summary, logger)

        # Show summary
        logger.info("Evaluation summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        print("\n=== TRACKING SUMMARY ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        return 1

    logger.info("Process completed successfully")
    return 0
