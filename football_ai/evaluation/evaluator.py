import numpy as np
from collections import defaultdict, Counter


class Evaluator:
    def __init__(self):
        pass

    # ----------------------------------------------
    # UTILS
    # ----------------------------------------------
    def create_default_dicts(self, num_dicts):
        return [defaultdict(list) for _ in range(num_dicts)]
    
    def init_dicts(self, frames, id_frames, id_teams, id_colors, id_bboxes, id_conf, id_bbox_sizes):
        for t, frame_dict in enumerate(frames):
            for tid, info in frame_dict.items():
                id_frames[tid].append(t)
                id_teams[tid].append(info['team'])
                id_colors[tid].append(info['shirt_color'])
                id_bboxes[tid].append(info['bbox'])
                id_conf[tid].append(info['confidence'])
                id_bbox_sizes[tid].append(info['bbox_size'])

    # ----------------------------------------------
    # SPEED + GAPS
    # ----------------------------------------------
    def get_gaps_and_speed(self, tid, frames_seen_sorted, id_bboxes):
        gap_lengths = []
        speed_values = []
        speed_frames = []    # list of (frame_prev, frame_curr)

        prev = frames_seen_sorted[0]
        bbox_prev = id_bboxes[tid][0]

        for n_frame, f in enumerate(frames_seen_sorted[1:]):
            frame_gap = f - prev - 1
            if f != prev + 1:
                gap_lengths.append(frame_gap)

            # n_frame starts at 0 for the second observed frame, so we need +1
            # to align with frames_seen_sorted[1:].
            bbox_current = id_bboxes[tid][n_frame + 1]
            x1, y1, _, _ = bbox_current
            x1_prev, y1_prev, _, _ = bbox_prev

            dx = (x1 - x1_prev) / (frame_gap + 1)
            dy = (y1 - y1_prev) / (frame_gap + 1)

            speed = float(np.sqrt(dx * dx + dy * dy))

            speed_values.append(speed)
            speed_frames.append((prev, f))

            prev = f
            bbox_prev = bbox_current
        
        speeds_np = np.array(speed_values) if len(speed_values) > 0 else np.array([])

        gaps = len(gap_lengths)
        mean_gap_len = float(np.mean(gap_lengths)) if gap_lengths else 0.0
        mean_speed = float(np.mean(speeds_np)) if speeds_np.size > 0 else 0.0
        max_speed = float(np.max(speeds_np)) if speeds_np.size > 0 else 0.0

        return gaps, mean_gap_len, mean_speed, max_speed, speed_values, speed_frames

    # ----------------------------------------------
    # TEAM + ENTROPY
    # ----------------------------------------------
    def get_flips_and_entropy(self, tid, id_teams):
        teams = id_teams.get(tid, [])
        team_mode = None
        flip_rate_static = 0.0
        flip_rate_dynamic = 0.0
        entropy = 0.0

        if teams:
            c = Counter(teams)
            team_mode, _ = c.most_common(1)[0]

            flips_static = sum(1 for x in teams if x != team_mode)
            flips_dynamic_bool = [teams[i] != teams[i - 1] for i in range(1, len(teams))]
            flips_dynamic = sum(teams[i] != teams[i - 1] for i in range(1, len(teams)))

            flip_rate_static = flips_static / len(teams)
            flip_rate_dynamic = flips_dynamic / (len(teams) - 1) if len(teams) > 1 else 0.0

            ps = np.array(list(c.values())) / sum(c.values())
            entropy = float(-np.sum(ps * np.log2(ps + 1e-12)))

        return team_mode, flips_dynamic_bool, flip_rate_static, flip_rate_dynamic, entropy

    # ----------------------------------------------
    # COLOR METRICS
    # ----------------------------------------------
    def get_colors_var(self, tid, id_colors):
        colors = np.array(id_colors.get(tid, []), dtype=float) if id_colors.get(tid) else None

        if colors is None or len(colors) == 0:
            return None, None

        color_var = float(np.mean(np.var(colors, axis=0)))
        color_diff = float(np.mean(np.linalg.norm(np.diff(colors, axis=0), axis=1))) if len(colors) > 1 else 0.0

        return color_var, color_diff

    # ----------------------------------------------
    # BBOX SIZE CONSISTENCY
    # ----------------------------------------------
    def get_bbox_size(self, tid, id_bbox_sizes):
        sizes = np.array(id_bbox_sizes.get(tid, []), dtype=float) if id_bbox_sizes.get(tid) else None

        if sizes is None or len(sizes) <= 1:
            return None, None
        
        return sizes, float(np.std(sizes) / np.mean(sizes))

    # ----------------------------------------------
    # METRICS FOR ONE TRACK
    # ----------------------------------------------
    def metrics_of_tid(self, T, tid, frames_seen, id_bboxes, id_teams, id_colors, id_bbox_sizes, id_conf):
        frames_seen_sorted = sorted(frames_seen)
        total_seen = len(frames_seen_sorted)
        coverage = total_seen / T

        start_frame = frames_seen_sorted[0]
        end_frame = frames_seen_sorted[-1]
        mid_frame = (start_frame + end_frame) // 2

        gaps, mean_gap_len, mean_speed, max_speed, speed_values, speed_frames = \
            self.get_gaps_and_speed(tid, frames_seen_sorted, id_bboxes)

        team_mode, flips_dynamic_bool, flip_rate_static, flip_rate_dynamic, entropy = \
            self.get_flips_and_entropy(tid, id_teams)

        color_var, color_diff = self.get_colors_var(tid, id_colors)

        sizes, size_cv = self.get_bbox_size(tid, id_bbox_sizes)

        mean_conf = float(np.mean(id_conf[tid])) if id_conf.get(tid) else None

        # -------- metric_events as LISTS of events per metric --------
        metrics_events = {}
        metrics_list = { "frames_seen": frames_seen_sorted,
                        "speed_values": speed_values, "speed_frames": speed_frames,
                        "flips": flips_dynamic_bool, "bbox_size": sizes, "confs": id_conf[tid] }

        def add_metric_event(key, value, frame=mid_frame):
            if value is None:
                event = {"id": int(tid), "frame": int(frame), "value": None}
            else:
                event = {"id": int(tid), "frame": int(frame), "value": float(value)}
            metrics_events.setdefault(key, []).append(event)

        # aggregated metrics (1 per track)
        add_metric_event("coverage", coverage)
        add_metric_event("total_seen", total_seen)
        add_metric_event("fragments", gaps)
        add_metric_event("mean_gap_len", mean_gap_len)
        add_metric_event("mean_speed", mean_speed)
        add_metric_event("max_speed", max_speed)
        add_metric_event("num_speed_samples", len(speed_values))
        add_metric_event("team_flip_rate_static", flip_rate_static)
        add_metric_event("team_flip_rate_dynamic", flip_rate_dynamic)
        add_metric_event("entropy", entropy)
        add_metric_event("color_var", color_var)
        add_metric_event("color_diff", color_diff)
        add_metric_event("bbox_size_cv", size_cv)
        add_metric_event("mean_confidence", mean_conf)

        metrics_events["speed_events"] =  list(zip(speed_frames, speed_values))        # for compatibility
        metrics_events["team_mode"] =  team_mode

        # speed metric: one event per (frame_prev, frame_curr)
        speed_metric_events = []
        for (frame_prev, frame_next), speed in zip(speed_frames, speed_values):
            speed_metric_events.append({
                "id": int(tid),
                "frame": int(frame_next),
                "frame_prev": int(frame_prev),
                "value": float(speed)
            })
        if speed_metric_events:
            metrics_events["speed"] = speed_metric_events
        
        return metrics_events, metrics_list

    # ----------------------------------------------
    # MAIN EVALUATION
    # ----------------------------------------------
    def evaluate_class(self, tracks, class_name="player"):
        frames = tracks[class_name]
        T = len(frames)

        id_frames, id_teams, id_colors, id_bboxes, id_conf, id_bbox_sizes = \
            self.create_default_dicts(6)

        self.init_dicts(frames, id_frames, id_teams, id_colors,
                       id_bboxes, id_conf, id_bbox_sizes)

        metrics = {}
        metrics_list = {}
        num_tracks = 0

        for tid, frames_seen in id_frames.items():
            if int(tid) > num_tracks:
                num_tracks = int(tid)

            metrics_tid, metrics_list_tid = self.metrics_of_tid(
                T, tid, frames_seen,
                id_bboxes, id_teams, id_colors,
                id_bbox_sizes, id_conf
            )
            metrics[tid] = metrics_tid
            metrics_list[tid] = metrics_list_tid

        # -------- global summary --------
        covs = [m["coverage"][0]["value"] for m in metrics.values()]
        total_seen = [m["total_seen"][0]["value"] for m in metrics.values()]
        frag = [m["fragments"][0]["value"] for m in metrics.values()]
        mean_gap_len = [m["mean_gap_len"][0]["value"] for m in metrics.values()]

        total_speed_samples = sum(m["num_speed_samples"][0]["value"] for m in metrics.values())
        mean_speed = (
            sum(m["mean_speed"][0]["value"] * m["num_speed_samples"][0]["value"] for m in metrics.values()) / total_speed_samples
            if total_speed_samples > 0 else 0.0
        )
        speed_list = [m["mean_speed"][0]["value"] for m in metrics.values()]
        max_speed_list = [m["max_speed"][0]["value"] for m in metrics.values()]
        
        flip_rates_static = [
            m["team_flip_rate_static"][0]["value"] for m in metrics.values()
            if m["team_flip_rate_static"][0]["value"] is not None
        ]
        flip_rates_dynamic = [
            m["team_flip_rate_dynamic"][0]["value"] for m in metrics.values()
            if m["team_flip_rate_dynamic"][0]["value"] is not None
        ]
        entropy = [m["entropy"][0]["value"] for m in metrics.values()]
        color_var = [m["color_var"][0]["value"] for m in metrics.values()]
        color_diff = [m["color_diff"][0]["value"] for m in metrics.values()]

        bbox_size = [
            m["bbox_size_cv"][0]["value"] for m in metrics.values()
            if m["bbox_size_cv"][0]["value"] is not None
        ]
        confidence = [
            m["mean_confidence"][0]["value"] for m in metrics.values()
            if m["mean_confidence"][0]["value"] is not None
        ]

        summary = {
            "n_tracks": num_tracks,
            "num_tracks": len(metrics),

            "mean_coverage": float(np.mean(covs)) if covs else 0.0,
            "median_coverage": float(np.median(covs)) if covs else 0.0,
            "25_coverage": float(np.quantile(covs, 0.25)) if covs else 0.0,

            "mean_fragments": float(np.mean(frag)) if frag else 0.0,
            "median_fragments": float(np.median(frag)) if frag else 0.0,

            "mean_gap_len_mean": float(np.mean(mean_gap_len)) if mean_gap_len else 0.0,
            "median_gap_len_mean": float(np.median(mean_gap_len)) if mean_gap_len else 0.0,

            "pct_tracks_with_fragments": float(
                np.mean([1 if f > 0 else 0 for f in frag])
            ) if frag else 0.0,

            "mean_speed": float(mean_speed),
            "mean_mean_speed": float(np.mean(speed_list)if speed_list else 0.0),
            "max_speed": float(np.max(max_speed_list) if speed_list else 0.0),
            "mean_max_speed": float(np.mean(max_speed_list)if speed_list else 0.0),

            "mean_team_flip_rate": float(np.mean(flip_rates_static)) if flip_rates_static else 0.0,
            "mean_team_flip_dynamic": float(np.mean(flip_rates_dynamic)) if flip_rates_dynamic else 0.0,

            "entropy": float(np.mean(entropy)) if entropy else 0.0,
            "color_var": float(np.mean(color_var)) if color_var else 0.0,
            "color_diff": float(np.mean(color_diff)) if color_diff else 0.0,
            "bbox_size": float(np.mean(bbox_size)) if bbox_size else 0.0,
            "confidence": float(np.mean(confidence)) if confidence else 0.0,
        }
        
        return metrics, metrics_list, summary, T
    
    def evaluate(self, classes, tracks):
        evaluation = {}
        for class_name in classes:
            metrics, metrics_list, summary, n_frames = self.evaluate_class(tracks, class_name)
            evaluation[class_name] = {
                "metrics": metrics,
                "metrics_list": metrics_list,
                "summary": summary,
                "n_frames": n_frames
            }
        return evaluation
