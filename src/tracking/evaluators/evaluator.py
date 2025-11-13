import numpy as np
from collections import defaultdict, Counter

class Evaluator:
    def __init__(self):
        pass

    def createDefaultDicts(self, num_dicts):
        return [defaultdict(list) for _ in range(num_dicts)]
    
    def initDicts(self, frames, id_frames, id_teams, id_colors, id_bboxes, id_conf, id_bbox_sizes):
        for t, frame_dict in enumerate(frames):
            for tid, info in frame_dict.items():
                id_frames[tid].append(t)
                id_teams[tid].append(info['team']), id_colors[tid].append(info['shirt_color']), id_bboxes[tid].append(info['bbox']), id_conf[tid].append(info['confidence']), id_bbox_sizes[tid].append(info['bbox_size'])

    def getGapsAndSpeed(self, tid, frames_seen_sorted, id_bboxes):
        gap_lengths = []
        movimiento_x = []
        movimiento_y = []
        prev = frames_seen_sorted[0]
        bbox_prev = id_bboxes[tid][0]
        for n_frame, f in enumerate(frames_seen_sorted[1:]):
            frame_gap = f - prev - 1
            if f != prev +1 :
                gap_lengths.append(frame_gap)

            bbox_current = id_bboxes[tid][n_frame]
            x1, y1, _, _ = bbox_current
            x1_prev, y1_prev, _, _ = bbox_prev

            dx, dy = (x1 - x1_prev) / (frame_gap+1), (y1 - y1_prev) / (frame_gap+1)

            movimiento_x.append(dx),  movimiento_y.append(dy)

            prev = f
            bbox_prev = bbox_current

        gaps = len(gap_lengths)
        mean_gap_len = np.mean(gap_lengths) if gap_lengths else 0.0
        speed = np.sqrt(np.array(movimiento_x)**2 + np.array(movimiento_y)**2) if movimiento_x else 0.0
        mean_speed = np.mean(np.sqrt(np.array(movimiento_x)**2 + np.array(movimiento_y)**2)) if movimiento_x else 0.0
        max_speed = np.max(np.sqrt(np.array(movimiento_x)**2 + np.array(movimiento_y)**2)) if movimiento_x else 0.0

        return gaps, mean_gap_len, speed, mean_speed, max_speed
    
    def getFlipsAndEntropy(self, tid, id_teams):
        teams = id_teams.get(tid, [])
        team_mode = None
        flip_rate_static, flip_rate_dynamic, entropy = 0.0, 0.0, 0.0
        if teams:
            c = Counter(teams)
            team_mode, _ = c.most_common(1)[0]
            flips_static = sum(1 for x in teams if x != team_mode)
            flips_dynamic_bool = [teams[i] != teams[i - 1] for i in range(1, len(teams))]
            flips_dynamic = sum(teams[i] != teams[i - 1] for i in range(1, len(teams)))
            flip_rate_static = flips_static / len(teams) if teams else None
            flip_rate_dynamic = flips_dynamic / (len(teams) - 1) if len(teams) > 1 else None

            # entropy
            ps = np.array(list(c.values()))/sum(c.values())
            entropy = -np.sum(ps * np.log2(ps + 1e-12))

        return team_mode, flips_dynamic_bool, flip_rate_static, flip_rate_dynamic, entropy
    
    def getColorsVar(self, tid, id_colors):
        colors = np.array(id_colors.get(tid, []), dtype=float) if id_colors.get(tid) else None
        color_var = float(np.mean(np.var(colors, axis=0))) if colors is not None and len(colors)>0 else None
        color_diff = float(np.mean(np.linalg.norm(np.diff(colors, axis=0), axis=1))) if colors is not None and len(colors)>0 else None

        return color_var, color_diff
    
    def getBBoxSize(self, tid, id_bbox_sizes):
        sizes = np.array(id_bbox_sizes.get(tid, []), dtype=float) if id_bbox_sizes.get(tid) else None
        size_cv = (np.std(sizes)/np.mean(sizes)) if sizes is not None and len(sizes)>1 else None

        return sizes, size_cv


    def metricsOfTic(self, T, tid, frames_seen, id_bboxes, id_teams, id_colors, id_bbox_sizes, id_conf):            
        frames_seen_sorted = sorted(frames_seen)
        total_seen = len(frames_seen_sorted)
        coverage = total_seen / T

        gaps, mean_gap_len, speed, mean_speed, max_speed = self.getGapsAndSpeed(tid, frames_seen_sorted, id_bboxes)

        team_mode, flips_dynamic_bool, flip_rate_static, flip_rate_dynamic, entropy = self.getFlipsAndEntropy(tid, id_teams)

        color_var, color_diff = self.getColorsVar(tid, id_colors)

        sizes, size_cv = self.getBBoxSize(tid, id_bbox_sizes)

        metrics_list = { "frames_seen": frames_seen_sorted, "speed": speed, "flips": flips_dynamic_bool, "bbox_size": sizes, "confs": id_conf[tid] }
        metrics = {
            "coverage": coverage, "total_seen": total_seen, "fragments": gaps, "mean_gap_len": mean_gap_len, "mean_speed": mean_speed, "max_speed": max_speed,
            "team_mode": team_mode, "team_flip_rate_static": flip_rate_static, "team_flip_rate_dynamic": flip_rate_dynamic, "team_entropy": entropy,
            "color_var": color_var, "color_diff": color_diff, "bbox_size_cv": size_cv, "mean_confidence": float(np.mean(id_conf[tid])) if id_conf.get(tid) else None
        }

        return metrics_list, metrics

    def evaluate(self, tracks, class_name="player"):
        frames = tracks[class_name]
        T = len(frames)
        id_frames, id_teams, id_colors, id_bboxes, id_conf, id_bbox_sizes = self.createDefaultDicts(6)
        
        self.initDicts(frames, id_frames, id_teams, id_colors, id_bboxes, id_conf, id_bbox_sizes)
        
        # Metrics per id
        metrics = {}
        metrics_list = {}
        num_tracks = 0
        for tid, frames_seen in id_frames.items():
            if int(tid) > num_tracks:    num_tracks = int(tid)
            metrics_list_tid, metricsTid = self.metricsOfTic(T, tid, frames_seen, id_bboxes, id_teams, id_colors, id_bbox_sizes, id_conf)
            
            metrics_list[tid] = metrics_list_tid
            metrics[tid] = metricsTid

        # Summary stats
        covs = [m["coverage"] for m in metrics.values()]
        total_seen = [m["total_seen"] for m in metrics.values()]
        frag = [m["fragments"] for m in metrics.values()]
        mean_gap_len = [m["mean_gap_len"] for m in metrics.values()]
        mean_speed = [m["mean_speed"] for m in metrics.values()]
        max_speed = [m["max_speed"] for m in metrics.values()]
        flip_rates_static  = [m["team_flip_rate_static" ] for m in metrics.values() if m["team_flip_rate_static"] is not None]
        flip_rates_dynamic = [m["team_flip_rate_dynamic"] for m in metrics.values() if m["team_flip_rate_dynamic"] is not None]
        entropy = [m["team_entropy"] for m in metrics.values()]
        color_var = [m["color_var"] for m in metrics.values()]
        color_diff = [m["color_diff"] for m in metrics.values()]
        bbox_size = [m["bbox_size_cv"] for m in metrics.values() if m["bbox_size_cv"] is not None]
        confidence = [m["mean_confidence"] for m in metrics.values() if m["mean_confidence"] is not None]

        summary = {
            "n_tracks": num_tracks,
            "num_tracks": len(metrics),

            "mean_coverage": float(np.mean(covs)) if covs else 0.0, "median_coverage": float(np.median(covs)) if covs else 0.0, "25_coverage": float(np.quantile(covs, 0.25)) if covs else 0.0,
            "mean_fragments": float(np.mean(frag)) if frag else 0.0, "median_fragments": float(np.median(frag)) if frag else 0.0, "mean_gap_len_mean": float(np.mean(mean_gap_len)) if mean_gap_len else 0.0, "median_gap_len_mean": float(np.median(mean_gap_len)) if mean_gap_len else 0.0, 
            
            "pct_tracks_with_fragments": float(np.mean([1 if f>0 else 0 for f in frag])) if frag else 0.0,

            "mean_speed": float(np.mean(mean_speed) if mean_speed else 0.0),
            "max_speed": float(np.mean(max_speed) if max_speed else 0.0),
            
            "mean_team_flip_rate": float(np.mean(flip_rates_static)) if flip_rates_static else 0.0,
            "mean_team_flip_dynamic": float(np.mean(flip_rates_dynamic)) if flip_rates_dynamic else 0.0,
            "entropy": float(np.mean(entropy)) if entropy else 0.0,
            "color_var": float(np.mean(color_var)) if color_var else 0.0,
            "color_diff": float(np.mean(color_diff)) if color_diff else 0.0,
            "bbox_size": float(np.mean(bbox_size)) if bbox_size else 0.0,
            "confidence": float(np.mean(confidence)) if confidence else 0.0,
        }

        return metrics, metrics_list, summary, T
