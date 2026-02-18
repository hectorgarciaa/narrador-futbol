import logging

import pandas as pd
import plotly.express as px
import numpy as np

logger = logging.getLogger(__name__)


class MetricsVisualizer:

    # ============================================================
    # SPEED EVENTS (per frame)
    # ============================================================
    def collect_speed_events(self, metrics):
        """
        Returns a list of dicts:
        [
            {id, frame_prev, frame_next, frame, speed},
            ...
        ]
        Using metrics[tid]["speed_events"]
        """
        events = []
        for tid, m in metrics.items():
            for (frame_prev, frame_next), speed in m["speed_events"]:
                events.append({
                    "id": int(tid),
                    "frame_prev": int(frame_prev),
                    "frame_next": int(frame_next),
                    "frame": int(frame_next),   # we use frame_next as reference
                    "speed": float(speed)
                })
        return events

    def plot_speed_events_scatter(self, speed_events, title="Speed per frame"):
        if not speed_events:
            logger.warning("No speed events available.")
            return
        
        df = pd.DataFrame(speed_events)

        fig = px.scatter(
            df,
            x="frame",
            y="speed",
            hover_data=["id", "frame_prev", "frame_next", "speed"],
            title=title,
            labels={"frame": "Frame", "speed": "Speed (px/frame)"}
        )
        fig.update_traces(marker=dict(size=6, opacity=0.7))
        fig.show()

    # ============================================================
    # GENERIC METRIC EVENTS (coverage, mean_speed, color_var, etc.)
    # ============================================================
    def collect_metric_events(self, metrics, metric_key):
        """
        Returns a list:
        [
            {"id": X, "frame": Y, "value": Z, ...},
            ...
        ]
        metric_events[metric_key] can be:
          - a list of events (general case)
          - or a single dict (compatibility)
        """
        events = []
        for tid, m in metrics.items():
            # metric_dict = m.get("metric_events", {})
            # entry = metric_dict.get(metric_key)
            entry = m.get(metric_key)

            if entry is None:
                continue

            # if it's a single dict
            if isinstance(entry, dict):
                if entry.get("value") is not None:
                    events.append(entry)
            # if it's a list of events
            else:
                try:
                    for ev in entry:
                        if ev.get("value") is not None:
                            events.append(ev)
                except TypeError:
                    # in case it's something unexpected
                    pass

        return events

    def plot_metric_events_scatter(self, metric_events, metric_key):
        """
        Interactive scatter: frame vs value, hover => id, frame, value (+ other fields)
        """
        if not metric_events:
            logger.warning(f"No metric events for '{metric_key}'.")
            return
        
        df = pd.DataFrame(metric_events)

        # we want to see all useful info in hover
        hover_cols = list(df.columns)

        fig = px.scatter(
            df,
            x="frame",
            y="value",
            hover_data=hover_cols,
            title=f"Metric: {metric_key}",
            labels={"frame": "Frame", "value": metric_key}
        )
        fig.update_traces(marker=dict(size=7, opacity=0.7))
        fig.show()

    # ============================================================
    # OPTIONAL HISTOGRAMS
    # ============================================================
    def plot_metric_histogram(self, metric_events, metric_key, bins=40):
        if not metric_events:
            logger.warning(f"No metric events for '{metric_key}'.")
            return
        
        df = pd.DataFrame(metric_events)

        fig = px.histogram(
            df,
            x="value",
            nbins=bins,
            title=f"Histogram for {metric_key}",
            labels={"value": metric_key}
        )
        fig.show()

    def plot_speed_histogram(self, speed_events, bins=40):
        if not speed_events:
            logger.warning("No speed events available.")
            return
        
        df = pd.DataFrame(speed_events)

        fig = px.histogram(
            df,
            x="speed",
            nbins=bins,
            title="Histogram of Speed",
            labels={"speed": "Speed (px/frame)"}
        )
        fig.show()
