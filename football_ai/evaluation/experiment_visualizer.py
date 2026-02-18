import logging

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from football_ai.evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)

class ExperimentVisualizer:
    def __init__(self, experimentos_tracks, classes):
        self.evaluator = Evaluator()
        self.exps = self.create_exps_rows(experimentos_tracks, classes)
    
    def create_exps_rows(self, experimentos_tracks, classes):
        rows = {class_name: [] for class_name in classes}
        for i, exp in enumerate(experimentos_tracks):
            evaluation = self.evaluator.evaluate(classes, exp["track"])
            for class_name in classes:
                row = {}
                row.update({"conf": exp["conf"], "mcf": exp["mcf"], "mt": exp["mt"], "tt": exp["tt"]})
                row["experiment_id"] = i
                row["metrics"] = evaluation[class_name]["metrics"]
                row["summary"] = evaluation[class_name]["summary"]
                rows[class_name].append(row)
        return rows
    
    def create_df_summary(self, class_name):
        return pd.DataFrame([
            {
                **{k: v for k, v in exp.items() if k not in ['metrics', 'summary', 'track']},
                **exp['summary']
            }
            for exp in self.exps[class_name]
        ])
    
    def create_dfs_metrics(self, class_name):
        rows_detailed = []
        for exp in self.exps[class_name]:
            exp_info = {k: v for k, v in exp.items() if k not in ['metrics', 'summary', 'track']}
            for tid, m in exp['metrics'].items():
                flat_metrics = {}
                for k, v in m.items():
                    if k not in ["speed_events", "team_mode"]:
                        try:
                            flat_metrics[k] = v[0]["value"]
                        except Exception as e:
                            logger.debug(f"Unexpected metric format for key '{k}': {v}")
                            raise e

                rows_detailed.append({
                    **exp_info,
                    "track_id": tid,
                    **flat_metrics
                })

        return pd.DataFrame(rows_detailed)
    
    def show_bars_of_experiments(self, class_name, df_filter=None):
        if df_filter is not None:
            df = df_filter
        else:
            df = self.create_df_summary(class_name)
        
        n_rows = len(df.columns[5:])
        fig, axes = plt.subplots(n_rows, 1, figsize=(25, 2 * n_rows), sharex=True)
        axes = axes.flatten()
        for i, c in enumerate(df.iloc[:, 5:]):
            sns.barplot(data=df, x="experiment_id", y=c, ax=axes[i])
            axes[i].set_ylabel(c)

        plt.tight_layout()
        plt.show()

    def show_boxplot_of_experiments(self, class_name, y, df_filter=None):
        if df_filter is not None:
            df = df_filter
        else:
            df = self.create_dfs_metrics(class_name)
        # Coverage distribution per track
        plt.figure(figsize=(25, 5))
        sns.boxplot(data=df, x="experiment_id", y=y)
        plt.title("Coverage distribution per track")
        plt.xlabel("Experiment")
        plt.ylabel("Coverage")
        plt.show()

    def show_params_comparison(self, class_name, df_filter=None):
        if df_filter is not None:
            df = df_filter
        else:
            df = self.create_df_summary(class_name)
        sns.set(style="whitegrid", palette="muted", font_scale=1.1)

        params = ['conf', 'mcf', 'mt', 'tt']

        nrows = len(df.columns[5:])
        ncols = len(params)
        fig, axes = plt.subplots(nrows, ncols, figsize=(25, nrows*3))
        for i, c in enumerate(df.columns[5:]):
            for j, p in enumerate(params):
                sns.barplot(data=df, x=p, y=c, ax=axes[i, j])
                axes[i, j].set_title(f"{c} vs {p}"), axes[i, j].set_xlabel(p), axes[i, j].set_ylabel(c)

        plt.tight_layout()
        sns.despine()
        plt.show()

    def get_best_exps(self, class_name, sort_by, ascending):
        return self.create_df_summary(class_name).sort_values(by=sort_by, ascending=ascending)
