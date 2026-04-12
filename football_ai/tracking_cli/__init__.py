"""CLI helpers for tracking orchestration."""

__all__ = ["run_tracking_pipeline"]


def __getattr__(name: str):
    if name == "run_tracking_pipeline":
        from .pipeline import run_tracking_pipeline

        return run_tracking_pipeline
    raise AttributeError(name)
