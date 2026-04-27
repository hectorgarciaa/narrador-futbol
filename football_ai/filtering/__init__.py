"""
Filtering utilities shared across tracking and homography pipelines.
"""

from .post_projection import filter_reference_points

__all__ = ["filter_reference_points"]
