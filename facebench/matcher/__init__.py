"""Embedding similarity matchers."""

from facebench.matcher.base import (
    BaseMatcher,
    CosineMatcher,
    EuclideanMatcher,
    create_matcher,
)

__all__ = [
    "BaseMatcher",
    "CosineMatcher",
    "EuclideanMatcher",
    "create_matcher",
]
