"""CLI smoke tests for Milestone M1."""

from __future__ import annotations

from facebench.main import build_parser, main


def test_build_parser_help() -> None:
    """Parser exposes the expected top-level commands."""
    parser = build_parser()
    actions = {action.dest for action in parser._actions}
    assert "command" in actions


def test_main_help_returns_zero() -> None:
    """``facebench`` with no command prints help and exits 0."""
    assert main([]) == 0


def test_list_datasets_command() -> None:
    """``list-datasets`` exits successfully."""
    assert main(["list-datasets"]) == 0
