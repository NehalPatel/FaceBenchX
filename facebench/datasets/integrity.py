"""Dataset integrity validation helpers."""

from __future__ import annotations

from pathlib import Path

from facebench.datasets.types import ValidationResult


class IntegrityValidator:
    """Validate local public-dataset layouts without downloading data.

    The validator checks that configured roots exist and that required
    relative paths (files or directories) are present. It never fetches
    remote archives.
    """

    def validate(
        self,
        root_path: str | Path,
        *,
        required_paths: list[str] | None = None,
        require_subdirectories: bool = False,
        min_subdirectories: int = 1,
        prep_doc: str | None = None,
        extra_checks: list[tuple[bool, str]] | None = None,
    ) -> ValidationResult:
        """Run layout checks against a local dataset root.

        Args:
            root_path: Dataset root directory from configuration.
            required_paths: Relative files/dirs that must exist under
                ``root_path`` (or absolute paths when given as such).
            require_subdirectories: When ``True``, require at least
                ``min_subdirectories`` immediate child directories.
            min_subdirectories: Minimum child directory count.
            prep_doc: Documentation path returned on failure.
            extra_checks: Optional ``(passed, message)`` tuples for
                adapter-specific conditions.

        Returns:
            :class:`ValidationResult` summarizing success or failures.
        """
        root = Path(root_path).expanduser().resolve()
        missing: list[str] = []
        messages: list[str] = []

        if not root.exists():
            missing.append(str(root))
            messages.append(f"Dataset root does not exist: {root}")
            return ValidationResult(
                ok=False,
                missing=missing,
                messages=messages,
                prep_doc=prep_doc,
                checked_path=str(root),
            )

        if not root.is_dir():
            missing.append(str(root))
            messages.append(f"Dataset root is not a directory: {root}")
            return ValidationResult(
                ok=False,
                missing=missing,
                messages=messages,
                prep_doc=prep_doc,
                checked_path=str(root),
            )

        for relative in required_paths or []:
            candidate = Path(relative)
            target = candidate if candidate.is_absolute() else root / candidate
            if not target.exists():
                missing.append(str(target))
                messages.append(f"Missing required path: {target}")

        if require_subdirectories:
            child_dirs = [p for p in root.iterdir() if p.is_dir()]
            if len(child_dirs) < min_subdirectories:
                messages.append(
                    "Expected at least "
                    f"{min_subdirectories} identity/subdirectory folder(s) "
                    f"under {root}, found {len(child_dirs)}"
                )
                if not child_dirs:
                    missing.append(f"{root}/*/")

        for passed, message in extra_checks or []:
            if not passed:
                messages.append(message)

        ok = not missing and not messages
        if ok:
            messages.append(f"Integrity checks passed for {root}")

        return ValidationResult(
            ok=ok,
            missing=missing,
            messages=messages,
            prep_doc=None if ok else prep_doc,
            checked_path=str(root),
        )

    def merge(self, *results: ValidationResult) -> ValidationResult:
        """Merge multiple validation results into one aggregate result.

        Args:
            *results: Individual validation results.

        Returns:
            Combined :class:`ValidationResult` (``ok`` only if all pass).
        """
        missing: list[str] = []
        messages: list[str] = []
        prep_doc: str | None = None
        checked_path: str | None = None
        for result in results:
            missing.extend(result.missing)
            messages.extend(result.messages)
            if prep_doc is None and result.prep_doc:
                prep_doc = result.prep_doc
            if checked_path is None:
                checked_path = result.checked_path
        ok = all(result.ok for result in results) if results else True
        return ValidationResult(
            ok=ok,
            missing=missing,
            messages=messages,
            prep_doc=None if ok else prep_doc,
            checked_path=checked_path,
        )
