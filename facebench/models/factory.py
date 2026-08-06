"""Factory for constructing recognizer adapters from config names."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from facebench.models.adaface.recognizer import AdaFaceRecognizer
from facebench.models.base import BaseRecognizer
from facebench.models.buffalo_l.recognizer import BuffaloLRecognizer
from facebench.models.dlib_fr.recognizer import DlibRecognizer
from facebench.models.facenet.recognizer import FaceNetRecognizer
from facebench.models.magface.recognizer import MagFaceRecognizer


class ModelFactoryError(ValueError):
    """Raised when a recognizer adapter cannot be constructed."""


class ModelFactory:
    """Create :class:`BaseRecognizer` adapters by model name.

    Milestone M5 registers FaceNet, Dlib, Buffalo-L, AdaFace, and MagFace.
    """

    def __init__(self) -> None:
        """Initialize the factory with M4/M5 builders."""
        self._builders: dict[str, Callable[..., BaseRecognizer]] = {
            "facenet": self._build_facenet,
            "dlib": self._build_dlib,
            "buffalo_l": self._build_buffalo_l,
            "adaface": self._build_adaface,
            "magface": self._build_magface,
        }
        self._aliases: dict[str, str] = {
            "FaceNet": "facenet",
            "FACENET": "facenet",
            "Dlib": "dlib",
            "DLIB": "dlib",
            "dlib_fr": "dlib",
            "dlib-face": "dlib",
            "dlib_face_recognition": "dlib",
            "buffalo-l": "buffalo_l",
            "Buffalo-L": "buffalo_l",
            "BUFFALO_L": "buffalo_l",
            "insightface": "buffalo_l",
            "AdaFace": "adaface",
            "ADAFACE": "adaface",
            "MagFace": "magface",
            "MAGFACE": "magface",
        }

    def available(self) -> list[str]:
        """Return canonical model names with implemented adapters.

        Returns:
            Sorted model names.
        """
        return sorted(self._builders.keys())

    def canonicalize(self, name: str) -> str:
        """Resolve a model name or alias to a canonical builder key.

        Args:
            name: Model name from configuration.

        Returns:
            Canonical builder key.

        Raises:
            ModelFactoryError: If the name is unknown.
        """
        key = name.strip()
        if key in self._builders:
            return key
        if key in self._aliases:
            return self._aliases[key]
        lowered = key.lower().replace("-", "_")
        if lowered in self._builders:
            return lowered
        for candidate in list(self._builders) + list(self._aliases):
            if candidate.lower() == key.lower():
                return (
                    self._aliases[candidate]
                    if candidate in self._aliases
                    else candidate
                )
        raise ModelFactoryError(
            f"Unknown model {name!r}. Implemented: {', '.join(self.available())}."
        )

    def create(self, name: str, device: str = "cpu", **kwargs: Any) -> BaseRecognizer:
        """Construct a recognizer adapter.

        Args:
            name: Model name or alias.
            device: Target device string.
            **kwargs: Forwarded to the adapter constructor.

        Returns:
            Concrete :class:`BaseRecognizer`.

        Raises:
            ModelFactoryError: If the model is unknown.
        """
        canonical = self.canonicalize(name)
        return self._builders[canonical](device=device, **kwargs)

    @staticmethod
    def _build_facenet(device: str = "cpu", **kwargs: Any) -> BaseRecognizer:
        """Build the FaceNet adapter."""
        return FaceNetRecognizer(device=device, **kwargs)

    @staticmethod
    def _build_dlib(device: str = "cpu", **kwargs: Any) -> BaseRecognizer:
        """Build the Dlib adapter."""
        return DlibRecognizer(device=device, **kwargs)

    @staticmethod
    def _build_buffalo_l(device: str = "cpu", **kwargs: Any) -> BaseRecognizer:
        """Build the Buffalo-L adapter."""
        return BuffaloLRecognizer(device=device, **kwargs)

    @staticmethod
    def _build_adaface(device: str = "cpu", **kwargs: Any) -> BaseRecognizer:
        """Build the AdaFace adapter."""
        return AdaFaceRecognizer(device=device, **kwargs)

    @staticmethod
    def _build_magface(device: str = "cpu", **kwargs: Any) -> BaseRecognizer:
        """Build the MagFace adapter."""
        return MagFaceRecognizer(device=device, **kwargs)


def create_model(name: str, device: str = "cpu", **kwargs: Any) -> BaseRecognizer:
    """Convenience wrapper around :meth:`ModelFactory.create`.

    Args:
        name: Model name or alias.
        device: Target device string.
        **kwargs: Adapter keyword arguments.

    Returns:
        Concrete recognizer adapter.
    """
    return ModelFactory().create(name, device=device, **kwargs)
