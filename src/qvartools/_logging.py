"""
_logging --- Structured logging configuration for qvartools
=============================================================

Provides a package-level logging setup with consistent formatting.
Users can control log level via ``qvartools.configure_logging()``
or the ``QVARTOOLS_LOG_LEVEL`` environment variable.
"""

from __future__ import annotations

import logging
import os
import sys
import threading

__all__ = [
    "configure_logging",
    "get_logger",
]

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False
_lock = threading.Lock()


def configure_logging(
    level: str | int | None = None,
    *,
    stream: bool = True,
    log_file: str | None = None,
) -> None:
    """Configure the qvartools logging hierarchy.

    Safe to call multiple times; only the first call sets up handlers.
    Subsequent calls update the log level only.

    Parameters
    ----------
    level : str or int or None
        Log level (e.g. ``"INFO"``, ``"DEBUG"``, ``logging.WARNING``).
        If ``None``, reads from ``QVARTOOLS_LOG_LEVEL`` env var,
        defaulting to ``"WARNING"``.
    stream : bool
        If ``True`` (default), add a stderr ``StreamHandler``.
    log_file : str or None
        If set, also log to this file path.
    """
    global _configured

    if level is None:
        level = os.environ.get("QVARTOOLS_LOG_LEVEL", "WARNING")
    if isinstance(level, str):
        resolved = getattr(logging, level.upper(), None)
        if resolved is None:
            import warnings

            warnings.warn(
                f"Unknown log level {level!r}, defaulting to WARNING.",
                stacklevel=2,
            )
            resolved = logging.WARNING
        level = resolved

    root_logger = logging.getLogger("qvartools")
    root_logger.setLevel(level)

    with _lock:
        if not _configured:
            formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

            if stream:
                sh = logging.StreamHandler(sys.stderr)
                sh.setFormatter(formatter)
                root_logger.addHandler(sh)

            if log_file:
                fh = logging.FileHandler(log_file, mode="a")
                fh.setFormatter(formatter)
                root_logger.addHandler(fh)

            # Prevent propagation to root logger to avoid duplicate messages
            root_logger.propagate = False
            _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``qvartools`` hierarchy.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)
