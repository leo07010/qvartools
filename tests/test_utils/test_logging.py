"""Tests for qvartools logging configuration."""

from __future__ import annotations

import logging


class TestConfigureLogging:
    """Tests for configure_logging and get_logger."""

    def test_get_logger_returns_child(self):
        from qvartools._logging import get_logger

        logger = get_logger("qvartools.test_module")
        assert logger.name == "qvartools.test_module"
        assert logger.parent.name == "qvartools"  # type: ignore[union-attr]

    def test_configure_logging_sets_level(self):
        from qvartools._logging import configure_logging

        configure_logging("DEBUG")
        root = logging.getLogger("qvartools")
        assert root.level == logging.DEBUG

        # Restore
        configure_logging("WARNING")

    def test_configure_logging_from_int(self):
        from qvartools._logging import configure_logging

        configure_logging(logging.ERROR)
        root = logging.getLogger("qvartools")
        assert root.level == logging.ERROR

        configure_logging("WARNING")

    def test_package_exports_logging(self):
        import qvartools

        assert hasattr(qvartools, "configure_logging")
        assert hasattr(qvartools, "get_logger")

    def test_root_logger_has_handler(self):
        from qvartools._logging import configure_logging

        configure_logging("WARNING")
        root = logging.getLogger("qvartools")
        assert len(root.handlers) >= 1
        assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)
