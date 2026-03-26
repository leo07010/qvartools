"""Tests for experiments/config_loader.py.

Covers the two bugs from George930502/qvartools#1:
  - positional argument with dest kwarg (ValueError)
  - store_true action with nargs=0 (ValueError)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

# config_loader requires pyyaml, which lives in the "configs" extra
pytest.importorskip("yaml")

# Make experiments/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from config_loader import (
    _get_explicit_cli_args,
    _load_yaml,
    create_base_parser,
    load_config,
)


class TestGetExplicitCliArgs:
    """Tests for _get_explicit_cli_args sentinel-based detection."""

    def test_positional_arg_detected(self):
        """Positional 'molecule' is detected when explicitly provided."""
        parser = create_base_parser("test")
        with mock.patch("sys.argv", ["test", "lih"]):
            explicit = _get_explicit_cli_args(parser)
        assert "molecule" in explicit

    def test_positional_arg_not_detected_when_omitted(self):
        """Positional 'molecule' is NOT in explicit set when omitted."""
        parser = create_base_parser("test")
        with mock.patch("sys.argv", ["test"]):
            explicit = _get_explicit_cli_args(parser)
        assert "molecule" not in explicit

    def test_store_true_flag_detected(self):
        """--verbose (store_true) is detected when explicitly provided."""
        parser = create_base_parser("test")
        parser.add_argument("--verbose", action="store_true", default=None)
        with mock.patch("sys.argv", ["test", "--verbose"]):
            explicit = _get_explicit_cli_args(parser)
        assert "verbose" in explicit

    def test_store_true_flag_not_detected_when_omitted(self):
        """--verbose is NOT in explicit set when omitted."""
        parser = create_base_parser("test")
        parser.add_argument("--verbose", action="store_true", default=None)
        with mock.patch("sys.argv", ["test"]):
            explicit = _get_explicit_cli_args(parser)
        assert "verbose" not in explicit

    def test_store_false_flag_detected(self):
        """--no-cache (store_false) is detected when explicitly provided."""
        parser = create_base_parser("test")
        parser.add_argument("--no-cache", action="store_false", dest="cache")
        with mock.patch("sys.argv", ["test", "--no-cache"]):
            explicit = _get_explicit_cli_args(parser)
        assert "cache" in explicit

    def test_store_false_flag_not_detected_when_omitted(self):
        """--no-cache is NOT in explicit set when omitted."""
        parser = create_base_parser("test")
        parser.add_argument("--no-cache", action="store_false", dest="cache")
        with mock.patch("sys.argv", ["test"]):
            explicit = _get_explicit_cli_args(parser)
        assert "cache" not in explicit

    def test_optional_with_value_detected(self):
        """--device cpu is detected when explicitly provided."""
        parser = create_base_parser("test")
        with mock.patch("sys.argv", ["test", "--device", "cuda"]):
            explicit = _get_explicit_cli_args(parser)
        assert "device" in explicit

    def test_positional_and_store_true_together(self):
        """Regression: both positional + store_true in same parser (issue #1)."""
        parser = create_base_parser("test")
        parser.add_argument("--verbose", action="store_true", default=None)
        with mock.patch("sys.argv", ["test", "h2o", "--verbose"]):
            explicit = _get_explicit_cli_args(parser)
        assert "molecule" in explicit
        assert "verbose" in explicit


class TestLoadConfig:
    """Tests for load_config merging precedence."""

    def test_cli_overrides_defaults(self):
        """Explicit CLI args override built-in defaults."""
        parser = create_base_parser("test")
        with mock.patch("sys.argv", ["test", "beh2", "--device", "cpu"]):
            args, config = load_config(parser)
        assert config["molecule"] == "beh2"
        assert config["device"] == "cpu"

    def test_defaults_used_when_no_cli(self):
        """Built-in defaults apply when no CLI args are given."""
        parser = create_base_parser("test")
        with mock.patch("sys.argv", ["test"]):
            args, config = load_config(parser)
        assert config["molecule"] == "h2"

    def test_yaml_overrides_defaults(self, tmp_path):
        """YAML values override built-in defaults."""
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("molecule: n2\ndevice: cuda\n")
        parser = create_base_parser("test")
        with mock.patch("sys.argv", ["test", "--config", str(cfg_file)]):
            args, config = load_config(parser)
        assert config["molecule"] == "n2"
        assert config["device"] == "cuda"

    def test_cli_overrides_yaml(self, tmp_path):
        """Explicit CLI args override YAML values."""
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("molecule: n2\ndevice: cuda\n")
        parser = create_base_parser("test")
        with mock.patch(
            "sys.argv", ["test", "lih", "--config", str(cfg_file), "--device", "cpu"]
        ):
            args, config = load_config(parser)
        assert config["molecule"] == "lih"
        assert config["device"] == "cpu"

    def test_store_true_value_in_merged_config(self):
        """--verbose sets config['verbose'] = True when explicitly provided."""
        parser = create_base_parser("test")
        parser.add_argument("--verbose", action="store_true", default=None)
        with mock.patch("sys.argv", ["test", "--verbose"]):
            _, config = load_config(parser)
        assert config["verbose"] is True

    def test_namespace_reflects_merged_config(self, tmp_path):
        """args namespace is updated to match the final merged config."""
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("device: cuda\n")
        parser = create_base_parser("test")
        with mock.patch("sys.argv", ["test", "--config", str(cfg_file)]):
            args, _ = load_config(parser)
        # device came from YAML, not CLI
        assert args.device == "cuda"


class TestLoadYaml:
    """Tests for _load_yaml error handling."""

    def test_file_not_found(self, tmp_path):
        """Raises FileNotFoundError for nonexistent path."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            _load_yaml(str(tmp_path / "nonexistent.yaml"))

    def test_empty_yaml_returns_empty_dict(self, tmp_path):
        """Empty YAML file returns {}."""
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("")
        assert _load_yaml(str(cfg)) == {}

    def test_non_dict_yaml_raises_value_error(self, tmp_path):
        """YAML with a list at top level raises ValueError."""
        cfg = tmp_path / "list.yaml"
        cfg.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            _load_yaml(str(cfg))

    def test_valid_yaml_returns_dict(self, tmp_path):
        """Valid YAML mapping is returned as dict."""
        cfg = tmp_path / "valid.yaml"
        cfg.write_text("molecule: lih\nmax_epochs: 200\n")
        result = _load_yaml(str(cfg))
        assert result == {"molecule": "lih", "max_epochs": 200}
