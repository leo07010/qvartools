"""YAML configuration loader with CLI override support.

Loads pipeline parameters from YAML config files, with command-line
arguments taking precedence over file values. Falls back to sensible
defaults when neither source provides a value.

Usage::

    from config_loader import load_config, create_base_parser

    parser = create_base_parser("NF-SKQD pipeline")
    args, config = load_config(parser)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "molecule": "h2",
    "device": "auto",
    "verbose": True,
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def create_base_parser(
    description: str = "qvartools pipeline",
) -> argparse.ArgumentParser:
    """Create an :class:`argparse.ArgumentParser` with common pipeline flags.

    Parameters
    ----------
    description:
        Human-readable description shown in ``--help`` output.

    Returns
    -------
    argparse.ArgumentParser
        A parser pre-configured with ``molecule``, ``--config``, and
        ``--device`` arguments.  Callers may add extra arguments before
        passing the parser to :func:`load_config`.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "molecule",
        nargs="?",
        default=_DEFAULTS["molecule"],
        help="Molecule identifier (e.g. h2, lih, h2o)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string (e.g. cpu, cuda, auto)",
    )
    return parser


def load_config(
    parser: argparse.ArgumentParser,
) -> tuple[argparse.Namespace, dict[str, Any]]:
    """Parse CLI args, load an optional YAML config, and merge them.

    Precedence (highest to lowest):

    1. Explicitly-provided CLI arguments
    2. Values from the YAML config file (``--config``)
    3. Built-in defaults

    Parameters
    ----------
    parser:
        An argument parser (typically from :func:`create_base_parser`).

    Returns
    -------
    tuple[argparse.Namespace, dict[str, Any]]
        ``(args, config)`` where *args* is the parsed namespace and
        *config* is a flat dictionary containing every resolved parameter.
    """
    args = parser.parse_args()
    explicitly_provided = _get_explicit_cli_args(parser)

    # Start with built-in defaults
    config: dict[str, Any] = dict(_DEFAULTS)

    # Layer YAML values on top of defaults
    if args.config is not None:
        yaml_values = _load_yaml(args.config)
        config.update(yaml_values)

    # Layer explicit CLI args on top (highest precedence)
    args_dict = vars(args)
    for key in explicitly_provided:
        config[key] = args_dict[key]

    # Ensure the namespace reflects the final merged config
    for key, value in config.items():
        setattr(args, key, value)

    return args, config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: str) -> dict[str, Any]:
    """Read and parse a YAML file, returning a flat dictionary.

    Raises
    ------
    FileNotFoundError
        If *path* does not point to an existing file.
    yaml.YAMLError
        If the file contains invalid YAML.
    ValueError
        If the top-level YAML value is not a mapping.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Config file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a YAML mapping at the top level, got {type(data).__name__}"
        )
    return dict(data)


def _get_explicit_cli_args(parser: argparse.ArgumentParser) -> set[str]:
    """Return the set of argument *dest* names explicitly provided on the CLI.

    We parse ``sys.argv`` a second time using a sentinel default so that
    we can distinguish "user typed ``--device cpu``" from "argparse
    filled in the default".
    """
    sentinel = object()
    probe = argparse.ArgumentParser(add_help=False)

    for action in parser._actions:  # noqa: SLF001
        if isinstance(action, argparse._HelpAction):  # noqa: SLF001
            continue

        kwargs: dict[str, Any] = {
            "default": sentinel,
            "dest": action.dest,
        }
        # Preserve nargs so positional / optional handling stays correct
        if action.nargs is not None:
            kwargs["nargs"] = action.nargs
        if action.type is not None:
            kwargs["type"] = action.type

        if action.option_strings:
            probe.add_argument(*action.option_strings, **kwargs)
        else:
            probe.add_argument(action.dest, **kwargs)

    probe_ns, _ = probe.parse_known_args()

    return {
        key
        for key, value in vars(probe_ns).items()
        if value is not sentinel
    }
