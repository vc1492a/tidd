"""
Load and validate experiment configuration from YAML files.
"""

from pathlib import Path
from typing import Any, Union

import yaml


_REQUIRED_KEYS = {
    "data": ["training_path", "validation_path"],
    "output": ["save_path"],
}

_DEFAULTS = {
    "data": {
        "training_path": "data/experiments/generalization_1/hawaii",
        "validation_path": "data/experiments/generalization_1/chile",
        "raw_paths": {
            "hawaii": "data/hawaii",
            "chile": "data/chile",
        },
    },
    "output": {
        "save_path": "output",
    },
    "model": {
        "architecture": "resnet34",
        "batch_size": 256,
        "learning_rate": 0.0001,
        "max_epochs": 250,
    },
    "pipeline": {
        "encoder": "gadf",
        "writer": "pil",
        "window_size": 60,
        "resample_freq": "1min",
        "min_sequence_length": 100,
        "output_path": "data/experiments",
        "workers": -1,
    },
    "experiment": {
        "name": "my_experiment",
        "generate_data": False,
        "cuda_device": 0,
    },
}


def load_config(
    config_path: Union[str, Path, None] = None,
    project_root: Union[str, Path, None] = None,
) -> dict[str, Any]:
    """
    Load experiment configuration from a YAML file.

    :param config_path: Path to the YAML config file. If None, looks for
        ``config.yaml`` in *project_root*.
    :param project_root: Base directory against which relative paths in the
        config are resolved. Defaults to the repository root (parent of the
        ``tidd`` package directory).
    :returns: A validated configuration dictionary.
    """

    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    else:
        project_root = Path(project_root).resolve()

    if config_path is None:
        config_path = project_root / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            f"Copy config.example.yaml to config.yaml and edit as needed."
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    _validate(config)
    _resolve_paths(config, project_root)

    return config


def _validate(config: dict) -> None:
    """Raise ``ValueError`` when required keys are missing."""

    for section, keys in _REQUIRED_KEYS.items():
        if section not in config:
            raise ValueError(f"Config missing required section: '{section}'")
        for key in keys:
            if key not in config[section]:
                raise ValueError(
                    f"Config section '{section}' missing required key: '{key}'"
                )


def _resolve_paths(config: dict, project_root: Path) -> None:
    """Resolve relative paths in the config against *project_root*."""

    path_keys = {
        "data": ["training_path", "validation_path"],
        "output": ["save_path"],
        "pipeline": ["output_path"],
    }

    for section, keys in path_keys.items():
        if section not in config:
            continue
        for key in keys:
            if key in config[section]:
                p = Path(config[section][key])
                if not p.is_absolute():
                    config[section][key] = str(project_root / p)

    if "data" in config and "raw_paths" in config["data"]:
        for location, raw_path in config["data"]["raw_paths"].items():
            p = Path(raw_path)
            if not p.is_absolute():
                config["data"]["raw_paths"][location] = str(project_root / p)
