import argparse
import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs/default.json"


def load_config(path: str = None) -> Dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def get_config_section(config: Dict[str, Any], section: str) -> Dict[str, Any]:
    value = config.get(section, {})
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{section}' must be an object.")
    return value


def parse_config_path(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description, add_help=False)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="path to JSON config file")
    args, _ = parser.parse_known_args()
    return args
