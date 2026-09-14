import os
from pathlib import Path
from typing import Any, Dict
import yaml

def load_yaml(file_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_project_root() -> Path:
    """Return the absolute path of the project root directory."""
    # Since config.py is at src/medguard/utils/config.py, root is 3 levels up
    return Path(__file__).resolve().parent.parent.parent.parent

def load_all_configs(base_dir: str | Path | None = None) -> Dict[str, Any]:
    """Load all standard configurations (config.yaml, features.yaml, model_config.yaml)."""
    root = Path(base_dir) if base_dir else get_project_root()
    config_dir = root / "configs"
    
    main_cfg = load_yaml(config_dir / "config.yaml") if (config_dir / "config.yaml").exists() else {}
    features_cfg = load_yaml(config_dir / "features.yaml") if (config_dir / "features.yaml").exists() else {}
    models_cfg = load_yaml(config_dir / "model_config.yaml") if (config_dir / "model_config.yaml").exists() else {}
    
    return {
        "global": main_cfg,
        "features": features_cfg,
        "models": models_cfg,
    }
