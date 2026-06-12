from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def root(self) -> Path:
        return self.path.parent.parent

    def resolve(self, key_path: str) -> Path:
        value: Any = self.raw
        for key in key_path.split("."):
            value = value[key]
        p = Path(value)
        return p if p.is_absolute() else self.root / p


def load_config(config_path: str | Path) -> ExperimentConfig:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig(path=path, raw=raw)


def load_plate_map(cfg: ExperimentConfig) -> pd.DataFrame:
    plate_map = cfg.resolve("paths.plate_map_csv")
    if not plate_map.exists():
        raise FileNotFoundError(f"Plate map not found: {plate_map}")
    return pd.read_csv(plate_map)


def load_roi_annotations(cfg: ExperimentConfig) -> pd.DataFrame:
    roi_csv = cfg.resolve("paths.roi_annotations_csv")
    if not roi_csv.exists():
        raise FileNotFoundError(f"ROI annotations not found: {roi_csv}")
    return pd.read_csv(roi_csv)


def ensure_dirs(cfg: ExperimentConfig) -> None:
    for key in ["paths.raw_root", "paths.interim_root", "paths.processed_root"]:
        cfg.resolve(key).mkdir(parents=True, exist_ok=True)


def validate_config(cfg: ExperimentConfig) -> list[str]:
    messages: list[str] = []
    required = [
        "paths.raw_root",
        "paths.interim_root",
        "paths.processed_root",
        "paths.plate_map_csv",
        "channels.alignment.aliases",
        "channels.phenotype.name",
    ]
    for item in required:
        value: Any = cfg.raw
        try:
            for key in item.split("."):
                value = value[key]
            messages.append(f"OK: {item} = {value}")
        except KeyError:
            messages.append(f"MISSING: {item}")
    return messages
