from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModelConfig:
    path: str = "weights/S-N0-coco-best.pt"
    imgsz: int = 640
    confidence: float = 0.25
    iou: float = 0.50
    device: str | None = None


@dataclass(slots=True)
class TrackingConfig:
    enabled: bool = True
    tracker: str = "bytetrack.yaml"
    persist: bool = True
    missing_timeout_seconds: float = 2.0


@dataclass(slots=True)
class ClassConfig:
    person: list[str] = field(default_factory=lambda: ["person"])
    head: list[str] = field(default_factory=lambda: ["head"])
    helmet: list[str] = field(default_factory=lambda: ["helmet"])
    vest: list[str] = field(default_factory=lambda: ["vest"])


@dataclass(slots=True)
class AssociationConfig:
    enabled: bool = True
    mode: str = "roi"  # roi | full_body
    head_ratio: float = 0.35
    torso_top_ratio: float = 0.20
    torso_bottom_ratio: float = 0.85
    min_item_inside_ratio: float = 0.50


@dataclass(slots=True)
class EventConfig:
    enabled: bool = True
    mode: str = "consecutive"  # consecutive | majority
    violation_seconds: float = 2.0
    recovery_seconds: float = 0.5
    voting_window_seconds: float = 2.0
    voting_ratio: float = 0.70
    min_voting_samples: int = 5
    required_ppe: list[str] = field(default_factory=lambda: ["helmet", "vest"])


@dataclass(slots=True)
class StorageConfig:
    database: str = "data/detections.db"
    evidence_dir: str = "evidence"
    save_evidence: bool = True


@dataclass(slots=True)
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 5000
    jpeg_quality: int = 85


@dataclass(slots=True)
class AppConfig:
    profile: str = "D_full_system"
    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    classes: ClassConfig = field(default_factory=ClassConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    event: EventConfig = field(default_factory=EventConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    config_dir: Path = field(default_factory=lambda: Path.cwd(), repr=False)

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.config_dir / path).resolve()

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "model": vars_from_slots(self.model),
            "tracking": vars_from_slots(self.tracking),
            "classes": vars_from_slots(self.classes),
            "association": vars_from_slots(self.association),
            "event": vars_from_slots(self.event),
            "storage": vars_from_slots(self.storage),
            "api": vars_from_slots(self.api),
        }


def vars_from_slots(value: Any) -> dict[str, Any]:
    return {name: getattr(value, name) for name in value.__dataclass_fields__}


def _build(cls: type[Any], raw: dict[str, Any] | None) -> Any:
    allowed = cls.__dataclass_fields__.keys()
    return cls(**{key: val for key, val in (raw or {}).items() if key in allowed})


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path = "config.yaml", profile: str | None = None) -> AppConfig:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    selected_profile = profile or raw.get("profile", "D_full_system")
    profiles = raw.pop("ablation_profiles", {})
    if selected_profile:
        if selected_profile not in profiles:
            available = ", ".join(sorted(profiles))
            raise ValueError(
                f"Unknown ablation profile '{selected_profile}'. Available: {available}"
            )
        raw = _deep_merge(raw, profiles[selected_profile])
    return AppConfig(
        profile=selected_profile,
        model=_build(ModelConfig, raw.get("model")),
        tracking=_build(TrackingConfig, raw.get("tracking")),
        classes=_build(ClassConfig, raw.get("classes")),
        association=_build(AssociationConfig, raw.get("association")),
        event=_build(EventConfig, raw.get("event")),
        storage=_build(StorageConfig, raw.get("storage")),
        api=_build(ApiConfig, raw.get("api")),
        config_dir=config_path.parent,
    )
