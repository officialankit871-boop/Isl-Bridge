"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the ISL Bridge backend."""

    app_name: str = Field(default="ISL Bridge Backend", validation_alias="APP_NAME")
    app_version: str = Field(default="0.1.0", validation_alias="APP_VERSION")
    environment: str = Field(default="development", validation_alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    model_checkpoint_path: str = Field(
        default="checkpoints/cnn_lstm_best.pt",
        validation_alias="MODEL_CHECKPOINT_PATH",
    )
    dataset_root: str = Field(
        default="data/raw/archive/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus",
        validation_alias="DATASET_ROOT",
    )
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        validation_alias="CORS_ORIGINS",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def project_root(self) -> Path:
        """Return the repository root relative to this configuration file."""
        return Path(__file__).resolve().parents[2]

    @property
    def resolved_model_checkpoint_path(self) -> Path:
        """Return the configured checkpoint path resolved against the repo root."""
        candidate = Path(self.model_checkpoint_path)
        return candidate if candidate.is_absolute() else self.project_root / candidate

    @property
    def resolved_dataset_root(self) -> Path:
        """Return the configured dataset path resolved against the repo root."""
        candidate = Path(self.dataset_root)
        return candidate if candidate.is_absolute() else self.project_root / candidate

    @property
    def cors_origin_list(self) -> list[str]:
        """Return the configured CORS origins as a normalized list."""
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()
