"""Cloud configuration loader for Confluencia 2.0.

Reads settings from cloud_config.yaml (or path in CLOUD_CONFIG_PATH env var).
All cloud features degrade gracefully: if the config file or settings are missing,
the system falls back to fully local operation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CloudServerConfig:
    url: str = ""
    api_prefix: str = "/api/v1"
    timeout: int = 300
    max_retries: int = 3
    retry_delay: float = 2.0


@dataclass
class CloudAuthConfig:
    token: str = ""
    header_name: str = "X-API-Token"


@dataclass
class CloudModeConfig:
    default: str = "local"  # "local" | "cloud" | "hybrid"
    prefer_cloud_train: bool = False
    prefer_cloud_predict: bool = True


@dataclass
class CloudStorageConfig:
    auto_upload_model: bool = False
    max_upload_rows: int = 0


@dataclass
class CloudConfig:
    enabled: bool = False
    server: CloudServerConfig = field(default_factory=CloudServerConfig)
    auth: CloudAuthConfig = field(default_factory=CloudAuthConfig)
    mode: CloudModeConfig = field(default_factory=CloudModeConfig)
    storage: CloudStorageConfig = field(default_factory=CloudStorageConfig)

    @property
    def base_url(self) -> str:
        url = self.server.url.rstrip("/")
        prefix = self.server.api_prefix.strip("/")
        return f"{url}/{prefix}" if prefix else url

    def is_cloud_mode(self) -> bool:
        return self.mode.default in ("cloud", "hybrid") and self.enabled

    def should_use_cloud_train(self) -> bool:
        if not self.is_cloud_mode():
            return False
        if self.mode.default == "cloud":
            return True
        return self.mode.prefer_cloud_train

    def should_use_cloud_predict(self) -> bool:
        if not self.is_cloud_mode():
            return False
        if self.mode.default == "cloud":
            return True
        return self.mode.prefer_cloud_predict


def _find_config_path() -> Optional[Path]:
    env_path = os.environ.get("CLOUD_CONFIG_PATH", "").strip()
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
    # Search relative to this file's parent directories
    candidates = [
        Path(__file__).resolve().parent.parent / "cloud_config.yaml",
        Path(__file__).resolve().parent / "cloud_config.yaml",
        Path.cwd() / "cloud_config.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _parse_yaml_simple(text: str) -> Dict[str, Any]:
    """Minimal YAML-like parser that handles flat and one-level-nested keys.

    Handles patterns like:
        key: value
        section:
          key: value
    """
    result: Dict[str, Any] = {}
    current_section: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.lstrip()

        # Skip empty / comments
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(stripped)

        if indent == 0 and stripped.endswith(":") and not ":" in stripped[:-1]:
            # Top-level section header
            current_section = stripped[:-1].strip()
            result.setdefault(current_section, {})
            continue

        # Parse key: value
        colon_idx = stripped.find(":")
        if colon_idx < 0:
            continue

        key = stripped[:colon_idx].strip()
        val_str = stripped[colon_idx + 1:].strip()

        # Remove inline comments
        if "#" in val_str:
            val_str = val_str[: val_str.index("#")].strip()

        # Type coercion
        val: Any = val_str
        if val_str.startswith('"') and val_str.endswith('"'):
            val = val_str[1:-1]
        elif val_str.lower() in ("true", "yes"):
            val = True
        elif val_str.lower() in ("false", "no"):
            val = False
        elif val_str == "0" or (val_str.isdigit() and not val_str.startswith("0")):
            val = int(val_str)
        else:
            try:
                val = float(val_str)
            except ValueError:
                pass

        if indent > 0 and current_section:
            result.setdefault(current_section, {})[key] = val
        else:
            result[key] = val
            current_section = None

    return result


def load_cloud_config(path: Optional[str | Path] = None) -> CloudConfig:
    """Load cloud configuration. Returns a disabled CloudConfig if no file found."""
    if path is not None:
        p = Path(path)
    else:
        p = _find_config_path()

    if p is None or not p.exists():
        return CloudConfig(enabled=False)

    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        return CloudConfig(enabled=False)

    try:
        data = _parse_yaml_simple(text)
    except Exception:
        return CloudConfig(enabled=False)

    server_data = data.get("server", {}) or {}
    auth_data = data.get("auth", {}) or {}
    mode_data = data.get("mode", {}) or {}
    storage_data = data.get("storage", {}) or {}

    server_cfg = CloudServerConfig(
        url=str(server_data.get("url", "")),
        api_prefix=str(server_data.get("api_prefix", "/api/v1")),
        timeout=int(server_data.get("timeout", 300)),
        max_retries=int(server_data.get("max_retries", 3)),
        retry_delay=float(server_data.get("retry_delay", 2.0)),
    )

    auth_cfg = CloudAuthConfig(
        token=str(auth_data.get("token", "")),
        header_name=str(auth_data.get("header_name", "X-API-Token")),
    )

    mode_cfg = CloudModeConfig(
        default=str(mode_data.get("default", "local")),
        prefer_cloud_train=bool(mode_data.get("prefer_cloud_train", False)),
        prefer_cloud_predict=bool(mode_data.get("prefer_cloud_predict", True)),
    )

    storage_cfg = CloudStorageConfig(
        auto_upload_model=bool(storage_data.get("auto_upload_model", False)),
        max_upload_rows=int(storage_data.get("max_upload_rows", 0)),
    )

    enabled = bool(server_cfg.url) and bool(auth_cfg.token)

    return CloudConfig(
        enabled=enabled,
        server=server_cfg,
        auth=auth_cfg,
        mode=mode_cfg,
        storage=storage_cfg,
    )


def save_cloud_config(config: CloudConfig, path: Optional[str | Path] = None) -> Path:
    """Persist a CloudConfig back to a YAML file."""
    if path is not None:
        p = Path(path)
    else:
        p = _find_config_path()
        if p is None:
            p = Path(__file__).resolve().parent.parent / "cloud_config.yaml"

    lines = [
        "# Confluencia 2.0 云服务器配置",
        "",
        "server:",
        f"  url: \"{config.server.url}\"",
        f"  api_prefix: \"{config.server.api_prefix}\"",
        f"  timeout: {config.server.timeout}",
        f"  max_retries: {config.server.max_retries}",
        f"  retry_delay: {config.server.retry_delay}",
        "",
        "auth:",
        f"  token: \"{config.auth.token}\"",
        f"  header_name: \"{config.auth.header_name}\"",
        "",
        "mode:",
        f'  default: "{config.mode.default}"',
        f"  prefer_cloud_train: {str(config.mode.prefer_cloud_train).lower()}",
        f"  prefer_cloud_predict: {str(config.mode.prefer_cloud_predict).lower()}",
        "",
        "storage:",
        f"  auto_upload_model: {str(config.storage.auto_upload_model).lower()}",
        f"  max_upload_rows: {config.storage.max_upload_rows}",
        "",
    ]
    p.write_text("\n".join(lines), encoding="utf-8")
    return p
