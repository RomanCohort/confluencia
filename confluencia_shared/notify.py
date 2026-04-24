from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass(frozen=True)
class TwilioConfig:
    account_sid: str
    auth_token: str
    from_number: str
    to_number: str


@dataclass(frozen=True)
class FeishuConfig:
    webhook_url: str
    secret: Optional[str] = None


def load_twilio_config_from_env(prefix: str = "") -> Optional[TwilioConfig]:
    """Load Twilio SMS config from environment variables.

    Required env vars (optionally with a prefix):
      - {prefix}TWILIO_ACCOUNT_SID
      - {prefix}TWILIO_AUTH_TOKEN
      - {prefix}TWILIO_FROM_NUMBER
      - {prefix}FEEDBACK_SMS_TO

    Notes:
      - Keep the recipient number out of the repo; set via env var.
    """

    def _get(name: str) -> str:
        return str(os.environ.get(f"{prefix}{name}", "")).strip()

    account_sid = _get("TWILIO_ACCOUNT_SID")
    auth_token = _get("TWILIO_AUTH_TOKEN")
    from_number = _get("TWILIO_FROM_NUMBER")
    to_number = _get("FEEDBACK_SMS_TO")

    if not (account_sid and auth_token and from_number and to_number):
        return None

    return TwilioConfig(
        account_sid=account_sid,
        auth_token=auth_token,
        from_number=from_number,
        to_number=to_number,
    )


def send_sms_twilio(body: str, cfg: TwilioConfig, timeout: float = 8.0) -> str:
    """Send a SMS via Twilio REST API.

    Returns:
      message_sid

    Raises:
      requests.RequestException on network errors
      RuntimeError on non-2xx
    """

    url = f"https://api.twilio.com/2010-04-01/Accounts/{cfg.account_sid}/Messages.json"
    data = {
        "From": cfg.from_number,
        "To": cfg.to_number,
        "Body": body,
    }

    resp = requests.post(url, data=data, auth=(cfg.account_sid, cfg.auth_token), timeout=timeout)
    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(f"Twilio SMS failed: {resp.status_code} {resp.text[:500]}")

    try:
        payload = resp.json()
    except Exception:
        payload = {}

    sid = str(payload.get("sid", "")).strip()
    return sid or ""


def load_feishu_config_from_env(prefix: str = "") -> Optional[FeishuConfig]:
    """Load Feishu/Lark webhook config from environment variables.

    Required env vars (optionally with a prefix):
      - {prefix}FEISHU_WEBHOOK_URL
    Optional:
      - {prefix}FEISHU_SECRET (for signature)
    """

    def _get(name: str) -> str:
        return str(os.environ.get(f"{prefix}{name}", "")).strip()

    webhook_url = _get("FEISHU_WEBHOOK_URL")
    if not webhook_url:
        return None

    secret = _get("FEISHU_SECRET") or None
    return FeishuConfig(webhook_url=webhook_url, secret=secret)


def _feishu_sign(secret: str, timestamp: int) -> str:
    string_to_sign = f"{timestamp}\n{secret}".encode("utf-8")
    h = hmac.new(secret.encode("utf-8"), string_to_sign, hashlib.sha256).digest()
    return base64.b64encode(h).decode("utf-8")


def send_feishu_webhook(body: str, cfg: FeishuConfig, timeout: float = 8.0) -> None:
    """Send a text message to Feishu/Lark via webhook.

    Raises:
      requests.RequestException on network errors
      RuntimeError on non-2xx
    """

    payload = {
        "msg_type": "text",
        "content": {"text": body},
    }

    if cfg.secret:
        ts = int(time.time())
        payload["timestamp"] = ts
        payload["sign"] = _feishu_sign(cfg.secret, ts)

    resp = requests.post(cfg.webhook_url, json=payload, timeout=timeout)
    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(f"Feishu webhook failed: {resp.status_code} {resp.text[:500]}")
