"""
Centralized N8N webhook notifications.
Set N8N_WEBHOOK_URL and optional N8N_WEBHOOK_TOKEN in .env
"""
from __future__ import annotations
import json
import time
from datetime import datetime
from typing import Any, Dict, Optional
from urllib import request as _req
from urllib.error import URLError, HTTPError
import socket
from django.conf import settings
import logging

logger = logging.getLogger(__name__)


def _pick_n8n_url_and_token():
    """Return (url, token, mode) preferring production over test.
    mode is 'prod', 'test', or 'none'.
    """
    prod_url = getattr(settings, "N8N_WEBHOOK_URL", "") or ""
    test_url = getattr(settings, "N8N_WEBHOOK_TEST_URL", "") or ""
    prod_tok = getattr(settings, "N8N_WEBHOOK_TOKEN", "") or ""
    test_tok = getattr(settings, "N8N_WEBHOOK_TEST_TOKEN", "") or ""
    if prod_url:
        return prod_url, prod_tok, 'prod'
    if test_url:
        return test_url, test_tok, 'test'
    return "", "", 'none'


def _post_json(url: str, payload: Dict[str, Any], token: str = "", timeout: int = 6) -> bool:
    data = json.dumps(payload).encode("utf-8")
    base_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, */*",
        "User-Agent": "FraudB/1.0 (+n8n-integration)",
        # Avoid keep-alive to mitigate BrokenPipe on some proxies
        "Connection": "close",
    }
    if token:
        base_headers["Authorization"] = f"Bearer {token}"

    attempts = 3
    for attempt in range(1, attempts + 1):
        req = _req.Request(url, data=data, headers=base_headers, method="POST")
        try:
            with _req.urlopen(req, timeout=timeout) as resp:
                return 200 <= int(getattr(resp, 'status', 0)) < 300
        except (HTTPError, URLError, socket.timeout, BrokenPipeError) as e:
            logger.warning("[n8n] _post_json attempt %s/%s failed: %s", attempt, attempts, getattr(e, 'reason', e))
            if attempt == attempts:
                return False
            try:
                time.sleep(0.3 * attempt)
            except Exception:
                pass


def send_to_n8n(event_type: str, payload: Dict[str, Any], user: Optional[Any] = None, meta: Optional[Dict[str, Any]] = None) -> bool:
    """
    Send a structured event to N8N webhook.
    - event_type: e.g. 'prediction', 'alert', 'alert_action', 'drift_snapshot', 'drift_alert', 'batch_summary', 'profile_updated'
    - payload: domain-specific data
    - user: optional Django user to enrich metadata
    - meta: additional metadata
    Returns True if delivered, False otherwise.
    """
    url, token, mode = _pick_n8n_url_and_token()
    if not url:
        logger.warning("[n8n] send_to_n8n skipped: no webhook URL (prod/test) configured")
        return False
    if mode != 'prod':
        logger.info("[n8n] send_to_n8n using %s webhook", mode)
    envelope = {
        "event": event_type,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "payload": payload or {},
        "meta": {**(meta or {}), "n8n_mode": mode},
    }
    if user is not None:
        try:
            envelope["user"] = {
                "id": str(getattr(user, "id", "")),
                "email": getattr(user, "email", ""),
                "username": getattr(user, "username", ""),
                "role": getattr(user, "role", ""),
            }
        except Exception:
            pass
    ok = _post_json(url, envelope, token)
    if not ok:
        logger.warning("[n8n] send_to_n8n delivery failed: mode=%s url=%s event=%s", mode, url, event_type)
    return ok


def call_n8n(event_type: str, payload: Dict[str, Any], user: Optional[Any] = None, meta: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Dict[str, Any]:
    """
    Call N8N webhook and return parsed JSON response.
    Expected n8n workflow response schema (example):
    {
      "status": "ok",
      "reason": "AI-generated reason string",
      "recommendations": ["...", "..."],
      "adjusted_risk": 0.68,
      "per_item": {
         "<transaction_id>": {"reason": "...", "recommendations": [..], "adjusted_risk": 0.7}
      }
    }
    """
    url, token, mode = _pick_n8n_url_and_token()
    if not url:
        logger.warning("[n8n] call_n8n skipped: no webhook URL (prod/test) configured")
        return {}
    if mode != 'prod':
        logger.info("[n8n] call_n8n using %s webhook", mode)
    envelope: Dict[str, Any] = {
        "event": event_type,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "payload": payload or {},
        "meta": {**(meta or {}), "n8n_mode": mode},
    }
    if user is not None:
        try:
            envelope["user"] = {
                "id": str(getattr(user, "id", "")),
                "email": getattr(user, "email", ""),
                "username": getattr(user, "username", ""),
                "role": getattr(user, "role", ""),
            }
        except Exception:
            pass
    data = json.dumps(envelope).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, */*",
        "User-Agent": "FraudB/1.0 (+n8n-integration)",
        "Connection": "close",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    def _try_request(_url: str, _token: str, _mode: str) -> Dict[str, Any]:
        _headers = dict(headers)
        if _token:
            _headers["Authorization"] = f"Bearer {_token}"
        _attempts = 3
        for attempt in range(1, _attempts + 1):
            req = _req.Request(_url, data=data, headers=_headers, method="POST")
            try:
                with _req.urlopen(req, timeout=timeout) as resp:
                    status = int(getattr(resp, 'status', 0) or 0)
                    body = resp.read().decode("utf-8", errors="ignore")
                    if not (200 <= status < 300):
                        logger.warning("[n8n] call_n8n non-2xx (%s): status=%s body=%s", _mode, status, body[:2000])
                    try:
                        return json.loads(body) if body else {}
                    except Exception:
                        logger.info("[n8n] call_n8n returned non-JSON body (%s); exposing as raw", _mode)
                        return {"raw": body}
            except HTTPError as e:
                try:
                    err_body = e.read().decode("utf-8", errors="ignore")
                except Exception:
                    err_body = ""
                logger.error("[n8n] call_n8n HTTPError (attempt %s/%s, %s): code=%s body=%s", attempt, _attempts, _mode, getattr(e, 'code', None), err_body[:2000])
            except (URLError, socket.timeout, BrokenPipeError) as e:
                logger.error("[n8n] call_n8n transport error (attempt %s/%s, %s): %s", attempt, _attempts, _mode, getattr(e, 'reason', e))
            try:
                time.sleep(0.4 * attempt)
            except Exception:
                pass
        return {}

    # First try the selected (prod or test) URL
    result = _try_request(url, token, mode)
    # If prod was selected and result looks empty/failed, fallback to test if configured
    if mode == 'prod' and (not result or (isinstance(result, dict) and not result.get('reply') and 'raw' in result)):
        test_url = getattr(settings, "N8N_WEBHOOK_TEST_URL", "") or ""
        test_tok = getattr(settings, "N8N_WEBHOOK_TEST_TOKEN", "") or ""
        if test_url:
            logger.warning("[n8n] call_n8n falling back to TEST webhook")
            # Mark meta for the fallback attempt
            try:
                env2 = dict(envelope)
                env2['meta'] = {**env2.get('meta', {}), 'n8n_mode': 'fallback_test'}
                data = json.dumps(env2).encode("utf-8")
            except Exception:
                pass
            result = _try_request(test_url, test_tok, 'fallback_test')
    return result
