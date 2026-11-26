# SPDX-FileCopyrightText: 2026 Christoph Medicus
# SPDX-License-Identifier: MIT

# pollinations_auth.py

import os, time, json
from pathlib import Path
import requests

APP_KEY = os.environ.get("POLLINATIONS_APP_KEY", "")
_KEY_FILE = Path(__file__).parent / ".pollinations_key"

_user_key: str | None = None
_device_state: dict = {}


def _load_key() -> str | None:
    """Load persisted key from disk."""
    global _user_key
    if _user_key:
        return _user_key
    try:
        if _KEY_FILE.exists():
            data = json.loads(_KEY_FILE.read_text())
            _user_key = data.get("key")
            # Optional: check if it's still within expiry
            if data.get("expires_at") and time.time() > data["expires_at"]:
                _user_key = None
                _KEY_FILE.unlink(missing_ok=True)
    except Exception:
        _user_key = None
    return _user_key


def _save_key(key: str, expires_in: int = 604800) -> None:
    """Persist key to disk."""
    _KEY_FILE.write_text(json.dumps({
        "key": key,
        "saved_at": time.time(),
        "expires_at": time.time() + expires_in,
    }))


def _invalidate() -> None:
    """Remove key from memory and disk (call on 401/402)."""
    global _user_key
    _user_key = None
    _KEY_FILE.unlink(missing_ok=True)


def get_key() -> str | None:
    """Return BYOP key (from memory or disk), else None."""
    return _load_key()


def start_device_flow() -> dict:
    global _device_state
    resp = requests.post(
        "https://enter.pollinations.ai/api/device/code",
        json={"client_id": APP_KEY},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    _device_state["device_code"] = data["device_code"]
    _device_state["interval"] = data.get("interval", 5)
    return {
        "verification_uri": data.get("verification_uri_complete", data["verification_uri"]),
        "user_code": data["user_code"],
        "expires_in": data["expires_in"],
    }


async def poll_for_token() -> str:
    import asyncio
    global _user_key
    while True:
        await asyncio.sleep(_device_state.get("interval", 5))
        try:
            resp = requests.post(
                "https://enter.pollinations.ai/api/device/token",
                json={"device_code": _device_state["device_code"]},
                timeout=10,
            )
            data = resp.json()
            if resp.status_code == 200:
                _user_key = data["access_token"]
                _save_key(_user_key, data.get("expires_in", 604800))
                return _user_key
            if data.get("error") in ("authorization_pending", "slow_down"):
                continue
            raise RuntimeError(f"Auth failed: {data}")
        except RuntimeError:
            raise
        except Exception as e:
            print(f"[BYOP] poll error: {e}")
            continue
