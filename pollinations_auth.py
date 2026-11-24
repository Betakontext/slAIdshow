# pollinations_auth.py
import os, time, requests

APP_KEY = os.environ.get("POLLINATIONS_APP_KEY", "")  # pk_... from enter.pollinations.ai/keys
_user_key: str | None = None  # in-memory, never persisted

def start_device_flow() -> dict:
    """Initiate device flow, return info for the UI."""
    resp = requests.post(
        "https://enter.pollinations.ai/api/device/code",
        json={"client_id": APP_KEY},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    # Store device_code for the polling coroutine
    _device_state["device_code"] = data["device_code"]
    _device_state["interval"] = data.get("interval", 5)
    return {
        "verification_uri": data["verification_uri_complete"],
        "user_code": data["user_code"],
        "expires_in": data["expires_in"],
    }

_device_state: dict = {}

async def poll_for_token() -> str:
    """Poll until user approves (called from a background task)."""
    import asyncio
    while True:
        await asyncio.sleep(_device_state.get("interval", 5))
        resp = requests.post(
            "https://enter.pollinations.ai/api/device/token",
            json={"device_code": _device_state["device_code"]},
            timeout=10,
        )
        data = resp.json()
        if resp.status_code == 200:
            global _user_key
            _user_key = data["access_token"]  # sk_...
            return _user_key
        if data.get("error") == "authorization_pending":
            continue
        if data.get("error") == "slow_down":
            continue  # just keep polling
        raise RuntimeError(f"Auth failed: {data}")

def get_key() -> str | None:
    """Return BYOP key if connected, else None."""
    return _user_key  # None, if now device connected
