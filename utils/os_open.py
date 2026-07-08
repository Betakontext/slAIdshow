# utils/os_open.py
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Dict


class OpenDirError(RuntimeError):
    pass


def _clean_env_for_gui() -> Dict[str, str]:
    """
    Return a copy of os.environ with Qt-related variables stripped to avoid
    accidental plugin loading from OpenCV/Qt in virtualenvs.
    """
    env = dict(os.environ)
    for var in (
        "QT_QPA_PLATFORM",
        "QT_PLUGIN_PATH",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        "QT_DEBUG_PLUGINS",
        "QTWEBENGINE_CHROMIUM_FLAGS",
        "QT_DIR",
    ):
        env.pop(var, None)
    # Also avoid PYTHONPATH bleeding into spawned GUI apps
    env.pop("PYTHONPATH", None)
    return env


def open_folder_os(path: Path) -> None:
    """
    Open a folder in the native file explorer safely.
    - Windows: explorer / os.startfile (WSL via powershell.exe)
    - macOS: open
    - Linux: xdg-open (fallbacks: nautilus, dolphin, thunar, pcmanfm, nemo)
    Notes:
    - Uses a scrubbed environment to avoid Qt plugin poisoning from venvs.
    - Treats xdg-open timeout as success (GUI takes over asynchronously).
    Raises OpenDirError on hard failures only.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)

    system = platform.system()
    env = _clean_env_for_gui()

    try:
        if system == "Windows":
            if "WSL_DISTRO_NAME" in os.environ:
                win_path = subprocess.check_output(["wslpath", "-w", str(p)], text=True, env=env).strip()
                subprocess.Popen(
                    ["powershell.exe", "-NoProfile", "-Command", f"Start-Process '{win_path}'"],
                    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                return
            else:
                try:
                    os.startfile(str(p))  # type: ignore[attr-defined]
                    return
                except Exception:
                    subprocess.Popen(["explorer", str(p)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return

        if system == "Darwin":
            proc = subprocess.Popen(["open", str(p)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            rc = proc.wait(timeout=5)
            if rc != 0:
                err = (proc.stderr.read() or b"").decode("utf-8", "ignore")
                raise OpenDirError(f"open failed (rc={rc}): {err.strip()}")
            return

        # Linux and others
        xdg = shutil.which("xdg-open")
        if xdg:
            proc = subprocess.Popen([xdg, str(p)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                rc = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Consider success: the desktop handler should take over
                return
            if rc != 0:
                # Try explicit file managers as fallback before raising
                for fm in ("nautilus", "dolphin", "thunar", "pcmanfm", "nemo"):
                    fm_path = shutil.which(fm)
                    if fm_path:
                        subprocess.Popen([fm_path, str(p)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        return
                err = (proc.stderr.read() or b"").decode("utf-8", "ignore")
                raise OpenDirError(f"xdg-open failed (rc={rc}): {err.strip()}")
            return

        # Final fallback when xdg-open is missing
        for fm in ("nautilus", "dolphin", "thunar", "pcmanfm", "nemo"):
            fm_path = shutil.which(fm)
            if fm_path:
                subprocess.Popen([fm_path, str(p)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return

        raise OpenDirError("No GUI file manager found (xdg-open and common FMs are missing).")

    except OpenDirError:
        raise
    except Exception as e:
        raise OpenDirError(f"Failed to open folder: {e}") from e
