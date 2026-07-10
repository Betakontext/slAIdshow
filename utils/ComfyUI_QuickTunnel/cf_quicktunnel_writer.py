#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Cloudflared Quick Tunnel URL Writer with:
- built-in stdlib HTTP pull endpoint (for LAN pull)
- optional FTPS upload to stable public web space (Greensta) for Internet pull
- .env support (stdlib-only) for credentials and configuration

Why:
- Quick Tunnels rotate hostnames. To provide a stable pull URL for slAIdshow,
  we additionally upload tunnel_url.json/txt to a fixed HTTPS location on your web host.
- This bypasses CGNAT and avoids router port forwards.

Endpoints served locally (optional for LAN):
  GET /bridge/tunnel_url.json
  GET /bridge/tunnel_url.txt
  GET /health
  GET /

FTPS upload (optional, stdlib-only via ftplib.FTP_TLS):
  - When enabled, after each URL change, upload JSON and TXT to the remote dir.
  - Creates subdirectories if missing (best-effort).

.env usage:

- Place a .env file next to this script (or provide --env-file) with lines like:
    FTPS_ENABLE=true
    FTPS_HOST=
    FTPS_USER=
    FTPS_PASS=
    FTPS_DIR=
    COMFY_URL=http://127.0.0.1:8188
    HTTP_HOST=0.0.0.0
    HTTP_PORT=8799
    HTTP_CORS=false
    CLOUDFLARED=cloudflared
    OUT_DIR=./bridge_output
    EDGE_PROTOCOL=http2

- Values may be quoted: KEY="value with spaces"

Priority:
- CLI arguments override environment variables.
- .env is loaded into os.environ before full CLI parsing.
- If --env-file is omitted, the loader auto-discovers .env next to the script,
  then in current working directory.

Prerequisites:

-> Place this file and the .env.example into the folder where you have your cloudflared.exe
-> Fill .env.example with your host adress and keys and save it as .env

Typical start:

with .env in the same directory:

    python cf_quicktunnel_writer.py

without .env:

  python cf_quicktunnel_writer.py ^
    --cloudflared "cloudflared-windows-amd64.exe" ^
    --comfy-url "http://127.0.0.1:8188" ^
    --out-dir "C:\Users\Administrator\Documents\Arbeiten\0000_DEV\ComfyUI\output" ^
    --http-host 0.0.0.0 --http-port 8799 ^
    --ftps-enable ^
    --ftps-host "*****" ^
    --ftps-user "********" ^
    --ftps-dir "*****"

Security tips:
- Do NOT commit credentials. Prefer the .env file excluded via .gitignore, or use process env vars.
- FTPS uses TLS but still verify you trust the hosting.

Test checklist:
1) Start cloudflared binary is reachable (or specify --cloudflared).
2) Start the script without FTPS to verify local endpoints:
   - curl http://127.0.0.1:8799/health
   - curl http://127.0.0.1:8799/bridge/tunnel_url.json
3) Enable FTPS and check uploads on URL change. Ensure remote path is correct.
4) Try with .env only (no CLI), then override one value via CLI to confirm priority.
"""

import argparse
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple

# ========== Cloudflared URL detection ==========
TRYCF_RE = re.compile(r"https://[a-z0-9\-]+\.trycloudflare\.com", re.IGNORECASE)

# ========== .env loader (stdlib-only) ==========

def _strip_quotes(value: str) -> str:
    """Strip wrapping single or double quotes if present."""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value

def load_env_file(path: str, overwrite: bool = False) -> int:
    """
    Load KEY=VALUE pairs from a .env file into os.environ.
    - Supports comments (#), blank lines, quoted values.
    - overwrite=False means: do not override existing environment keys.
    Returns number of keys set/updated.
    """
    set_count = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                # Allow inline comments only if preceded by a space
                # but keep it simple and only support full-line comments.
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                val = _strip_quotes(val)
                if not key:
                    continue
                if overwrite or (key not in os.environ):
                    os.environ[key] = val
                    set_count += 1
        return set_count
    except FileNotFoundError:
        return 0
    except Exception as e:
        print(f"[env] failed to load {path}: {e}", flush=True)
        return 0

def auto_discover_env_file(script_dir: str, cwd: str) -> Optional[str]:
    """
    Returns the first existing .env path among:
      1) script_dir/.env
      2) cwd/.env (if different from script_dir)
    """
    candidates = [
        os.path.join(script_dir, ".env"),
    ]
    if os.path.abspath(cwd) != os.path.abspath(script_dir):
        candidates.append(os.path.join(cwd, ".env"))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

def env_flag_truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")

# ========== Helpers ==========

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write(path: str, data: bytes):
    """Atomically write bytes to a file path (safe replace)."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    base = os.path.basename(path)
    fd, tmppath = tempfile.mkstemp(prefix=base + ".", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmppath, path)
    except Exception:
        try:
            os.unlink(tmppath)
        except Exception:
            pass
        raise


def read_json_file(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_text_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


def parse_url_parts(url: str) -> Tuple[str, str, int]:
    """Parse scheme, host, port with stdlib only."""
    scheme = "https" if url.lower().startswith("https://") else "http"
    rest = url.split("://", 1)[1] if "://" in url else url
    hostport = rest.split("/", 1)[0]
    if ":" in hostport:
        host, port_s = hostport.rsplit(":", 1)
        try:
            port = int(port_s)
        except ValueError:
            port = 443 if scheme == "https" else 80
    else:
        host = hostport
        port = 443 if scheme == "https" else 80
    return scheme, host, port


# ========== FTPS Upload (stdlib: ftplib) ==========
from ftplib import FTP_TLS, error_perm, all_errors as ftplib_errors
from contextlib import contextmanager

@contextmanager
def ftps_connect(host: str, user: str, password: str, timeout: float = 20.0):
    """
    Context manager to connect/login to FTPS (explicit TLS).
    Performs PROT P to encrypt data channel.
    """
    ftps = FTP_TLS()
    ftps.connect(host=host, timeout=timeout)
    ftps.auth()  # upgrade control connection to TLS
    ftps.prot_p()  # protect data channel
    ftps.login(user=user, passwd=password)
    try:
        yield ftps
    finally:
        try:
            ftps.quit()
        except Exception:
            try:
                ftps.close()
            except Exception:
                pass


def _ftps_mkdirs(ftps: FTP_TLS, remote_dir: str):
    """
    Create remote_dir and parents if missing. Tries to CWD stepwise and MKD on failure.
    Works for typical FTP servers; silently continues if dirs exist.
    """
    if not remote_dir or remote_dir == "/":
        return
    parts = [p for p in remote_dir.strip("/").split("/") if p]
    path_so_far = ""
    try:
        ftps.cwd("/")
    except Exception:
        pass
    for p in parts:
        path_so_far = path_so_far + "/" + p
        try:
            ftps.cwd(path_so_far)
        except Exception:
            try:
                ftps.mkd(path_so_far)
            except ftplib_errors as e:
                msg = str(e).lower()
                if "exists" not in msg and "file unavailable" not in msg:
                    raise
            ftps.cwd(path_so_far)


def ftps_upload_file(host: str, user: str, password: str, local_path: str, remote_dir: str, remote_filename: Optional[str] = None, timeout: float = 20.0):
    """
    Upload local_path to host:/remote_dir/remote_filename via FTPS with binary transfer.
    Ensures remote_dir exists (best-effort).
    """
    if remote_filename is None:
        remote_filename = os.path.basename(local_path)
    with ftps_connect(host, user, password, timeout=timeout) as ftps:
        _ftps_mkdirs(ftps, remote_dir)
        if remote_dir and remote_dir != "/":
            ftps.cwd(remote_dir)
        with open(local_path, "rb") as lf:
            ftps.storbinary(f"STOR {remote_filename}", lf)


def upload_with_retries(host: str, user: str, password: str, local_path: str, remote_dir: str, remote_filename: Optional[str] = None, retries: int = 5, base_delay: float = 1.0):
    """
    Retry FTPS upload with exponential backoff.
    """
    attempt = 0
    last_exc = None
    while attempt < retries:
        try:
            ftps_upload_file(host, user, password, local_path, remote_dir, remote_filename)
            print(f"[ftps] uploaded {local_path} -> {host}:{remote_dir}/{remote_filename or os.path.basename(local_path)}", flush=True)
            return True
        except Exception as e:
            last_exc = e
            delay = base_delay * (1.7 ** attempt)
            print(f"[ftps] upload failed (attempt {attempt+1}/{retries}): {e}; retrying in {delay:.1f}s", flush=True)
            time.sleep(delay)
            attempt += 1
    print(f"[ftps] upload permanently failed: {last_exc}", flush=True)
    return False


# ========== Cloudflared writer ==========

class TunnelWriter:
    """
    Manage cloudflared subprocess, detect public URL, persist JSON/TXT, and optionally FTPS-upload.
    """

    def __init__(self, cf_bin: str, comfy_url: str, out_dir: str, protocol: str = "http2",
                 ftps_enable: bool = False, ftps_host: str = "", ftps_user: str = "", ftps_pass: str = "",
                 ftps_dir: str = "", ftps_retries: int = 5):
        self.cf_bin = cf_bin
        self.comfy_url = comfy_url
        self.out_dir = out_dir
        self.protocol = protocol
        self.proc: Optional[subprocess.Popen] = None
        self.stop_evt = threading.Event()
        self.current_url = ""
        self.backoff = 2.0

        # FTPS config
        self.ftps_enable = ftps_enable
        self.ftps_host = ftps_host
        self.ftps_user = ftps_user
        self.ftps_pass = ftps_pass
        self.ftps_dir = ftps_dir.rstrip("/") if ftps_dir else ""
        self.ftps_retries = ftps_retries

        os.makedirs(self.out_dir, exist_ok=True)
        self.json_path = os.path.join(self.out_dir, "tunnel_url.json")
        self.txt_path = os.path.join(self.out_dir, "tunnel_url.txt")

    def write_urls_local(self, url: str):
        """Persist URL to .json and .txt atomically (local)."""
        payload = {"url": url, "updated_at": utc_now_iso()}
        data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        data_txt = (url or "").encode("utf-8")
        atomic_write(self.json_path, data_json)
        atomic_write(self.txt_path, data_txt)
        print(f"[writer] wrote {self.json_path} and {self.txt_path}", flush=True)

    def upload_remote_if_enabled(self):
        """Upload local JSON and TXT to FTPS target when enabled."""
        if not self.ftps_enable:
            return
        if not (self.ftps_host and self.ftps_user and self.ftps_pass and self.ftps_dir):
            print("[ftps] missing credentials or remote dir; skipping upload", flush=True)
            return
        # Upload JSON
        upload_with_retries(self.ftps_host, self.ftps_user, self.ftps_pass,
                            self.json_path, self.ftps_dir, remote_filename="tunnel_url.json",
                            retries=self.ftps_retries)
        # Upload TXT
        upload_with_retries(self.ftps_host, self.ftps_user, self.ftps_pass,
                            self.txt_path, self.ftps_dir, remote_filename="tunnel_url.txt",
                            retries=self.ftps_retries)

    def run_once(self):
        cmd = [
            self.cf_bin,
            "tunnel",
            "--no-autoupdate",
            "--protocol",
            self.protocol,
            "--url",
            self.comfy_url,
        ]
        print(f"[cloudflared] starting: {' '.join(cmd)}", flush=True)
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                universal_newlines=True,
            )
        except Exception as e:
            print(f"[cloudflared] spawn error: {e}", flush=True)
            return False

        assert self.proc.stdout is not None
        try:
            for line in self.proc.stdout:
                if self.stop_evt.is_set():
                    break
                line = line.rstrip("\r\n")
                print(f"[cloudflared] {line}", flush=True)
                m = TRYCF_RE.search(line)
                if m:
                    url = m.group(0)
                    if url != self.current_url:
                        self.current_url = url
                        print(f"[cloudflared] detected tunnel URL: {url}", flush=True)
                        try:
                            self.write_urls_local(url)
                            # Attempt FTPS upload after local write
                            self.upload_remote_if_enabled()
                        except Exception as we:
                            print(f"[writer] error after detection: {we}", flush=True)
        except Exception as e:
            print(f"[cloudflared] read error: {e}", flush=True)

        # Wait for exit
        try:
            rc = self.proc.wait(timeout=2)
        except Exception:
            rc = None
        print(f"[cloudflared] exited rc={rc}", flush=True)
        return True

    def run_forever(self):
        while not self.stop_evt.is_set():
            started = self.run_once()
            if self.stop_evt.is_set():
                break
            # Backoff and retry on crash/failure
            t = self.backoff
            self.backoff = min(self.backoff * 1.5, 30.0)
            for _ in range(int(t / 0.1)):
                if self.stop_evt.is_set():
                    break
                time.sleep(0.1)

    def is_running(self) -> bool:
        try:
            return self.proc is not None and self.proc.poll() is None
        except Exception:
            return False

    def stop(self):
        self.stop_evt.set()
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass


# ========== HTTP server for LAN pull ==========

class BridgeRequestHandler(BaseHTTPRequestHandler):
    """
    Serves minimal endpoints from local files:
      - GET /bridge/tunnel_url.json
      - GET /bridge/tunnel_url.txt
      - GET /health
      - GET /
    """

    out_dir: str = "."
    json_path: str = "tunnel_url.json"
    txt_path: str = "tunnel_url.txt"
    enable_cors: bool = False
    writer_ref: Optional[TunnelWriter] = None

    server_version = "BridgeServer/1.2"
    sys_version = ""

    def _set_common_headers(self, status: int, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if self.enable_cors:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        if self.enable_cors:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        try:
            if self.path == "/":
                self._set_common_headers(HTTPStatus.OK, "text/plain; charset=utf-8")
                body = (
                    "Cloudflared Quick Tunnel Bridge\n"
                    "Endpoints:\n"
                    "  GET /bridge/tunnel_url.json\n"
                    "  GET /bridge/tunnel_url.txt\n"
                    "  GET /health\n"
                )
                self.wfile.write(body.encode("utf-8"))
                return

            if self.path == "/bridge/tunnel_url.json":
                data = read_json_file(self.json_path)
                if data is None:
                    self._set_common_headers(HTTPStatus.NOT_FOUND, "application/json")
                    self.wfile.write(b'{"error":"not_found"}')
                    return
                payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self._set_common_headers(HTTPStatus.OK, "application/json")
                self.wfile.write(payload)
                return

            if self.path == "/bridge/tunnel_url.txt":
                txt = read_text_file(self.txt_path)
                if txt is None:
                    self._set_common_headers(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8")
                    self.wfile.write(b"")
                    return
                self._set_common_headers(HTTPStatus.OK, "text/plain; charset=utf-8")
                self.wfile.write(txt.encode("utf-8"))
                return

            if self.path == "/health":
                data = read_json_file(self.json_path) or {}
                url = data.get("url") if isinstance(data, dict) else None
                running = False
                if self.writer_ref is not None:
                    running = self.writer_ref.is_running()
                resp = {
                    "status": "ok",
                    "cloudflared_running": bool(running),
                    "url": url or "",
                    "updated_at": data.get("updated_at") if isinstance(data, dict) else None,
                }
                payload = json.dumps(resp, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self._set_common_headers(HTTPStatus.OK, "application/json")
                self.wfile.write(payload)
                return

            self._set_common_headers(HTTPStatus.NOT_FOUND, "application/json")
            self.wfile.write(b'{"error":"not_found"}')
        except BrokenPipeError:
            pass
        except Exception as e:
            try:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, explain=str(e))
            except Exception:
                pass

    def log_message(self, fmt, *args):
        sys.stdout.write("[http] " + (fmt % args) + "\n")


class HttpServerThread(threading.Thread):
    def __init__(self, host: str, port: int, handler_cls: type):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.handler_cls = handler_cls
        self.httpd: Optional[ThreadingHTTPServer] = None
        self._stopped = threading.Event()

    def run(self):
        try:
            ThreadingHTTPServer.allow_reuse_address = True
            self.httpd = ThreadingHTTPServer((self.host, self.port), self.handler_cls)
            sa = self.httpd.socket.getsockname()
            print(f"[http] serving on {sa[0]}:{sa[1]}", flush=True)
            self.httpd.serve_forever(poll_interval=0.5)
        except OSError as e:
            print(f"[http] failed to bind {self.host}:{self.port} -> {e}", flush=True)
        except Exception as e:
            print(f"[http] server error: {e}", flush=True)
        finally:
            self._stopped.set()
            print("[http] server thread exited", flush=True)

    def stop(self):
        try:
            if self.httpd:
                self.httpd.shutdown()
                self.httpd.server_close()
        except Exception:
            pass
        try:
            with socket.create_connection((self.host, self.port), timeout=0.2):
                pass
        except Exception:
            pass
        self._stopped.wait(timeout=3.0)


# ========== Utilities ==========

def find_default_cloudflared(script_path: str) -> str:
    base = os.path.dirname(os.path.abspath(script_path))
    candidates = [
        os.path.join(base, "cloudflared-windows-amd64.exe"),
        os.path.join(base, "cloudflared.exe"),
        "cloudflared",
    ]
    for c in candidates:
        if c == "cloudflared":
            return c
        if os.path.isfile(c):
            return c
    return "cloudflared"


# ========== Main ==========

def parse_stage1_args(argv=None):
    """Parse minimal args to control .env loading."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--env-file", type=str, default="", help="Path to .env file to load before full parsing")
    p.add_argument("--no-auto-env", action="store_true", help="Disable auto-discovery of .env")
    return p.parse_known_args(argv)

def parse_stage2_args(argv=None):
    """Full parser using environment defaults (possibly populated from .env)."""
    p = argparse.ArgumentParser(description="Cloudflared Quick Tunnel URL Writer with LAN HTTP endpoint, optional FTPS upload, and .env support")
    # Primary settings
    p.add_argument("--cloudflared", type=str, default=os.getenv("CLOUDFLARED", ""), help="Path to cloudflared binary (auto-discover if empty or 'cloudflared')")
    p.add_argument("--comfy-url", type=str, default=os.getenv("COMFY_URL", "http://127.0.0.1:8188"), help="Local ComfyUI URL to expose")
    p.add_argument("--out-dir", type=str, default=os.getenv("OUT_DIR", ""), help="Directory to write tunnel_url.json and tunnel_url.txt")
    p.add_argument("--protocol", type=str, default=os.getenv("EDGE_PROTOCOL", "http2"), choices=["quic", "http2"], help="Cloudflared edge protocol")

    # HTTP endpoint (LAN)
    p.add_argument("--http-host", type=str, default=os.getenv("HTTP_HOST", "0.0.0.0"), help="HTTP bind host for pull endpoint")
    p.add_argument("--http-port", type=int, default=int(os.getenv("HTTP_PORT", "8799")), help="HTTP bind port for pull endpoint")
    default_cors_env = os.getenv("HTTP_CORS", "false")
    p.add_argument("--http-cors", action="store_true", default=env_flag_truthy(default_cors_env), help="Enable CORS (Access-Control-Allow-Origin: *)")

    # FTPS upload options
    default_ftps_enable = env_flag_truthy(os.getenv("FTPS_ENABLE", "false"))
    p.add_argument("--ftps-enable", action="store_true", default=default_ftps_enable, help="Enable FTPS upload on URL changes")
    p.add_argument("--ftps-host", type=str, default=os.getenv("FTPS_HOST", ""), help="FTPS host (e.g., web8.greensta.de)")
    p.add_argument("--ftps-user", type=str, default=os.getenv("FTPS_USER", ""), help="FTPS username")
    p.add_argument("--ftps-pass", type=str, default=os.getenv("FTPS_PASS", ""), help="FTPS password")
    p.add_argument("--ftps-dir", type=str, default=os.getenv("FTPS_DIR", ""), help="Remote directory path (e.g., /dev.betakontext.de/slAIdshow/bridge)")
    p.add_argument("--ftps-retries", type=int, default=int(os.getenv("FTPS_RETRIES", "5")), help="Retry count for FTPS uploads")

    # Keep stage1 flags too for help visibility
    p.add_argument("--env-file", type=str, default=os.getenv("ENV_FILE", ""), help="Path to .env file (already loaded if provided earlier)")
    p.add_argument("--no-auto-env", action="store_true", default=env_flag_truthy(os.getenv("NO_AUTO_ENV", "false")), help="Disable auto-discovery of .env")

    return p.parse_args(argv)

def main():
    # Stage 1: early parse for .env loading controls
    stage1_args, remaining = parse_stage1_args()

    # Load .env if provided, else auto-discover unless disabled
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    env_loaded_from = None

    if stage1_args.env_file:
        count = load_env_file(stage1_args.env_file, overwrite=False)
        env_loaded_from = stage1_args.env_file if count >= 0 else None
        print(f"[env] loaded {count} keys from {stage1_args.env_file}", flush=True)
    elif not stage1_args.no_auto_env:
        auto_env = auto_discover_env_file(script_dir, cwd)
        if auto_env:
            count = load_env_file(auto_env, overwrite=False)
            env_loaded_from = auto_env if count >= 0 else None
            print(f"[env] auto-loaded {count} keys from {auto_env}", flush=True)
        else:
            print("[env] no .env discovered", flush=True)
    else:
        print("[env] auto-discovery disabled", flush=True)

    # Stage 2: full parse with defaults from os.environ (potentially populated)
    args = parse_stage2_args(remaining)

    # Resolve cloudflared binary and out dir
    cf_bin = (args.cloudflared or "").strip() or find_default_cloudflared(__file__)
    out_dir = (args.out_dir or "").strip() or (os.path.dirname(os.path.abspath(__file__)) or ".")

    # Mask sensitive info for logs
    masked_pass = "****" if args.ftps_pass else ""

    print(
        f"[main] cloudflared={cf_bin}, comfy={args.comfy_url}, out_dir={out_dir}, "
        f"http={args.http_host}:{args.http_port}, protocol={args.protocol}, cors={args.http_cors}, "
        f"ftps_enable={args.ftps_enable}, ftps_host={args.ftps_host}, ftps_user={args.ftps_user}, "
        f"ftps_pass={masked_pass}, ftps_dir={args.ftps_dir}, ftps_retries={args.ftps_retries}, "
        f"env_file={(env_loaded_from or 'none')}",
        flush=True,
    )

    # Validate FTPS config if enabled
    if args.ftps_enable:
        missing = []
        if not args.ftps_host:
            missing.append("FTPS_HOST/--ftps-host")
        if not args.ftps_user:
            missing.append("FTPS_USER/--ftps-user")
        if not args.ftps_pass:
            missing.append("FTPS_PASS/--ftps-pass")
        if not args.ftps_dir:
            missing.append("FTPS_DIR/--ftps-dir")
        if missing:
            print(f"[main] error: --ftps-enable requires: {', '.join(missing)}", flush=True)
            sys.exit(2)

    # Prepare writer
    tw = TunnelWriter(
        cf_bin=cf_bin,
        comfy_url=args.comfy_url,
        out_dir=out_dir,
        protocol=args.protocol,
        ftps_enable=bool(args.ftps_enable),
        ftps_host=args.ftps_host.strip(),
        ftps_user=args.ftps_user.strip(),
        ftps_pass=args.ftps_pass,  # keep as-is
        ftps_dir=args.ftps_dir.strip(),
        ftps_retries=args.ftps_retries,
    )

    # Prepare HTTP handler class with shared config
    BridgeRequestHandler.out_dir = out_dir
    BridgeRequestHandler.json_path = os.path.join(out_dir, "tunnel_url.json")
    BridgeRequestHandler.txt_path = os.path.join(out_dir, "tunnel_url.txt")
    BridgeRequestHandler.enable_cors = bool(args.http_cors)
    BridgeRequestHandler.writer_ref = tw

    # Start HTTP server thread (so LAN pull works even before URL appears)
    http_thread = HttpServerThread(args.http_host, args.http_port, BridgeRequestHandler)
    http_thread.start()

    def handle_signal(signum, frame):
        print("[main] shutdown requested", flush=True)
        try:
            tw.stop()
        except Exception:
            pass
        try:
            http_thread.stop()
        except Exception:
            pass

    # Register signal handlers (best effort on Windows)
    try:
        signal.signal(signal.SIGINT, handle_signal)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, handle_signal)
    except Exception:
        pass

    # Run writer loop (blocking)
    try:
        tw.run_forever()
    finally:
        try:
            tw.stop()
        except Exception:
            pass
        try:
            http_thread.stop()
        except Exception:
            pass
        print("[main] stopped", flush=True)


if __name__ == "__main__":
    main()
