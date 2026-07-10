### Cloudflared Quick Tunnel URL Bridge

I build this tool to get a cloudflared quicktunnel URL to use ComfyUI from my local network.

-------------------------

It'a s small, stdlib-only Python tool to expose ComfyUI with Cloudflared Quick Tunnel, write the current URL to files/HTTP, and upload to web space via FTPS.:

- Starts a Cloudflared Quick Tunnel to expose a local ComfyUI (or any HTTP service).
- Detects the rotating Quick Tunnel URL from Cloudflared logs.
- Writes the current public URL to local files (tunnel_url.json/txt).
- Optionally uploads those files via FTPS to a stable web space, so other devices can fetch the active URL.
- Serves a tiny HTTP endpoint for LAN clients to read the current URL.
- Supports configuration via a .env file (no extra dependencies).

This helps bypass CGNAT and router port forwarding by providing a stable location (your public web space) where your other device(s) can pull the current tunnel URL.

#### Features

- Stdlib only (no external Python deps).
- Cloudflared subprocess management with auto-restart/backoff.
- Robust URL detection (trycloudflare.com).
- Atomic local file writes.
- Optional FTPS upload (FTP over TLS) with retries and auto-directory creation.
- Lightweight HTTP pull server with:
  - GET /bridge/tunnel_url.json
  - GET /bridge/tunnel_url.txt
  - GET /health
  - GET /

#### Requirements

- Python 3.8+
- Cloudflared binary available on PATH or alongside the script
- Optional: FTPS-capable hosting (explicit TLS)

#### Quick Start

1) Install Cloudflared:
- https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
- Place cloudflared.exe next to the script or ensure it’s on PATH.

----------------------
-> Specify your cloudflared binary via .env or CLI (no need to edit the script):

In .env:

    CLOUDFLARED=.\cloudflared-windows-amd64.exe (Windows) or CLOUDFLARED=/usr/local/bin/cloudflared (Linux/macOS)

Or via CLI:

    --cloudflared "<path-to-cloudflared>"

If unset, the script auto-discovers cloudflared next to the script or on PATH

----------------------

Create a .env file next to the script (recommended):

    COMFY_URL=http://127.0.0.1:8188 # ComfyUI (or any local HTTP service you want to expose)
    OUT_DIR=./bridge_output  # Output directory for local copies of tunnel_url.json/txt
    HTTP_HOST=0.0.0.0 # HTTP pull endpoint (LAN)
    HTTP_PORT=8799
    HTTP_CORS=false
    EDGE_PROTOCOL=http2 # Cloudflared edge protocol
    CLOUDFLARED=cloudflared # Optional: explicit cloudflared path/name
    FTPS_ENABLE=false # enable to upload URL files to your web space
    FTPS_HOST=
    FTPS_USER=
    FTPS_PASS=
    FTPS_DIR=
    FTPS_RETRIES=5 # optional


#### Run:

    python cf_quicktunnel_writer.py

The script auto-discovers .env next to the script. You can also pass a custom file:

    python cf_quicktunnel_writer.py --env-file ./.env

4) Check the local health endpoint:

- http://127.0.0.1:8799/health
- http://127.0.0.1:8799/bridge/tunnel_url.json

When Cloudflared prints the public URL (e.g., https://abc123.trycloudflare.com), the script writes:
- OUT_DIR/tunnel_url.json
- OUT_DIR/tunnel_url.txt

If FTPS is enabled and configured, those files are uploaded to FTPS_DIR as tunnel_url.json and tunnel_url.txt.

#### CLI Overrides

CLI arguments override .env values. Examples:

Use a specific cloudflared binary:

    python cf_quicktunnel_writer.py --cloudflared ".\cloudflared-windows-amd64.exe"

Change the HTTP port temporarily:

    python cf_quicktunnel_writer.py --http-port 8800

Enable FTPS for this run (assuming the rest is in .env):

    python cf_quicktunnel_writer.py --ftps-enable

Full set of flags is available via:

    python cf_quicktunnel_writer.py --help



#### Endpoints (local pull)

- GET /bridge/tunnel_url.json → {"url":"https://...trycloudflare.com","updated_at":"...Z"}
- GET /bridge/tunnel_url.txt → plain URL text
- GET /health → {"status":"ok","cloudflared_running":true,"url":"...","updated_at":"...Z"}
- GET / → brief info page

These endpoints are served by a simple threaded HTTP server for LAN consumption. CORS can be enabled with HTTP_CORS=true or --http-cors.

#### FTPS Upload

When FTPS_ENABLE=true (or --ftps-enable), the script uploads both files to your FTPS host and directory every time the tunnel URL changes.

- FTPS (explicit TLS) via Python’s ftplib.FTP_TLS
- Auto-creates nested directories best-effort
- Retries with exponential backoff on failures
- Password is never printed in logs

Ensure your hosting supports FTPS (AUTH TLS) and that your credentials and directory are correct.

#### Security

Never commit credentials. Keep .env out of version control:

    echo ".env" >> .gitignore

If a file is already tracked, run `git rm --cached .env` and commit.

The script masks FTPS_PASS in logs. Review your hosting’s certificate and trust settings if needed.

#### Windows Notes

Prefer launching with the Python you expect:

    py -3 cf_quicktunnel_writer.py

If cloudflared.exe is in the same folder, no extra flag is needed. Otherwise pass --cloudflared with the path.

#### Troubleshooting

- cloudflared not found:
- Put the binary next to the script or set CLOUDFLARED in .env, or install on PATH.
- No URL detected:
- Check that Cloudflared prints a trycloudflare.com URL in stdout.
- Verify COMFY_URL is reachable (e.g., http://127.0.0.1:8188).
- FTPS upload failing:
- Confirm FTPS host, user, pass, and dir.
- Verify FTPS (AUTH TLS) support.
- Check server logs for permissions.
- Port in use:
- Change HTTP_PORT or stop the conflicting service.



#### License

MIT

#### Disclaimer

Use responsibly and according to your Cloudflare and hosting provider terms. This tool is provided “as is,” without warranty of any kind.

Contact: dev@betakontext.de | https://dev.betakontext.de

If you like it and want to support further developments, feel free to fork and buy me a coffee: https://buymeacoffee.com/betakontext

