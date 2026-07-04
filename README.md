### slAIdshow | Local AI Speech‑to‑Image Generator

### Whisper.cpp → Ollama → ComfyUI/Pollinations pipeline for live illustrations (local, Browser UI)

slAIdshow turns live voice into near‑real‑time images for talks, readings, and presentations. It runs locally with Whisper.cpp (via pywhispercpp) for transcription, uses a local LLM via Ollama for prompt optimization, and generates images via ComfyUI (local, same‑LAN, or remote). Alternatively, you can switch to Pollinations cloud for image generation. A simple browser UI provides Start/Stop for mic input, a text prompt field, optional LLM prompt optimization, a backend selector (Comfy local/LAN, Comfy remote, Pollinations), and a ComfyUI workflow selector. With the default setup, new images appear every 6–10 seconds, along with live transcripts and the latest prompts.

Ollama Vision can interpret a reference image across all three modes (local/LAN, remote, cloud) when enabled. The UI is served from /web; the root (/) redirects to /web/index.html. Images are saved under outputs/images and served at /static.

---

### Features

- Local Browser UI (FastAPI) with Start/Stop controls
- Live audio capture with periodic transcription snapshots via pywhispercpp (configurable, e.g., every 3–6 s)
- Prompt optimization via Ollama (/api/generate and /api/chat), tuned for low latency (low temperature, short context)
- Image generation backends:
  - ComfyUI local (127.0.0.1:8188) or same‑LAN host (ComfyUI listening on 0.0.0.0)
  - ComfyUI remote (reachable from another network via VPN/tunnel/reverse proxy)
  - Pollinations cloud (always reachable; requires API key)
- Live updates via Server‑Sent Events (SSE): status, transcripts, prompts, image job notifications
- Style Engine across all backends:
  - Free‑text style prompt
  - Optional reference image (upload or URL)
  - Optional Ollama Vision reference interpretation
  - Stable merge: vision tags + style text with deduplication and token caps
- Negative prompt injection, width/height control, ComfyUI workflow selector
- Asynchronous pipeline (asyncio) overlapping long image jobs
- Robust retry logic with timeouts and exponential backoff for Ollama, ComfyUI, Pollinations
- Local‑first defaults for safety; LAN/remote/cloud are opt‑in via .env

---

### System Requirements

- OS: Linux tested (PipeWire/PulseAudio). macOS/Windows should work with adjusted device names.
- Python: 3.9+ (3.10+ recommended)
- Working microphone
- pywhispercpp installed locally (transcription)
- Ollama installed and running locally (LLM)
- ComfyUI accessible (local/LAN/remote) with API on port 8188
- Optional: Pollinations account + API key

Hardware notes:
- Runs smoothly on a single RTX 3060 (6 GB). CPU‑only works but is slower.

---

### Repository Layout

	├── app.py
	├── comfyui_bridge.py
	├── comfyui.service
	├── data
	│   ├── references
	│   └── refs
	├── image_backend.py
	├── models
	│   ├── ggml-base.bin
	│   └── ggml-tiny.bin
	├── outputs
	│   ├── images
	├── README.md
	├── requirements.txt
	├── run.ps1
	├── run.sh
	├── slAIdshow_summary.txt
	├── static
	│   └── uploads
	├── style_engine.py
	├── utils
	│   ├── audio_test.py
	│   ├── dev_check.py
	│   ├── mic_check_whisper.py
	│   ├── test_comfy_local.py
	│   └── verify_runtime.py
	│   ├── test_pollinations_style_ref_auto_style.py
	│   └── style_features.py
	├── web
	│   └── index.html
	└── workflows
		└── text2img_SD15-FP16.json

---

### Prerequisites

Linux (Debian/Ubuntu) — native deps for audio I/O and builds:

	sudo apt update
	sudo apt install -y build-essential cmake pkg-config python3-dev libportaudio2 libasound2-dev
	# OS libraries used by sounddevice/PortAudio and potential builds; not part of requirements.txt.

macOS:

	xcode-select --install
	brew install portaudio

Windows:

- Recommended: Use the PowerShell script below (creates a venv and installs wheels).
- If building from source, you may need Microsoft C++ Build Tools.
- If native deps are problematic, consider WSL (use the Linux steps).

---

### Step‑by‑Step Setup | with strictly local components

1) Clone the repository and open a terminal in the project directory.

2) Create your personalized .env from .env.example:

BASH

	cp .env.example .env

3) Fetch a Whisper.cpp model (e.g., ggml‑base.bin, default in .env) and place it in /models. Adjust the path in .env if you use another model.

4) Install Ollama, pull an LLM model, e.g., gemma3:1b (default in .env). Adjust .env to match your model. If you want Style Engine reference interpretation, also pull a vision‑capable model and set APP_OLLAMA_VISION_MODEL.

5) Install ComfyUI and pull a diffusion model.

ComfyUI suggests v1-5-pruned-emaonly-fp16.safetensors. Place it into <ComfyUI>/models/checkpoints (inside ComfyUI, not in slAIdshow). Example direct download for an alternative model:

	curl -L -O "https://huggingface.co/datasets/tyDiffusion/Diffusion/resolve/7f894348dd1cb8a86a81f48d426277cf6d810af1/dreamshaper-8-1.5.safetensors"

If ComfyUI runs on the same device, set APP_COMFY_OUTPUT_DIR in .env to your <ComfyUI>/output path. For LAN usage, let ComfyUI listen on 0.0.0.0 and set APP_COMFY_HOST to the LAN IP. Put your custom ComfyUI workflows into /workflows and select them in the UI.

Start ComfyUI from its main folder:

BASH

	# With ~6GB VRAM GPU
	python main.py --listen 127.0.0.1 --port 8188 --lowvram
	# With CPU
	# python main.py --listen 127.0.0.1 --port 8188 --CPU
	# Expose to LAN
	# python main.py --listen 0.0.0.0 --port 8188 --lowvram  # or --CPU

Open ComfyUI in the browser:

	# strictly local
	http://127.0.0.1:8188
	# reachable over LAN (if running on another device)
	http://<lan-ip-of-comfyui-host>:8188

6) Optional: Pollinations cloud backend — sign up, get API key, set in .env:

	POLLINATIONS_API_KEY=sk_*************

---

### slAIdshow Installation

#### Option A — Helper scripts (recommended)

Linux / macOS:

BASH

	chmod +x run.sh
	./run.sh

Windows (PowerShell):

BASH

	Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
	.\run.ps1

What the scripts do:

- Create project virtual environment
- Install requirements
- Ensure pywhispercpp and optional webrtcvad wheels
- Preflight checks for Ollama and ComfyUI
- Start FastAPI app via uvicorn (app:app)

Open the UI in your browser:

- http://127.0.0.1:8080

Stop via the UI or press Ctrl+C in the terminal.

Summary — important .env values:

	APP_IMAGE_BACKEND=comfyui            # or pollinations
	APP_OUTPUT_DIR=./outputs/images
	APP_COMFY_WORKFLOW=./workflows/text2img_SD15-FP16.json
	APP_COMFY_HOST=127.0.0.1
	APP_COMFY_PORT=8188
	APP_COMFY_OUTPUT_DIR=/path/to/ComfyUI/output
	APP_ALLOW_REMOTE_BACKENDS=0          # set 1 to allow LAN/remote Comfy
	APP_ALLOW_REMOTE_VISION=0            # set 1 to allow non-local vision
	APP_OLLAMA_HOST=127.0.0.1
	APP_OLLAMA_PORT=11434
	APP_OLLAMA_MODEL=gemma3:1b
	APP_OLLAMA_VISION_MODEL=llava:latest # example, choose your own
	POLLINATIONS_API_KEY=sk_xxxx         # required if using Pollinations

Start manually if needed:

Linux/macOS:

	source .venv/bin/activate
	python app.py

Windows:

	.venv\Scripts\Activate.ps1
	python app.py

or:

	uvicorn app:app --host 127.0.0.1 --port 8080

Open the UI:

- http://127.0.0.1:8080

---

#### Option B — Manual setup

Create your personalized .env:

BASH

	cp .env.example .env

Create and activate a virtual environment, then install dependencies:

	python3 -m venv .venv

Linux/macOS:

	source .venv/bin/activate

Windows PowerShell:

	.venv\Scripts\Activate.ps1
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	# pywhispercpp installation:
	pip install --no-cache-dir pywhispercpp

Optional VAD:

	pip install webrtcvad
	# or wheels:
	pip install --no-cache-dir webrtcvad-wheels

Fetch a Whisper.cpp model and place it in ./models (ggml‑base.bin is default in .env):

BASH

	curl -L -o models/ggml-base.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
	# or tiny:
	curl -L -o models/ggml-tiny.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin

PowerShell:

	curl.exe -L "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin?download=true" -o "models/ggml-base.bin"
	# or tiny:
	curl.exe -L "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin?download=true" -o "models/ggml-tiny.bin"

Start the server:

	python app.py
	# or:
	uvicorn app:app --host 127.0.0.1 --port 8080

Open the UI:

- http://127.0.0.1:8080

---

### Usage

- Click “Start” → local audio recording begins; SSE status messages appear. Every ~3–6 s (APP_SNAPSHOT_SEC) a transcript snapshot is produced. The LLM prompt is generated if Ollama is running. If ComfyUI is unavailable, a clear status is shown and the app continues. If APP_IMAGE_BACKEND=pollinations and a valid key is present, images are generated via Pollinations.
- Use the backend selector to switch between:
  - Comfy local (same machine or same‑LAN host)
  - Comfy remote (another network via reachable URL/VPN)
  - Pollinations cloud (always available)
- Use the Style Engine: enter a style prompt, optionally upload/link a reference image, toggle “Use Ollama Vision” for reference interpretation (policy‑gated), and apply. The computed style_positive is injected into generation.
- Adjust width/height and negative prompt; apply as needed.
- Click “Stop” in the UI or press Ctrl+C in the terminal.

---

### Web UI and API Notes

- Web UI served from /web; root (/) redirects to /web/index.html
- Images are written to outputs/images and exposed via /static
- SSE at /events streams status, transcripts, LLM prompts, and image notifications
- Selected API endpoints:
  - /status, /start, /stop, /shutdown, /config, /open_dir_hint
  - /api/plan, /api/image/direct
  - /api/settings/image_backend, /api/settings/image_size, /api/settings/negative_prompt, /api/settings/workflow
  - /api/workflows, /workflows/index.json
  - /api/style/build, /api/style/upload, /api/style/save_url, /api/style/reset

---

### Tests

Check pywhispercpp import:

	python - <<'PY'
	from pywhispercpp.model import Model as WhisperModel
	print("pywhispercpp import OK")
	PY

If not found:

	pip uninstall -y pywhispercpp
	pip install --no-cache-dir pywhispercpp

Audio device listing:

	python - <<'PY'
	import sounddevice as sd
	print(sd.query_devices())
	PY

Look for an input device where max_input_channels > 0. Use the device index/name in your .env if needed.

Microphone level test:

	python utils/audio_test.py

Whisper live mic test:

	python utils/mic_check_whisper.py

ComfyUI local reachability:

	python utils/test_comfy_local.py

---

### Ollama setup hints (LLM backend for automated image prompts)

Install Ollama (see official docs), then:

	ollama serve
	ollama pull gemma3:1b  # or phi3:mini, llama3, mistral, ...

Sanity check:

	curl -s http://127.0.0.1:11434/api/generate -H "Content-Type: application/json" -d '{"model":"gemma3:1b","prompt":"Say hello","stream":false,"options":{"temperature":0.2}}'

Ensure APP_OLLAMA_MODEL in .env matches the pulled model and that Ollama listens on 127.0.0.1:11434. If you use style reference interpretation, also pull a compatible vision model and configure APP_OLLAMA_VISION_MODEL. By default, Ollama is kept strictly local for security.

---

### ComfyUI setup (Local/LAN/Remote)

Install and start ComfyUI (API at http://127.0.0.1:8188). Place required models under ComfyUI/models as requested by your workflow (checkpoints/vae/lora/etc.).

Optional systemd service (Linux):

	sudo cp comfyui.service /etc/systemd/system/
	sudo systemctl daemon-reload
	sudo systemctl enable comfyui
	sudo systemctl start comfyui
	sudo systemctl status comfyui

CPU‑only mode inside the ComfyUI venv:

	pip install --upgrade --force-reinstall --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	pip uninstall -y xformers triton
	python - << 'PY'
	import torch
	print("torch", torch.__version__, "cuda_available?", torch.cuda.is_available())
	PY

Expose to LAN or remote access:

	python main.py --listen 0.0.0.0 --port 8188 --lowvram
	# Secure appropriately (firewall/VPN/reverse proxy). In slAIdshow set:
	# APP_ALLOW_REMOTE_BACKENDS=1 and APP_COMFY_HOST=<LAN/remote host>.

Access from other device:

	http://<ip-of-device-running-comfyui>:8188

---

### Pollinations setup (Cloud backend)

Use Pollinations as an alternative image backend (cloud). Requires an API key.

1) Create a Pollinations account and obtain your API key.
2) Add the key to `.env`:

	POLLINATIONS_API_KEY=sk_xxxx

3) Optional tuning vars (if supported by your bridge):

	POLLINATIONS_MODEL=flux
	POLLINATIONS_NOLOGO=1
	POLLINATIONS_SEED=1234
	POLLINATIONS_USE_V1=1
	POLLINATIONS_WIDTH=1024
	POLLINATIONS_HEIGHT=1024

4) With a valid key, switch backend to “pollinations” in the UI or via /api/settings/image_backend.

Notes:
- Cloud service — prompts/requests go over the internet.
- Keep your API key private. Do not commit `.env`.

---

### Troubleshooting

- No transcripts:
  - Check audio levels (utils/audio_test.py).
  - Ensure correct input device (APP_AUDIO_DEVICE).
  - Try APP_DISABLE_VAD=1; tune APP_RMS_VAD_THRESHOLD (e.g., 0.01–0.02).
- comfy_unavailable:
  - Start ComfyUI or switch backend. For LAN/remote, set APP_ALLOW_REMOTE_BACKENDS=1 and use correct host/port.
- pipeline_error (LLM):
  - Is Ollama running (ollama serve)?
  - Pulled a model (ollama pull <model>) and .env matches?
- Pollinations errors:
  - Check POLLINATIONS_API_KEY in `.env` and network access.
  - Ensure APP_IMAGE_BACKEND=pollinations.
- Microphone muted:
  - Linux: pavucontrol → Input Devices → unmute; set ~70–90%.
- Port conflict on 8080:
  - Use another port: uvicorn app:app --port 8081.
- Windows/macOS device names:
  - Use sounddevice.query_devices(); set APP_AUDIO_DEVICE accordingly.

If `.env` was committed accidentally:

	git rm --cached .env
	git add .gitignore
	git commit -m "Remove .env from repo and stop tracking"
	git push
	# optionally scrub history:
	pipx install git-filter-repo
	git filter-repo --path .env --invert-paths
	git push --force

---

### Security and Privacy

- Ollama core is restricted to 127.0.0.1 by default.
- Remote image backends (ComfyUI LAN/remote) are disabled unless APP_ALLOW_REMOTE_BACKENDS=1.
- Ollama Vision remote/cloud usage is gated by APP_ALLOW_REMOTE_VISION.
- Pollinations is cloud; avoid sensitive data in prompts.
- Audio is processed in RAM only; no raw audio is saved by default.
- Logs and images remain local unless you choose a cloud backend.

---

### Roadmap

- Solidify Style Engine across all three backends incl. Ollama Vision reference interpretation
- Harden remote ComfyUI (LAN/remote) connectivity and fallbacks
- Lower‑latency transcription (streaming/VAD tuning)
- Optional GPU utilization improvements
- Expanded Pollinations controls (models, safety)

---

### License

- MIT Licence

---

### Contact

- Betakontext | Christoph Medicus — dev@betakontext.de — https://dev.betakontext.de

- Contributions and issues are welcome. Please open an issue with logs and your environment (.env without secrets) if you need help.

### Support

Buy me a coffee on https://buymeacoffee.com/betakontext
