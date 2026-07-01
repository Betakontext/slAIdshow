### slAIdshow | local AI speech-to-image generator

## Whisper–Ollama–ComfyUI Pipeline for AI Live Illustrations (local, Browser UI)

As a multi‑modal harness slAIdshow converts live voice inputs into real‑time visual illustrations for talks, readings, speeches and presentations.
It runs strictly local on your machine via Whisper (https://github.com/openai/whisper) as backend for audio transcriptions, local LLM prompt optimizations via Ollama (https://github.com/ollama/ollama), and image generations via ComfyUI (https://github.com/comfy-org/comfyui) locally, or online if switched to Pollinations cloud services (https://github.com/pollinations/pollinations). A simple browser UI provides Start/Stop of audio input, text input field for direct prompts, optional prompt optimizations via Ollama and a workflow selector for custom ComfyUI workflows. With the default setup, delivered in this repository, it shows the generated images every 6–10 seconds, along with live transcripts and the latest prompts. The app runs flawlessly on a RTX3060 GPU with 6GB VRAM, and slows down a bit if run with CPU only.

---

### Features

- Local Browser UI (FastAPI) with Start/Stop controls
- Audio capture from system devices with periodic transcription snapshots via Whisper (configurable, e.g., every 3–6 s)
- Prompt optimizations for picture generation from those transcripts via Ollama (localhost:11434)
- Image generations via:
  - ComfyUI locally (127.0.0.1:8188 or 0.0.0.0:8188 for same‑LAN hosts)
  - ComfyUI remote (reachable from another network via VPN/tunnel)
  - Pollinations Cloud (always reachable; requires API key)
- Live updates in the browser via Server‑Sent Events (SSE)
- Strictly local connections for local backends (127.0.0.1) and LAN availability via 0.0.0.0
- Style Engine across all backends:
  - Free‑text style prompt
  - Optional reference image (upload or URL)
  - Optional Ollama Vision model for reference interpretation
- Negative prompt injection, width/height control, and ComfyUI workflow selector
- Async pipeline to overlap long image jobs, plus robust retry logic (Ollama/ComfyUI/Pollinations)
- Web UI with DE/EN language toggle, fullscreen viewer, and header status pills

---

### System Requirements

- OS: Linux tested (PipeWire/PulseAudio). macOS/Windows should work with adjusted device names.
- Python: 3.9 or newer (3.10+ recommended)
- Working microphone
- pywhispercpp installed locally (for audio transcription)
- Ollama installed and running locally (for LLM prompt optimization)
- ComfyUI running locally with API on port 8188 (for local image generation)
- Optional: Pollinations account + API key (for cloud image generation)

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
	│   ├── audio
	│   ├── config
	│   │   └── style.json
	│   ├── images
	├── README.md
	├── requirements.txt
	├── run.ps1
	├── run.sh
	├── slAIdshow_summary.txt
	├── static
	│   └── uploads
	├── style_engine.py
	├── style_refs
	├── utils
	│   ├── audio_test.py
	│   ├── dev_check.py
	│   ├── mic_check_whisper.py
	│   ├── test_comfy_local.py
	│   └── verify_runtime.py
	├── web
	│   └── index.html
	└── workflows
		└── text2img_SD15-FP16.json

---

### Prerequisites

Linux (Debian/Ubuntu) — install OS packages for native dependencies and audio I/O:

	sudo apt update
	sudo apt install -y build-essential cmake pkg-config python3-dev libportaudio2 libasound2-dev
	# Note: These are OS libraries and headers used by sounddevice/PortAudio and potential builds. They are not part of requirements.txt.

macOS:

	xcode-select --install
	brew install portaudio

Windows:

- Recommended: Use the PowerShell script below (creates a venv and installs wheels).
- If building from source, you may need Microsoft C++ Build Tools.
- If native deps are problematic, consider WSL (use the Linux steps).

To use the full combination of services strictly local, install:

---

#### Step by step setup:

To use the full combination of services:

1. Clone the repository and open a terminal in the project directory.

2. Create your personalized .env from .env.example:

BASH

	cp .env.example .env

3. Fetch a Whisper‑model, e.g., ggml‑base.bin (default definition in .env) and place it in /models folder. Redefine it in .env if you use another model.

4. Install Ollama, pull an LLM model, e.g., gemma3:1b (default in .env). Redefine it in .env if you use another model. Also pull a vision‑capable model if you want reference interpretation (for the Style Engine).

5. Install ComfyUI and pull a diffusion model.

ComfyUI by default suggests: v1-5-pruned-emaonly-fp16.safetensors. Place it into -> /ComfyUI/models/checkpoints (in ComfyUI main folder, not in slAIdshow). If you want to download it directly from Hugging Face you can use a direct link to the wished model. In this case: dreamshaper-8-1.5.safetensors:

	curl -L -O "https://huggingface.co/datasets/tyDiffusion/Diffusion/resolve/7f894348dd1cb8a86a81f48d426277cf6d810af1/dreamshaper-8-1.5.safetensors"

Define yourpathto/ComfyUI/output in slAIdshow -> .env if you use ComfyUI on the same device. If you use over LAN switch to 0.0.0.0 in .env -> APP_COMFY_OUTPUT_DIR=/yourpath/to/ComfyUI/output. Put your customized ComfyUI workflows into /workflows. You can switch between workflows in the UI. Start ComfyUI from its main folder with:

BASH

	# With e.g. 6GB VRAM GPU
	python main.py --listen 127.0.0.1 --port 8188 --lowvram
	# With CPU
	# python main.py --listen 127.0.0.1 --port 8188 --CPU
	# To open it to your local network via LAN
	# python main.py --listen 0.0.0.0 --port 8188 --lowvram # --CPU

To open ComfyUI in the browser:

	# strictly local
	http://127.0.0.1:8188
	# to reach over LAN, if running on another device
	http://<ip-address-of-device-with-ComfyUI-running>:8188

6. If you want to use the integrated option to switch to cloud image generations, sign up to Pollinations and get your Pollinations key. Set it up in -> .env -> POLLINATIONS_API_KEY=sk_*************

---

### slAIdshow Installation

#### Option A — via Helper Scripts (recommended)

Open a terminal in the main folder of the repo:

Linux / macOS:

BASH

	chmod +x run.sh
	./run.sh

Windows (PowerShell):

BASH

	Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
	.\run.ps1

What the scripts create:

- Project virtual environment
- Requirements installation
- Optional webrtcvad‑wheels, required pywhispercpp
- Preflight checks for Ollama and ComfyUI
- Start FastAPI app via uvicorn (app:app)

Open the UI in your browser:

- http://127.0.0.1:8080

Stop via the UI or press Ctrl+C in the terminal.

Summary:

It is important to fill .env with your local values:

	APP_OUTPUT_DIR=./outputs/images
	APP_COMFY_WORKFLOW=./workflows/text2img_SD15-FP16.json
	APP_COMFY_OUTPUT_DIR=/yourpath/to/ComfyUI/output
	POLLINATIONS_API_KEY=sk_xxxx  #(required if using Pollinations)

Load env and start:

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

#### Option B — Manual Setup of slAIdshow

Create your personalized .env from .env.example:

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
	# Pywhisper installation:
	pip install --no-cache-dir pywhispercpp

Optional VAD (may require source build on some systems):

	pip install webrtcvad

Alternatively, try wheels:

	pip install --no-cache-dir webrtcvad-wheels

Fetch a Whisper model and place it in ./models. ggml‑base.bin is default in .env

BASH

	curl -L -o models/ggml-base.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
	# or tiny:
	curl -L -o models/ggml-tiny.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin

or in PS

	curl.exe -L "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin?download=true" -o "models/ggml-base.bin"
	# or tiny:
	curl.exe -L "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin?download=true" -o "models/ggml-tiny.bin"

Start the server:

App starts uvicorn from app.py:

	python app.py

Or via uvicorn directly:

	uvicorn app:app --host 127.0.0.1 --port 8080

Open the UI:

- http://127.0.0.1:8080

---

### Usage

- Click “Start” → local audio recording begins; status messages appear. Every ~3–6 s (configurable via APP_SNAPSHOT_SEC) a transcript snapshot is produced. The LLM prompt is generated if Ollama is running. If ComfyUI is not available, a clear status is shown and the app continues to function. If APP_IMAGE_BACKEND=pollinations and a valid key is present, images are generated via the Pollinations API.
- Use the backend selector in the UI to switch between:
  - Comfy local (same machine or same‑LAN host)
  - Comfy remote (another network via reachable URL/VPN)
  - Pollinations cloud (always available)
- Use the Style Engine: enter a style prompt, optionally upload/link a reference image, toggle “Use Ollama Vision” for reference interpretation, and apply. The computed style_positive is injected into generation.
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

Check Whisper import:

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

---

### Ollama Setup hints (LLM backend for automated image prompts)

Install Ollama (see official docs), then:

	ollama serve
	ollama pull gemma3:1b # or phi3:mini, llama3, mistral, ...

Sanity check:

	curl -s http://127.0.0.1:11434/api/generate -H "Content-Type: application/json" -d '{"model":"gemma3:1b","prompt":"Say hello","stream":false,"options":{"temperature":0.2}}'

Ensure APP_OLLAMA_MODEL in the file .env matches the model you pulled and that Ollama listens on 127.0.0.1:11434. If you use style reference interpretation, also pull a compatible vision model and configure it accordingly.

---

### ComfyUI Setup (Optional Local/Remote Image Generation)

Create a ComfyUI folder, where you want it.
Install and start ComfyUI locally (API at http://127.0.0.1:8188):

	sudo apt update
	sudo apt install -y git python3 python3-venv python3-dev build-essential libglib2.0-0 libsm6 libxrender1 libxext6
	git clone https://github.com/comfyanonymous/ComfyUI.git
	cd ComfyUI
	python3 -m venv .venv
	source .venv/bin/activate
	python -m pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
	python main.py --listen 127.0.0.1 --port 8188

Models for ComfyUI (checkpoints/VAEs/LoRAs) must be placed under ComfyUI/models as required by your workflow (e.g., models/checkpoints, models/vae, models/clip, models/unet, etc.).

Optional systemd service (Linux):

Copy comfyui.service to /etc/systemd/system/, adapt paths, then:

	sudo systemctl daemon-reload
	sudo systemctl enable comfyui
sudo systemctl start comfyui
	sudo systemctl status comfyui

CPU‑only mode (if no CUDA/GPU available) inside the ComfyUI venv:

	pip install --upgrade --force-reinstall --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	pip uninstall -y xformers triton
	python - << 'PY'
	import torch
	print("torch", torch.__version__, "cuda_available?", torch.cuda.is_available())
	PY

Pull a diffusion model via Hugging Face and place it into: /ComfyUI/models/checkpoints. ComfyUI by default suggests: v1-5-pruned-emaonly-fp16.safetensors which can be downloaded from inside the Comfy GUI. Make sure to place it in <yourpathto>/ComfyUI/models/checkpoints

Start ComfyUI

	# Start from inside your ComfyUI main folder
	python main.py --listen 127.0.0.1 --port 8188 --lowvram
	# If you only have a CPU available:
	# python main.py --listen 127.0.0.1 --port 8188 --CPU

To access ComfyUI from another device in the same local network add "--listen 0.0.0.0" in ComfyUI’s config/batch (or run script) and start with:

	python main.py --listen 0.0.0.0 --port 8188 --lowvram
	# If you only have a CPU available:
	# python main.py --listen 0.0.0.0 --port 8188 --CPU

→ Access the ComfyUI GUI from the other device in the browser via:

	http://<ip-of-device-ComfyUI-is-running-on>:8188

---

### Pollinations Setup (Optional Cloud Image Generation)

Use Pollinations as an alternative image backend (cloud). This requires an API key and will send prompts to Pollinations’ API.

1) Create a Pollinations account and obtain your API key.
2) Add the key to your `.env` (example variable names):

   - POLLINATIONS_API_KEY=sk_xxxx

3) Optional tuning vars (if supported by your bridge/implementation):

   - POLLINATIONS_MODEL=flux
   - POLLINATIONS_NOLOGO=1
   - POLLINATIONS_SEED=1234
   - POLLINATIONS_USE_V1=1
   - POLLINATIONS_WIDTH=1024
   - POLLINATIONS_HEIGHT=1024

4) If the key is set and valid, the app will allow you to switch to Pollinations cloud service as backend for image generation instead of ComfyUI.

Notes:
- Pollinations is a cloud service; prompts and generation requests are sent over the internet.
- Keep your API key private. Do not commit `.env` to version control.

---

### Troubleshooting

- No transcripts:
  Check audio levels (see utils/audio_test.py).
  Ensure the correct input device (APP_AUDIO_DEVICE).
  Try APP_DISABLE_VAD=1 and tune APP_RMS_VAD_THRESHOLD (e.g., 0.01–0.02).
- “comfy_unavailable”:
  ComfyUI is not running → OK. Start it later or disable ComfyUI features.
- “pipeline_error: …” for LLM:
  Is Ollama running (ollama serve)?
  Did you pull a local model (e.g., ollama pull phi3:mini)?
  Does APP_OLLAMA_MODEL match the pulled model?
- Pollinations errors:
  Check POLLINATIONS_API_KEY in `.env`.
  Ensure APP_IMAGE_BACKEND is set to pollinations.
  Network access is required (cloud service).
- Microphone muted:
  On Linux: pavucontrol → Input Devices → unmute and set level ~70–90%.
- Port conflict on 8080:
  Start uvicorn on another port: --port 8081.
- Windows/macOS:
  Device names differ. Use sounddevice.query_devices() to find names/indices. Set APP_AUDIO_DEVICE accordingly.

If .env is pushed accidentally to repository, change your API key and reset it in your local .env

Remove .env from git tracking:

	# Remove from index (tracking), keep file locally
	git rm --cached .env
	# Make sure to have .env in .gitignore
	git add .gitignore
	# Commit and push
	git commit -m "Remove .env from repo and stop tracking"
	git push

Install git-filter-repo (to clean history):

	pipx install git-filter-repo

Delete .env from the repo history:

	git filter-repo --path .env --invert-paths

Force‑Push (overwrites online history):

	git push --force

---

### Security and Privacy

- All local backends are restricted to 127.0.0.1 (or 0.0.0.0 if explicitly enabled for LAN).
- Using Pollinations sends prompts to a cloud API; do not include sensitive data in prompts when using cloud generation.
- Audio is processed in RAM only; no raw audio is saved by default.
- Logs and images remain local unless you enable a cloud backend.

---

### Roadmap

- Solidify Style Engine across all three backends, incl. Ollama Vision reference interpretation
- Harden remote ComfyUI (cross‑network “Comfy remote” scenarios)
- Lower‑latency transcription improvements and optional VAD refinements
- Optional GPU acceleration where available
- Expand Pollinations controls (model selection, safety filters)

---

### License

- MIT Licence

---

### Contact

- Betakontext | Christoph Medicus — dev@betakontext.de — https://dev.betakontext.de

- Contributions and issues are welcome. Please open an issue with logs and your environment (.env without secrets) if you need help.

### Support

Buy me a coffee on https://buymeacoffee.com/betakontext
