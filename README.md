# speechtoimage_ai

## Pipeline for AI Live Illustrations (local, Browser UI)

A local real-time application that listens via your system microphone and periodically generates images. Everything runs on your machine: audio capture, transcription via Whisper (pywhispercpp), optional prompt optimization via Ollama, and optional image generation via ComfyUI or Pollinations. A simple browser UI provides Start/Stop and shows a growing gallery of generated images along with live transcript and the latest prompt. The app also runs without ComfyUI; you will still see status and transcript events.

---

### Features

- Local Browser UI (FastAPI) with Start/Stop controls
- Local audio capture from system devices
- Periodic transcription snapshots (configurable, e.g., every 3–6 s)
- Optional: Prompt optimization via Ollama (localhost:11434)
- Optional: Image generation via:
  - ComfyUI (localhost:8188)
  - Pollinations (cloud; requires API key)
- Live updates in the browser via Server-Sent Events (SSE)
- Strictly local connections for local backends (127.0.0.1)

---

### System Requirements

- OS: Linux tested (PipeWire/PulseAudio). macOS/Windows should work with adjusted device names.
- Python: 3.9 or newer (3.10+ recommended)
- Working microphone
- pywhispercpp installed locally (for audio transcription)
- Ollama installed and running locally (for LLM prompt optimization)
- Optional: ComfyUI running locally with API on port 8188 (for image generation)
- Optional: Pollinations account + API key (for cloud image generation)

---

### Repository Layout

	├── app.py
	├── comfyui_bridge.py
	├── comfyui.service
	├── image_backend.py
	├── outputs
	│   └── images
	├── README.md
	├── requirements.txt
	├── run.ps1
	├── run.sh
	├── static
	├── utils
	│   ├── audio_test.py
	│   ├── dev_check.py
	│   ├── mic_check_whisper.py
	│   ├── test_comfy_local.py
	│   └── verify_runtime.py
	├── web
	│   └── index.html
	└── workflows
		└── text2img_any45.json


---

### Prerequisites

Linux (Debian/Ubuntu) — install OS packages for native dependencies and audio I/O:

	sudo apt update
	sudo apt install -y build-essential cmake pkg-config python3-dev libportaudio2 libasound2-dev
	# Note: These are OS libraries and headers used by sounddevice/PortAudio and potential builds. They are not part of 	requirements.txt.

macOS:

	xcode-select --install
	brew install portaudio

Windows:

- Recommended: Use the PowerShell script below (creates a venv and installs wheels).
- If building from source, you may need Microsoft C++ Build Tools.
- If native deps are problematic, consider WSL (use the Linux steps).

To use the full combination of Services strictly local, install:

- pywhispercpp (Clone the repository and create /model Folder in speechtoimage_ai and pull a model -> f.e. ggml-base)
- Ollama on 127.0.0.1:11434 (pull at least one model -> f.e. phi3:mini)
- ComfyUI on 127.0.0.1:8188 (Model pulled -> f.e. anything-v4.5-pruned.safetensors; API enabled)

To switch to Cloud servis (Pollinations)

- create Pollinations account and set your key in .env
-------

### Option A — Installation of speechtoimage_ai with Helper Scripts (recommended)

Linux / macOS:

	chmod +x run.sh
	./run.sh

Windows (PowerShell):

	Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
	.\run.ps1

What the scripts create:

- Project virtual environment
- Requirements installation
- Optional webrtcvad-wheels, required pywhispercpp
- Preflight checks for Ollama and ComfyUI
- Start FastAPI app via uvicorn (app:app)

Open the UI in your browser:

- http://127.0.0.1:8080

Stop via the UI or press Ctrl+C in the terminal.

------------ 

#### Step by step setup: 

1. Clone the repository and open a terminal in the project directory.

2. Create your personalized .env from .env.example:

	cp .env.example .env

3. Create the folder /models and fetch a Whisper-model, f.e. ggml-base.bin in there

4. Install Ollama, pull an LLM model f.e. phi3:mini and 
   define its usage in -> .env

5. Define the -> path to /ComfyUI/output in -> .env
   APP_COMFY_OUTPUT_DIR=/yourpath/to/ComfyUI/output
   APP_COMFY_WORKFLOW=./workflows/text2img_any45.json 	
   Start ComfyUi from ist main folder with: 

	python main.py --listen 127.0.0.1 --port 8188 --lowvra

6. Pull a diffusion model f.e.: anything-v4.5-pruned and 
   place it into -> /ComfyUI/models/checkpoints (in ComfyUI main folder, not in speechtoimage_ai)

7. To switch between local and cloud image generation, use Pollinations and 
   define your Pollinations key: sk-******* in -> .env
   POLLINATIONS_API_KEY=sk-xxxx  #(required if using Pollinations)



----------------------------------------------------------------------------------------------

### Option B — Manual Setup of speechtoimage_ai

Create and activate a virtual environment, then install dependencies:

	python3 -m venv .venv

Linux/macOS:

	source .venv/bin/activate

Windows PowerShell:

	.venv\Scripts\Activate.ps1
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install --no-cache-dir pywhispercpp

Optional VAD (may require source build on some systems):

	pip install webrtcvad

Alternatively, try wheels:

	pip install --no-cache-dir webrtcvad-wheels

Fetch a Whisper model and place it in ./models:

BASH
	mkdir -p models
	curl -L -o models/ggml-base.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
	# or tiny: curl -L -o models/ggml-tiny.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin

or in PS
	mkdir -p models
	curl.exe -L "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin?download=true" -o "models/ggml-base.bin"
	# or tiny: curl.exe -L "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin?download=true" -o "models/ggml-	tiny.bin 

Start the server:

App starts uvicorn from app.py:

	python app.py

Or via uvicorn directly:

	uvicorn app:app --host 127.0.0.1 --port 8080

Open the UI:

- http://127.0.0.1:8080

---

### Quick Tests

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

### Ollama Setup (Optional LLM)

Install Ollama (see official docs), then:

	ollama serve
	ollama pull phi3:mini    (or llama3, mistral, etc.)

Sanity check:

	curl -s http://127.0.0.1:11434/api/generate -H "Content-Type: application/json" -d '{"model":"phi3:mini","prompt":"Say hello","stream":false,"options":{"temperature":0.2}}'

Ensure APP_OLLAMA_MODEL IN the file .env matches the model you pulled and that Ollama listens on 127.0.0.1:11434.

---

### ComfyUI Setup (Optional Local Image Generation)

Create a ComfyUI folder, where you want it.
Install and start ComfyUI locally (API at http://127.0.0.1:8188):

	sudo apt update
	sudo apt install -y git python3 python3-venv python3-dev build-essential libglib2.0-0 libsm6 libxrender1 libxext6	git clone https://github.com/comfyanonymous/ComfyUI.git
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

CPU-only mode (if no CUDA/GPU available) inside the ComfyUI venv:

	pip install --upgrade --force-reinstall --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	pip uninstall -y xformers triton
	python - << 'PY'
	import torch
	print("torch", torch.__version__, "cuda_available?", torch.cuda.is_available())
	PY

Pull a Diffusion model : Open a terminal in the folder to pull into: /ComfyUI/models/checkpoints and install.
For manga or anime stylized pictures f.e.

	curl.exe -fL --retry 5 --retry-delay 3 "https://huggingface.co/shibal1/anything-v4.5-clone/resolve/main/anything-v4.5-	pruned.safetensors?download=true" -o "C:\Users\Administrator\Documents\Arbeiten\0000_DEV\ComfyUI\models\checkpoints\anything-v4.5-	pruned.safetensors"

or for realistc pictures:

	curl.exe -fL --retry 5 --retry-delay 3 "https://huggingface.co/casque/realisticVisionV51_v51VAE/resolve/main/realisticVisionV51_v51VAE.safetensors?download=true" -o "C:\Users\Administrator\Documents\Arbeiten\0000_DEV\ComfyUI\models\checkpoints\realisticVisionV51_v51VAE.safetensors"


Get sha after it completes:
PS

	$dst = "C:\Users\Administrator\Documents\Arbeiten\0000_DEV\ComfyUI\models\checkpoints\anything-v4.5-pruned.safetensors"
	$hash = (Get-FileHash $dst -Algorithm SHA256).Hash
	Set-Content "$dst.sha256.txt" $hash

and verify 

(anything-v4.5)

	$dst = "C:\Users\Administrator\Documents\Arbeiten\0000_DEV\ComfyUI\models\checkpoints\anything-v4.5-pruned.safetensors"
	$expected = Get-Content "$dst.sha256.txt"
	$current  = (Get-FileHash $dst -Algorithm SHA256).Hash
	if ($current -eq $expected) { "OK: hash matches" } else { "MISMATCH: file changed!" }

(realisticVision)

	Get-Item "C:\Users\Administrator\Documents\Arbeiten\0000_DEV\ComfyUI\models\checkpoints\realisticVisionV51_v51VAE.safetensors" | Select-Object 	FullName, Length, LastWriteTime
	Get-FileHash "C:\Users\Administrator\Documents\Arbeiten\0000_DEV\ComfyUI\models\checkpoints\realisticVisionV51_v51VAE.safetensors" -Algorithm SHA256


Start ComfyUI

	# Start From your ComfyUI folder
	python main.py --listen 127.0.0.1 --port 8188 --lowvram

To access ComfyUI from another device in the same local network
add "--listen 0.0.0.0" in ComfyUIs file run_nvidia_gpu.bat 
.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --listen 0.0.0.0
and start with this command:

	python main.py --listen 0.0.0.0 --port 8188 --lowvram

-> Access from the other device in the browser via: 

	<ip-of-device-ComfyUI-is-running-on>:8188


---

### Pollinations Setup (Optional Cloud Image Generation)

Use Pollinations as an alternative image backend (cloud). This requires an API key and will send prompts to Pollinations’ API.

1) Create a Pollinations account and obtain your API key.
2) Add the key to your `.env` (example variable names):

   - POLLINATIONS_API_KEY=sk-xxxx
   - APP_IMAGE_BACKEND=pollinations

3) Optional tuning vars (if supported by your bridge/implementation):

   - POLLINATIONS_MODEL=flux
   - POLLINATIONS_NOLOGO=1
   - POLLINATIONS_SEED=1234
   - POLLINATIONS_USE_V1=1
   - POLLINATIONS_WIDTH=1024
   - POLLINATIONS_HEIGHT=1024

4) Start the app; when APP_IMAGE_BACKEND=pollinations is set and the key is present, the app will call Pollinations for image generation instead of ComfyUI.

Notes:
- Pollinations is a cloud service; prompts and generation requests are sent over the internet.
- Keep your API key private. Do not commit `.env` to version control.

---

### Environment Variables to be set to your personal setting in (.env)

Create your personalized .env from .env.example:

	cp .env.example .env

Fill .env with your lokal values, never commit:

	APP_OUTPUT_DIR=./outputs/images
	APP_COMFY_WORKFLOW=./workflows/text2img_any45.json
	APP_COMFY_OUTPUT_DIR=/path/to/ComfyUI/output
	POLLINATIONS_API_KEY=sk-xxxx  #(required if using Pollinations)

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

### Usage

- Click “Start” → local audio recording begins; status messages appear. Every ~3–6 s (configurable via APP_SNAPSHOT_SEC) a transcript snapshot is produced. The LLM prompt is generated if Ollama is running. If ComfyUI is not available, a clear status is shown and the app continues to function. If APP_IMAGE_BACKEND=pollinations and a valid key is present, images are generated via the Pollinations API.

- Click “Stop” in the UI or press Ctrl+C in the terminal.

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


If .env is pushed accidentaly to repository, change your API key and reset it in your local .env

Remove .env from git tracking:

	# Remove from index (tracking), keep file locally
	git rm --cached .env
	# MAke sure to have .env in .gitignore
	git add .gitignore
	# Commit and push
	git commit -m "Remove .env from repo and stop tracking"
	git push


Install git-filter-repo (to clean history):

	pipx install git-filter-repo

Delete .env from the repo history:

	git filter-repo --path .env --invert-paths

Force-Push (overwrites online history):

	git push --force

---

### Security and Privacy

- All local backends are restricted to 127.0.0.1.
- Using Pollinations sends prompts to a cloud API; do not include sensitive data in prompts when using cloud generation.
- Audio is processed in RAM only; no raw audio is saved by default.
- Logs and images remain local unless you enable a cloud backend.

---

### Roadmap

- Improve prompt templates and add age-appropriate guardrails
- Lower-latency transcription improvements and optional VAD refinements
- Optional GPU acceleration where available
- Expand Pollinations controls (model selection, safety filters)

---

### License

- MIT

---

### Contact

- Christoph Medicus — dev@betakontext.de — https://dev.betakontext.de

- Contributions and issues are welcome. Please open an issue with logs and your environment (.env without secrets) if you need help.
