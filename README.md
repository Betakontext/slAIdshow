# speechtoimage_ai

## Pipeline for AI live-illustrations (lokal, Browser-UI)

A local real-time application that listens and periodically generates images. Everything runs on your machine: audio capture from the system microphone, transcription via Whisper (pywhispercpp), prompt optimization via Ollama (optional), and image generation via ComfyUI (optional). A simple browser UI provides Start/Stop and displays a growing gallery of generated images along with live transcript and the latest prompt.

Note: The app runs without ComfyUI; you will still see status and transcript events. ComfyUI can be added later.

---

### Features

- Browser UI (local, FastAPI) with Start/Stop
- Local audio capture from system device (e.g., PipeWire/PulseAudio)
- Periodic transcription snapshots (configurable, e.g., every 3–6 s)
- Optional: Prompt optimization via Ollama (localhost:11434)
- Optional: Image generation via ComfyUI (localhost:8188)
- Live updates in the browser via Server-Sent Events (SSE)
- Strictly local connections (127.0.0.1)

---

### System requirements

- OS: Linux tested (PipeWire/PulseAudio). macOS/Windows should work with adjusted device names.
- Python: 3.10 or newer
- Working microphone
- pywhispercpp installed and running locally (for audio transcription)
- Ollama installed and running locally (for LLM prompt optimization)
- ComfyUI running locally with API on port 8188 (for image generation)

---

### Repository layout

	├── app.py
├── comfyui_bridge.py
├── comfyui.service
├── image_backend.py
├── models
│   ├── ggml-base.bin
├── outputs
│   └── images
├── __pycache__
├── README.md
├── requirements.txt
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

- Python 3.9 or newer
- Local-only services:
  - Ollama running on 127.0.0.1:11434 (pull your model once)
  - ComfyUI running on 127.0.0.1:8188 (API enabled)

#### Linux (Debian/Ubuntu)

Install system packages:

BASH

	sudo apt update
	sudo apt install -y build-essential cmake pkg-config python3-dev \
	libportaudio2 libasound2-dev

Notes:

These are not in requirements.txt; they are OS libraries and headers used by sounddevice/PortAudio and some wheels/builds.


#### macOS

Install Command Line Tools: xcode-select --install
Install PortAudio via Homebrew:

BASH

	brew install portaudio

#### Windows

Recommended: run the PowerShell script below (it creates a venv and installs wheels).
If you build from source, you may need Microsoft C++ Build Tools.
If you have trouble with native deps, consider WSL where the Linux steps apply.


### Installation (Linux / macOS)

Clone the repository and open a new terminal in the project directory. Run helper script (recommended):


Install whisper, create /models folder in the repo and pull Whisper model:

BASH

	pip install --no-cache-dir pywhispercpp

BASH

	mkdir -p models
	curl -L -o models/ggml-base.bin \
	https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
	# or tiny:
	# curl -L -o models/ggml-tiny.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin

Linux / macOS:

BASH

	chmod +x run.sh
	./run.sh


Windows (PowerShell):

BASH

    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
    .\run.ps1

These scripts:

Create/activate a virtual environment
Upgrade pip and install requirements
Try to install optional extras (webrtcvad-wheels, pywhispercpp)
Start the server on 127.0.0.1:8080 (uvicorn)

-----------------------------------------

To start app.py (Server automaticly started)

	python app.py
	
Open the browser to use UI:

    http://127.0.0.1:8080


Stop via UI in the browser or in terminal via Ctrl+C.

-----------------------------------------

#### Or install dependencies manually (from requirements.txt):

BASH
	python3 -m venv .venv
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install --no-cache-dir webrtcvad-wheels
	pip install --no-cache-dir pywhispercpp

To load environment from .env:

BASH

	export $(grep -v '^#' .env | xargs -d '\n')

#### Quick tests

Whisper import:

BASH

	python - <<'PY'
	from pywhispercpp.model import Model as WhisperModel
	print("pywhispercpp import OK")
	PY

If not found, try:

BASH

	pip uninstall -y pywhispercpp
	pip install --no-cache-dir pywhispercpp

Create /models folder and pull Whisper model:

BASH

	mkdir -p models
	curl -L -o models/ggml-base.bin \
	https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
	# or tiny:
	# curl -L -o models/ggml-tiny.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin

Find audio device:

BASH

	python - <<'PY'
	import sounddevice as sd
	print(sd.query_devices())
	PY

Look for an input device (max_input_channels > 0). Use the exact name or index for configuration/tests.


Mic level test:

BASH

	python utils/audio_test.py

Expected example output when the mic is active:

Using default input @ 48000 Hz. Please speak for 3s…
Peak=0.368, RMS=0.052


####	Whisper setup (pywhispercpp)

Whisper test:

BASH

	python mic_check_whisper.py

Tips:

Language can be “de”, “en”, or “auto”
Tune threads to your CPU
Very short segments can be choppy → keep rolling window enabled (already in code)


Ollama setup (LLM, optional)
To install Ollama (see official documentation). Then:

BASH

	ollama serve
	ollama pull phi3:mini     # or another model (llama3, mistral, phi3, etc.)

Ensure the server listens on 127.0.0.1:11434 (default). Make sure APP_OLLAMA_MODEL matches your pulled model.

Sanity check:

BASH

	curl -s http://127.0.0.1:11434/api/generate \
	-H "Content-Type: application/json" \
	-d '{"model":"phi3:mini","prompt":"Say hello","stream":false,"options":{"temperature":0.2}}'
	
	
####	Pollinations setup (Cloud image generation)

Create a Pollinations account and sve your keys, f.e.:


-> Integrate your key sd ### in .env


####	ComfyUI installation and setup (optional, image generation)

To install and start ComfyUI, prepare your system:

BASH

	sudo apt update
	sudo apt install -y git python3 python3-venv python3-dev build-essential libglib2.0-0 libsm6 libxrender1 libxext6

Decide where you want to place ComfyUI on your system and open that in your terminal:

BASH

	git clone https://github.com/comfyanonymous/ComfyUI.git
	cd ComfyUI
	
Open virtual environment:

BASH

	python3 -m venv .venv
	source .venv/bin/activate
	python -m pip install --upgrade pip setuptools wheel

Install dependencies:

BASH

	pip install -r requirements.txt
	
Start locally (API at http://127.0.0.1:8188).

	BASH

	python main.py --listen 127.0.0.1 --port 8188
	
API-endpoints: /prompt, /history/{id}, /view

Modelle besorgen

ComfyUI does not automaticly load big modells. 
Depending on your workflow you may need heckpoints/VAEs/LoRAs:

main paths (under ComfyUI/models):

models/checkpoints → z. B. SDXL/SD1.5 .safetensors
models/vae
models/clip
models/unet
models/ipadapter, models/controlnet etc. (optional)

Copy the comfyui.service from the repo into your /etc/systemd/system/ or create the service file in  /etc/systemd/system/

comfyui.service:

	sudo cp -p yourpathto_speechtoimage_ai/comfyui.service /etc/systemd/system/

Addapt the paths to your ComfyUi Installation.

	# /etc/systemd/system/comfyui.service
	[Unit]
	Description=ComfyUI (local)
	After=network.target

	[Service]
	Type=simple
	User=cm
	WorkingDirectory=/home/cm/Dokumente/Arbeiten/0000_DEV/ComfyUI
	ExecStart=/home/cm/Dokumente/Arbeiten/0000_DEV/ComfyUI/.venv/bin/python main.py --listen 127.0.0.1 --port 8188
	Environment="PATH=/home/cm/Dokumente/Arbeiten/0000_DEV/ComfyUI/.venv/bin:%s"
	Restart=on-failure

	[Install]
	WantedBy=multi-user.target
	

Proof rw rights:
	
	sudo chown -R cm:cm /home/cm/Dokumente/Arbeiten/0000_DEV/ComfyUI/
	ls -ld /home/cm/Dokumente/Arbeiten/0000_DEV/ComfyUI/

CPU only mode if GPU is not available (f.e. on AMD hardware)

BASH : Check whats installed in the venv
	
	source .venv/bin/activate
	python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
	deactivate
	
Output f.e.: torch 2.12.0+cu130 cuda? False
-> No Cuda available -> CPU only mode

Switch torch to CPU in the venv:

BASH

	source .venv/bin/activate
	pip install --upgrade --force-reinstall --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	pip uninstall -y xformers triton
	python - << 'PY'
	import torch
	print("torch", torch.__version__, "cuda_available?", torch.cuda.is_available())
	PY
	deactivate
	
BASH 

	sudo systemctl daemon-reload
	sudo systemctl enable comfyui
	sudo systemctl start comfyui
	sudo systemctl status comfyui

Activate, test and start locally (API at http://127.0.0.1:8188)

START

BASH

	source .venv/bin/activate
	python main.py --listen 127.0.0.1 --port 8188 --cpu´
	
LOGS:

BASH
	
	journalctl -u comfyui -f
	
STOP

BASH

	sudo systemctl stop comfyui
	

Download a model for comfyUI checkpoint and place it under /ComfyUI/models/

BASH	

	mkdir -p ~/Downloads/tmp_models && cd ~/Downloads/tmp_models
	# Direktdownload-URL vom HF-Button "Download" kopieren:
	curl -L --fail -o anything-v4.5-pruned.safetensors \
	"https://huggingface.co/shibal1/anything-v4.5-clone/resolve/main/anything-v4.5-pruned.safetensors?download=true"

	# SHA256 berechnen und notieren
	sha256sum anything-v4.5-pruned.safetensors
	# macOS Alternative:
	# shasum -a 256 anything-v4.5-pruned.safetensors
	
TEST test_comfy_local.py with ComfyUI running on http://127.0.0.1:8188.
	
BASH
	
	APP_DISABLE_COMFYUI=0 python test_comfy_local.py --workflow ./workflows/text2img.json --prompt "Fotorealistisches Klassenzimmer, natürliches Licht" --width 512 --height 512 --steps 20 --cfg 6.5 --sampler dpmpp_2m --seed 1234 --timeout 300

Export your workflow JSON in ComfyUI.

In app.py, adapt:
build_comfy_prompt_from_text(): turn the LLM text into your node-graph (prompt/negative prompt, sampler, VAE, SaveImage, etc.).
comfyui_run_and_wait(): adjust reading the history (filename/subfolder).

Images can be written by ComfyUI to a known folder or copied into ./outputs/images. The UI serves images via /static/... (APP_OUTPUT_DIR). Without ComfyUI, the app remains useful (status/transcript/prompt events).

#### Environment variables / .env

See, use and addapt values in -> .env file

Output files are saved in:

	APP_OUTPUT_DIR=./outputs/images

Custom ComfyUI workflows (API export) can be saved and should be defined here

	APP_COMFY_WORKFLOW=./workflows/text2img_any45.json

Make sure to set the correct ComfyUi output folder here, to bridge correctly to speechtoimage_ai

	APP_COMFY_OUTPUT_DIR=/home/cm/Dokumente/Arbeiten/0000_DEV/ComfyUI/output/


#### Load environment variables and start the server:

BASH

	source .venv/bin/activate
	# Start server
	python app.py
	
oder

	uvicorn app:app --host 127.0.0.1 --port 8080
	
	
Open the browser to use UI:

    http://127.0.0.1:8080

--------------------------------------

Click “Start” → Local audio recording begins; status messages appear. Every ~3–6 s (configurable via APP_SNAPSHOT_SEC) a transcript snapshot is produced. The LLM prompt is generated if Ollama is running. If ComfyUI is not available, a clear status is shown and the app continues to function.

Click “Stop” in the UI or stop uvicorn in the terminal (Ctrl+C).

--------------------------------------


#### Troubleshooting

No transcripts:

Check audio levels (see test script).
Ensure the correct device is set (APP_AUDIO_DEVICE).
Set APP_DISABLE_VAD=1 and tune APP_RMS_VAD_THRESHOLD (e.g., 0.01–0.02).

“comfy_unavailable”:

ComfyUI is not running → OK for now. Start it later or disable ComfyUI code paths.

“pipeline_error: …” for LLM:

Is Ollama running (ollama serve)?
Is a local model available (ollama pull phi3:mini)?
Does APP_OLLAMA_MODEL match the pulled model?

Microphone muted:

pavucontrol → Input Devices → unmute and set level ~70–90%

Port conflict on 8080:

Start uvicorn on another port: --port 8081

Windows/macOS:

Device names differ. Use sounddevice.query_devices() to find names/indices. Set APP_AUDIO_DEVICE accordingly.


#### Security and privacy

All network calls are restricted to 127.0.0.1.
No external uploads or cloud services.
Audio is processed in RAM only; no raw audio is saved by default.
Logs and images remain local.


#### Roadmap

Integrate a concrete ComfyUI workflow (prompt mapping, sampler, VAE, SaveImage)
Improve prompt templates and add age-appropriate guardrails
UI enhancements: fullscreen slideshow, limit gallery to last N images
Lower-latency transcription improvements and optional VAD refinements
Optional GPU acceleration where available

License: MIT

Contact:

Christoph Medicus
dev@betakontext.de
https://dev.betakontext.de

Contributions and issues are welcome. Please open an issue with logs and your environment (.env without secrets) if you need help.
