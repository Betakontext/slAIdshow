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

	.
	├── app.py
	├── config.py
	├── mic_check_whisper.py
	├── models
	│ └── ggml-base.bin
	├── outputs
	│ └── images
	├── README.md
	├── requirements.txt
	├── run.sh
	├── static
	└── utils
	├── audio_test.py
	└── dev_check.py



---

### Installation (example: ThinkPad X260 / Ubuntu / local)

Install system packages:

BASH

	sudo apt update
	sudo apt install -y build-essential cmake pkg-config python3-dev \
	libportaudio2 libasound2-dev

Prepare project folder:

Put all files from the repo into a new directory, e.g., speechtoimage_ai/
Open a terminal in this directory

Create and activate a virtual environment:

BASH

	python3 -m venv .venv
	source .venv/bin/activate

Run helper script (optional):

BASH

	chmod +x run.sh
	./run.sh
	
Start app.py (Server automaticly started)

	python app.py
	
Open the browser to use UI:

    http://127.0.0.1:8080



-----------------------------------------

#### Or install dependencies manually (from requirements.txt):

fastapi==0.115.0
uvicorn[standard]==0.30.6
httpx==0.27.2
pydantic>=2,<3
numpy==2.0.1
sounddevice==0.4.7
python-dotenv==1.0.1
setuptools>=68
wheel>=0.41

Install:

BASH

	pip install --upgrade pip
	pip install -r requirements.txt
	pip install --no-cache-dir webrtcvad-wheels
	pip install --no-cache-dir pywhispercpp

Load environment from .env:

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

	python - <<'PY'
	import sounddevice as sd, numpy as np
	sr=48000; dur=3
	sd.default.samplerate = sr
	sd.default.channels = 1
	print(f"Using default input @ {sr} Hz. Please speak for {dur}s…")
	audio = sd.rec(int(dur*sr), samplerate=sr, channels=1, dtype='float32')
	sd.wait()
	x = audio[:,0]
	peak = float(np.max(np.abs(x))); rms = float(np.sqrt(np.mean(x**2)))
	print(f"Peak={peak:.3f}, RMS={rms:.3f}, Samples={audio.shape[0]}")
	PY

Expected example output when the mic is active:

Using default input @ 48000 Hz. Please speak for 3s…
Peak=0.368, RMS=0.052



Whisper (pywhispercpp)

Set model path and basic config:

BASH

	export APP_WHISPER_MODEL_PATH="$(pwd)/models/ggml-base.bin"
	export APP_WHISPER_LANGUAGE="de"        # or "en", or "auto"
	export APP_WHISPER_THREADS=4
	export APP_SAMPLE_RATE=48000
	# Optional: preferred input device by index or exact name
	# export APP_AUDIO_DEVICE="pulse"        # or exact name/index from sounddevice

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

ComfyUI setup (optional, image generation)

Install and start ComfyUI locally (API at http://127.0.0.1:8188).
Export your workflow JSON in ComfyUI.
In app.py, adapt:
build_comfy_prompt_from_text(): turn the LLM text into your node-graph (prompt/negative prompt, sampler, VAE, SaveImage, etc.).
comfyui_run_and_wait(): adjust reading the history (filename/subfolder).

Images can be written by ComfyUI to a known folder or copied into ./outputs/images. The UI serves images via /static/... (APP_OUTPUT_DIR). Without ComfyUI, the app remains useful (status/transcript/prompt events).
Environment variables

You can use .env or export directly:

BASH

	# Audio
	export APP_AUDIO_DEVICE=""            # index or exact source name
	export APP_SAMPLE_RATE="48000"
	export APP_FRAME_DURATION_MS="20"
	export APP_DISABLE_VAD="1"            # 1 = disable WebRTC VAD, use RMS gate
	export APP_VAD_AGGRESSIVENESS="0"     # if WebRTC VAD enabled later: 0–3
	export APP_RMS_VAD_THRESHOLD="0.012"
	export APP_MAX_SILENCE_MS="300"
	export APP_SNAPSHOT_SEC="6.0"         # new snapshot every X seconds

	# Whisper
	export APP_WHISPER_MODEL_PATH="$(pwd)/models/ggml-base.bin"
	export APP_WHISPER_LANGUAGE="de"
	export APP_WHISPER_THREADS="4"
	export APP_WHISPER_TEMPERATURE="0.0"

	# Ollama (optional)
	export APP_OLLAMA_HOST="127.0.0.1"
	export APP_OLLAMA_PORT="11434"
	export APP_OLLAMA_MODEL="glm-4.7:cloud"
	export APP_OLLAMA_TEMPERATURE="0.2"

	# ComfyUI (optional)
	export APP_COMFY_HOST="127.0.0.1"
	export APP_COMFY_PORT="8188"

	# Output directory for images
	export APP_OUTPUT_DIR="./outputs/images"

Start / Stop

Load environment variables and start the server:

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
