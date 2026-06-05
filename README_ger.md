# speechtoimage_ai

##	Local pipeline for AI live-illustrations

### Vorlesen → Bilder (lokal, Browser-UI)

Lokale Echtzeit-Anwendung zum Vorlesen mit periodischer Bildgenerierung. Die App läuft vollständig auf dem lokalen Rechner: Audioaufnahme über das Systemmikrofon, Transkript-Generierung via Whisper (Platzhalter, später pywhispercpp), Prompt-Optimierung per Ollama (optional), Bildgenerierung via ComfyUI (optional). Eine einfache Browser-UI zeigt einen Start/Stop-Button sowie eine wachsende Galerie der erstellten Bilder.

Hinweis: In der aktuellen Version sind Whisper und ComfyUI noch als Platzhalter eingebunden. Die App ist bereits nutzbar (Status/Transkript-Events) und kann schrittweise erweitert werden.

---

#### Features

- Browser-UI (lokal, FastAPI) mit Start/Stop-Button
- Lokale Audioaufnahme vom Systemgerät (z. B. PipeWire)
- Periodischer Snapshot der Transkription (alle 5–10 s konfigurierbar)
- Optional: Prompt-Optimierung via Ollama (localhost:11434)
- Optional: Bildgenerierung via ComfyUI (localhost:8188)
- Live-Updates im Browser per Server-Sent Events (SSE)
- Strikt lokale Verbindungen (127.0.0.1)

---

#### Systemvoraussetzungen

- Betriebssystem: Linux (getestet mit PipeWire/PulseAudio). macOS/Windows prinzipiell möglich, aber Audio-Device-Namen abweichend.
- Python: 3.10 oder neuer
- Mikrofon funktionsfähig (über PipeWire/PulseAudio ansprechbar)
- Optional:
  - Ollama lokal installiert und gestartet (für LLM Prompt-Optimierung)
  - ComfyUI lokal mit API auf Port 8188 (für Bildgenerierung)

---

speechtoimage_ai$ tree

	.
	├── app.py
	├── config.py
	├── mic_check_whisper.py
	├── models
	│   └── ggml-base.bin
	├── outputs
	│   └── images
	├── __pycache__
	│   └── app.cpython-312.pyc
	├── README.md
	├── requirements.txt
	├── run.sh
	├── static
	└── utils
		├── audio_test.py
		└── dev_check.py


### Installation
Z.B.: Auf Laptop Thinkpad X260 / Ubuntu / local

BASH

	sudo apt update
	sudo apt install -y build-essential cmake pkg-config python3-dev \
	libportaudio2 libasound2-dev

#### 1) Projektordner vorbereiten

- Lege alle Dateien asu dem Repo in ein neues Verzeichnis, z. B. `speechtoimage_ai/`
- Öffne ein Terminal in diesem Verzeichnis

#### 2) Virtuelle Umgebung erstellen und aktivieren

BASH

	python3 -m venv .venv
	source .venv/bin/activate

#### 3) Installation Script starten (optimiert für Thinkpad X260)

BASH

	chmod +x run.sh
	./run.sh


#### Oder Abhängigkeiten installieren / anpassen

Diese findest du in requirements.txt:

	fastapi==0.115.0
	uvicorn[standard]==0.30.6
	httpx==0.27.2
	pydantic>=2,<3
	numpy==2.0.1
	sounddevice==0.4.7
	python-dotenv==1.0.1
	setuptools>=68
	wheel>=0.41

Installation:

BASH

	pip install --upgrade pip
	pip install -r requirements.txt
	pip install --no-cache-dir webrtcvad-wheels
	pip install --no-cache-dir pywhispercpp

Umgebung laden:

BASH

	export $(grep -v '^#' .env | xargs -d '\n')

#### Testen:

Whisper import:

BASH

	python - <<'PY'
	from pywhispercpp.model import Model as WhisperModel
	print("pywhispercpp import OK")
	PY

Falls nichts gefunden wird nochmal:

	pip uninstall -y pywhispercpp
	pip install --no-cache-dir pywhispercpp


Audiogerät finden:

BASH

	python - <<'PY'
	import sounddevice as sd
	print(sd.query_devices())
	PY

Suche in der Ausgabe nach dem Eingabe-Device (input channels > 0).

Z.B: 15 pipewire, ALSA (64 in, 64 out)

Notiere dir entweder den Namen exakt oder den Index (Zahl) um sie in diesem Testskript ggf. anzupassen:

Testskript Lautstärke:

	python - <<'PY'
	import sounddevice as sd, numpy as np
	sr=48000; dur=3
	dev_index = 15  # pipewire
	sd.default.samplerate = sr
	sd.default.channels = 1
	sd.default.device = (None, dev_index)  # (output_device, input_device)
	print(f"Nutze Input-Device Index={dev_index} @ {sr} Hz. Bitte {dur}s sprechen…")
	audio = sd.rec(int(dur*sr), samplerate=sr, channels=1, dtype='float32')
	sd.wait()
	peak = float(np.max(np.abs(audio))); rms = float(np.sqrt(np.mean(audio**2)))
	print(f"Peak={peak:.3f}, RMS={rms:.3f}, Samples={audio.shape[0]}")
	PY

-> Wenn das Gerät angeschaltet ist und reagiert ist folgender output zu erwarten:

	Nutze Input-Device Index=15. Bitte 3s sprechen…
	Peak=0.368, RMS=0.052

#### Whisper einbinden (pywhispercpp)

BASH

	pip install pywhispercpp # Oben schon geschehen

Lade ein ggml-Modell (z. B. ggml-tiny.bin) in den lokalen models/-Ordner: zB auf einem CPU based Laptop:

	curl -L -o models/ggml-base.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
	# oder tiny:
	# curl -L -o models/ggml-tiny.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin

Modellpfad setzen:

	# base (empfohlen)
	export APP_WHISPER_MODEL_PATH="$(pwd)/models/ggml-base.bin"
	# oder tiny:
	# export APP_WHISPER_MODEL_PATH="$(pwd)/models/ggml-tiny.bin"

Whisper Test:

	export APP_WHISPER_MODEL_PATH="/home/mc/Documents/Arbeiten/0000_DEV/Python/speechtoimage_ai/models/ggml-base.bin"
	export APP_WHISPER_LANGUAGE="de"
	export APP_WHISPER_THREADS=4
	export APP_SAMPLE_RATE=48000
	# Optional: bevorzugtes Eingabegerät
	# export APP_AUDIO_DEVICE="pulse"   # oder exakter Name des Mikrofons

	python3 mic_check_whisper.py

Tipps:

Sprache („de“ oder „auto“)
Threads an CPU anpassen
Kurze Segmente können holprig sein → Rolling Window (bereits im Code) beibehalten.

### Ollama einrichten (LLM)

Installation: siehe offizielle Ollama-Doku

Starten:

BASH

	ollama serve
	ollama pull phi3:mini     # oder ein anderes Modell (llama, mistral, phi3, usw.)

Stelle sicher, dass der Server auf 127.0.0.1:11434 lauscht (Default).

Optional: ComfyUI einrichten (Bildgenerierung)

Installiere und starte ComfyUI lokal (API erreichbar auf http://127.0.0.1:8188).

Exportiere deinen Workflow-JSON in ComfyUI.
Ersetze im Code:

	build_comfy_prompt_from_text(): Baue aus dem LLM-Text deinen Node-Graph (Prompt/Negative Prompt, Sampler, VAE, SaveImage, etc.).
	comfyui_run_and_wait(): Passe das Auslesen der History an (filename/subfolder).

Bilder aus ComfyUI können in einen bekannten Ordner geschrieben oder nach ./outputs/images kopiert werden. Die UI lädt Bilder per /static/... (APP_OUTPUT_DIR).

Solange ComfyUI nicht läuft, bleibt die App funktionsfähig (Status-/Transkript-Events).



Die App nutzt Umgebungsvariablen aus der Datei .env mit Defaults.
Individuelle Änderungen bei Bedarf sind auch im Terminal möglich:

BASH

#### Audio

	export APP_AUDIO_DEVICE="15"     # oder exakter Quellname (pactl list sources short)
	export APP_SAMPLE_RATE="48000"
	export APP_FRAME_DURATION_MS="20"
	export APP_DISABLE_VAD="1"
	export APP_VAD_AGGRESSIVENESS="0"      # 0=am tolerantesten, 1=konservativer
	export APP_MAX_SILENCE_MS="300"
	export APP_SNAPSHOT_SEC="3.0"          # alle X Sekunden ein neuer Snapshot/Prompt

#### Ollama (optional)

	export APP_OLLAMA_HOST="127.0.0.1"
	export APP_OLLAMA_PORT="11434"
	export APP_OLLAMA_MODEL="phi3:mini"
	export APP_OLLAMA_TEMPERATURE="0.2"

#### ComfyUI (optional)

	export APP_COMFY_HOST="127.0.0.1"
	export APP_COMFY_PORT="8188"

#### Ausgabeordner für Bilder (falls du Bilder dorthin kopierst/ablegst)

	export APP_OUTPUT_DIR="./outputs/images"


## Hinweise:

Für Schulen/Datenschutz: Alle Ziele sind hart auf 127.0.0.1 beschränkt. Wenn du ein anderes Audio-Device nutzen willst, setze APP_AUDIO_DEVICE auf den genauen Namen (pactl list sources short hilft).

#### Start

Umgebungsvariablen laden und Server starten

BASH

	# Umgebunsvariablen laden
	export $(grep -v '^#' .env | xargs -d '\n')
	# Optional: prüfen, was wirklich gesetzt ist
	env | egrep 'APP_(AUDIO_DEVICE|SAMPLE_RATE|DISABLE_VAD|SNAPSHOT_SEC|WHISPER_)'
	# Server starten
	uvicorn app:app --host 127.0.0.1 --port 8080 --reload

Browser öffnen

    http://127.0.0.1:8080

Klicke „Start“ → Die Audioaufnahme beginnt lokal; Statusmeldungen erscheinen.

Alle ~6 s wird ein Transkript-Snapshot erzeugt.

Der LLM-Prompt wird optional über Ollama erzeugt (falls ollama serve läuft).

Falls ComfyUI nicht läuft, wird ein verständlicher Status gesendet („comfy_nicht_verfügbar“).

#### Stop

Klicke „Stop“ in der UI oder beende uvicorn im Terminal (Ctrl+C).


----------------------------------------------------------


#### Troubleshooting

Keine Transkripte:
	Prüfe Audio-Pegel (siehe Testskript).
	Setze APP_VAD_AGGRESSIVENESS=0 und APP_MAX_SILENCE_MS=250–350.
	Sicherstellen, dass das richtige Device gesetzt ist (APP_AUDIO_DEVICE).

„comfy_nicht_verfügbar“:
	ComfyUI läuft noch nicht → OK für jetzt. Später starten oder ComfyUI-Aufrufe im Code deaktivieren.

„fehler_pipeline: …“ bei LLM:
	Läuft Ollama (ollama serve)?
	Ist lokales Modell vorhanden (ollama pull llama3)?

Mikrofon stumm:
	pactl set-source-mute <quelle> 0
	pavucontrol → Eingabe-Geräte → Mikro unmute und Pegel ~80%

Port-Konflikt 8080:
	Uvicorn auf anderen Port starten: --port 8081

Windows/macOS:
	Device-Namen unterscheiden sich. Nutze sounddevice.query_devices(), um Namen/Indices zu finden. Setze APP_AUDIO_DEVICE passend.

#### Sicherheit und Datenschutz

Alle Netzwerkaufrufe sind auf 127.0.0.1 beschränkt.
Es gibt keine externen Uploads oder Cloud-Dienste.
Audio wird nur im RAM verarbeitet; keine automatische Speicherung von Roh-Audio.
Logs/Bilder verbleiben lokal.

#### Nächste Schritte

ComfyUI installieren und Workflow-JSON einpflegen.
Snapshot-Intervall feinjustieren (APP_SNAPSHOT_SEC=5.0 für häufigere Bilder).
Prompt-Templates verbessern (z. B. stilistische Leitplanken, Klassenstufen-gerechte Sprache).
Optionale UI-Erweiterungen: Live-Transcript im Browser, Vollbild-Slideshow, „Nur neueste 10 Bilder anzeigen“.



## Lizenz und Kontakt

Das Projekt läuft unter MIT licence.

Contact: Christoph Medicus
dev@betakontext.de
https://dev.betakontext.de

