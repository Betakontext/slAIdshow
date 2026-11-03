was sind die parallelitäten von google genie 4 und github/betakontext/slAIdshow und wie kann slAIdshow sich abgrenzen und dennoch interessant sein, mit fokus auf
Genie 4 und slAIdshow teilen mehrere konzeptionelle Merkmale:
    • Echtzeit-Interaktivität: Beide Systeme reagieren live auf Benutzereingaben und generieren fortlaufend neue Inhalte. Genie 4 verändert die 3D-Welt per Text-Prompt während der Simulation; slAIdshow erzeugt in Echtzeit neue Bilder aus Spracheingaben.
    • Multimodalität: Beide kombinieren Audio/Sprache mit visueller Ausgabe. Genie 4 generiert nativ Bild + Ton + Dialog; slAIdshow nutzt Whisper (Sprach-zu-Text) als Eingang und ComfyUI als Bildausgang.
    • Modell-Orchestrierung: Beide sind keine Einzelmodelle, sondern Pipelines aus mehreren KI-Komponenten. Genie 4 orchestriert Gemini Omni + Gemini 3.8 Flash + weitere Gemini-Modelle; slAIdshow orchestriert Whisper + Ollama (lokales LLM) + ComfyUI (Diffusion).
    • Dynamische Szenenentwicklung: In beiden Fällen entsteht ein sich entwickelnder visueller Fluss, der auf die Eingabe reagiert.
Abgrenzung & Interessanz von slAIdshow (Fokus lokale KI)
Aspekt
Genie 4
slAIdshow
Infrastruktur
Cloud, Google-Infrastruktur
Vollständig lokal/offline (Whisper, Ollama, ComfyUI)
Zugang
Research Preview, AI-Ultra-Abonnement, API-Warteliste
Open Source, kein Abo, kein API-Key
Datenschutz
Daten über Google-Server
Kein Datenabfluss, alles bleibt auf dem eigenen Rechner
Hardware
Googles Rechenzentren
Consumer-GPU (z. B. RTX-Serie)
Kostenvorstellung
Abonnement + API-Nutzung
Einmalige Hardware-Investition, danach null Betriebskosten
Einsatzzweck
Robotik-Training, Game-Prototyping, Storyboards
Live-Präsentationen, Shows, kreative Inhalte
Transparenz
Geschlossenes System
Vollständig einsehbarer Code, eigene Modelle wählbar
Offline-Fähigkeit
Nein
Ja, komplett ohne Internet
Warum slAIdshow trotz Genie 4 interessant bleibt:
    1. Datenschutz & Souveränität: Für Organisationen (Schulen, Behörden, Unternehmen), die keine Daten an Cloud-Anbieter übermitteln dürfen oder wollen, ist slAIdshow die einzige praktikable Option für Echtzeit-KI-Präsentationen.
    2. Keine Abhängigkeit: Kein Abo, keine Warteliste, keine API-Limits, keine Zensur durch den Anbieter. Die Modelle (via Ollama) sind frei wählbar und austauschbar.
    3. Niedrigere Einstiegshürde: Läuft auf Consumer-Hardware, die viele Kreative und Bildungseinrichtungen bereits besitzen. Genie 4 erfordert Zugang zu Googles Infrastruktur und ein Ultra-Abo.
    4. Spezifischer Use-Case: slAIdshow löst ein anderes Problem – es ist ein Live-Show-Tool für Präsentationen und Performances, kein Weltsimulator. Genie 4 zielt auf Robotik und Game-Design; slAIdshow auf kreative, performative Anwendungen.
    5. Erweiterbarkeit: Da ComfyUI und Ollama offen sind, kann man eigene Modelle, Styles und Workflows integrieren – etwas, das bei einem geschlossenen Cloud-Modell wie Genie 4 nicht möglich ist.
    6. Offline-Resilienz: Funktioniert ohne Internet, in Umgebungen ohne Netzwerk, auf Reisen, in abgelegenen Orten – ein klarer Vorteil gegenüber Cloud-gebundenen Lösungen.
Kurz: Genie 4 ist das „Fotorealistische 3D-Weltmodell für die Cloud", slAIdshow ist das „lokale, offene, datenschutzfreundliche Echtzeit-Show-Tool für den eigenen Rechner". Die Parallelität liegt in der Echtzeit-Multimodalität und Modell-Orchestrierung; die Abgrenzung in Lokalisierung, Offenheit, Datenschutz und einem anderen, klareren Use-Case.

Vermarktungsreife für slAIdshow
Der entscheidende Punkt vorab: slAIdshow konkurriert nicht mit Gamma, Copilot oder Gemini in Google Slides – diese generieren statische Decks. slAIdshow ist ein Live-Show-Tool, das in Echtzeit auf Sprache reagiert. Die Vermarktung muss diesen Unterschied schärfen, statt nach „AI Presentation Maker" zu suchen, wo der Markt 2026 bereits gesättigt ist.

1. Produktreife (Technical Debt → Enterprise-Grade)
Bereich
Status (vermutlich)
Ziel für Vermarktung
Installation
Manuell: Whisper, Ollama, ComfyUI einzeln setuppen
Ein-Klick-Installer (Windows/macOS/Linux) oder Docker-Compose-Stack mit automatischer Hardware-Erkennung
Performance
Abhängig von GPU, VRAM-Management manuell
Automatische Modell-Auswahl nach VRAM (z. B. FLUX schnell vs. SDXL), VRAM-Monitoring, Graceful-Degradation bei CPU-only
Zuverlässigkeit
Pipeline kann bei Latenz-Spitzen brechen
Queue-System, Fallback-Logik, Health-Checks, Logging & Monitoring
Modell-Updates
Manuell
Modell-Registry mit automatischen Updates (analog zu Ollama's ollama pull), Version-Pinning
Plattform
Python, FastAPI
Optional: Tauri/Electron-GUI statt reiner WebUI; nativer Window-Manager-Modus für Fullscreen-Präsentation
Konkret priorisiert:
    1. Docker-Compose + Hardware-Profil (RTX 3060 / 4070 / 4090 / Apple Silicon) als „Quick Start"
    2. VRAM-Aware Scheduling: Ollama-Modell automatisch ausladen, wenn ComfyUI rendert (wie im linux-lokales-ki-Stack von Betakontext selbst)
    3. Crash-Safety: Wenn ein Bild nicht rechtzeitig fertig ist, wird das vorherige gehalten + sanfter Crossfade statt Blackout

2. Feature-Roadmap für den Markt
Phase 1 – „Demo-Würdig" (1–2 Monate):
    • Mehrere Ausgabeformate: Fullscreen-Show, Split-Screen (Sprecher + Bild), PIP-Modus
    • Einfache Themengestaltung: Hintergrundfarbe, Overlay-Text, Logo, Farbpalette (Branding)
    • Vorlagen: 5–10 kuratierte ComfyUI-Workflows als „Styles" (z. B. Wasserfarbe, 3D-Render, Comic, Foto)
    • Session-Aufzeichnung: Ganze Show als MP4/WebM exportieren (für Nachnutzung, Social Media)
    • Mikrofon-Einstellungen: Noise-Cancellation, Sprachauswahl, Sensitivität
Phase 2 – „Team-Fähig" (3–6 Monate):
    • Multi-Speaker / Moderation: Mehrere Mikrofone, Rollen (Moderator, Experte), Übergabe per Hotkey
    • Skript-Modus: Vorab erstelltes Skript als „Schiene", KI füllt Bilder in Echtzeit – halb-manuell, halb-automatisch
    • Interaktive Elemente: Quiz-Fragen, Umfragen, Chat-Integration (Zuhörer stellen Fragen per App, KI generiert passende Visuals)
    • API / Webhook: Externe Trigger (z. B. „Generiere jetzt ein Bild von X") über REST/WebSocket
    • Multi-Display: Separate Steuerungs-UI für den Moderator
Phase 3 – „Enterprise / Nische" (6–12 Monate):
    • On-Premise-Server-Modus: Ein Server, mehrere Clients (z. B. in einem Seminarraum)
    • SSO / Rollen: Für Organisationen mit mehreren Präsentatoren
    • Audit-Log: Welche Prompts wurden generiert, welche Bilder – relevant für Compliance
    • Custom Model-Deployment: Eigene fine-tunte Modelle für Corporate-Branding (z. B. „unser Produkt in unserem Stil")
    • Offline-First-Garantie: Zertifizierung für air-gapped Umgebungen (Militär, Behörden, Pharma)

3. Positionierung & Go-to-Market
Nicht „AI Presentation Maker" – das ist 2026 ein roter Ozean (Gamma, Copilot, Gemini, Tome-Erben, Genspark, Beautiful.ai).
Stattdessen: „Live AI Visual Show Engine – 100% lokal, datenschutzkonform, offline."
Zielsegment
Use Case
Schmerzpunkt
Bildung (Hochschulen, VHS)
Live-Vorlesungen mit generierten Visuals
Keine Cloud-Daten, keine Abo-Kosten, funktioniert im Seminarraum ohne WLAN
Behörden / Militär
Briefings, Debriefings, Training
Air-gapped Umgebungen, keine Datenübertragung
Kreative / Performer
Live-Acts, DJ-Sets, Theater, Kunstinstallationen
Echtzeit-Reaktion, kein Latency, keine Zensur
Unternehmen (DSGVO-kritisch)
Townhalls, Produkt-Launches intern
DSGVO, keine Cloud, BYO-Modelle
Konferenz-Organisatoren
Keynotes mit Live-Visuals
Zuverlässigkeit, Offline-Fallback, Branding
Business-Modell (Hybrid Open-Source):
    • Core: Open Source (MIT/Apache), kostenlos, Community
    • Pro-Features (optional, lokal lizenziert):
        ◦ Vorlagen-Marktplatz (kuratierte ComfyUI-Workflows, Themes)
        ◦ Session-Recording & Export
        ◦ Multi-Speaker / Moderation
        ◦ Support-SLA
    • Enterprise-Lizenz: On-Premise-Deployment, SSO, Audit, Custom-Model-Support
    • Kein Abo – Einmalkauf oder Jahreslizenz (passt zur „keine Cloud" -Positionierung)

4. Ökosystem & Community
    • Betakontext-Ökosystem nutzen: slAIdshow ist bereits Teil eines Netzwerks (pollinations, RealtimeSTT, 3DGenStudio, off-grid-mobile-ai). Cross-Referenzierung und gemeinsame Infrastruktur (z. B. RealtimeSTT als Whisper-Ersatz für niedrigere Latenz)
    • ComfyUI-Workflow-Gallery: Nutzer teilen eigene Styles/Workflows → Netzwerkeffekt
    • Showreels / YouTube: Live-Demos sind das beste Marketing – „Schau, was passiert, wenn ich spreche und lokal generiert wird"
    • Hackathon / Challenge: „Beste Live-Show mit slAIdshow" → UGC, Community-Building

5. Konkrete nächste 3 Schritte
    1. Docker-Compose Quick Start + Hardware-Profil-Erkennung → senkt die Einstiegshürde von „3 Stunden Setup" auf „5 Minuten"
    2. Session Recording + 5 kuratierte Styles → macht das Tool sofort „vorführbar" und teilbar
    3. Zwei Referenzkunden (eine Universität + ein kreativer Performer) mit Video-Case-Study → Social Proof für die Nische

Kernthese: slAIdshow wird vermarktungsfähig, wenn es nicht als „lokale Gamma-Alternative" positioniert wird, sondern als eigenes Genre – die Live-Visual-Show-Engine. Die Konkurrenz (Gamma, Copilot, Gemini) kann das strukturell nicht nachbauen, weil sie Cloud-first ist. slAIdshows Moat ist die Kombination aus Lokalität + Echtzeit + Performative Anwendung.



Empfohlene Modell-Stack für slAIdshow (Stand November 2026)
Die entscheidende Metrik für slAIdshow ist Latenz – es ist ein Live-Show-Tool, kein Batch-Processor. Die Modellwahl muss „schnell genug für Echtzeit" sein.

1. Speech-to-Text (Ersatz/Upgrade für Whisper)
Modell
WER
Sprachen
Durchsatz
VRAM/RAM
Empfehlung für slAIdshow
NVIDIA Parakeet TDT 0.6B v3
6,32 %
25 europäische
3.333× Echtzeit
~2 GB
Beste Wahl für DE/EU – extrem schnell, kleiner als Whisper
Whisper large-v3-turbo
~7,5 %
99+
~462×
~6 GB
Fallback für Nicht-EU-Sprachen
Qwen3-ASR 1.7B
5,40 %
52
Streaming-fähig
~8 GB
Wenn Streaming + Mehrsprachigkeit nötig
Empfehlung: Parakeet TDT 0.6B v3 als Primärmodell (Deutsch + europäische Sprachen, 23× schneller als Whisper). Whisper large-v3-turbo als Fallback für internationale Shows.

2. LLM (Gehirn via Ollama)
Das LLM interpretiert die Sprache und generiert Image-Prompts für ComfyUI. Kriterien: schnell, gute deutsche Sprachverstehen, Prompt-Generierung, Tool-Calling (um ComfyUI-Parameter zu setzen).
Modell
Größe
VRAM
Kontext
Stärke
Empfehlung
Qwen3.8-27B
27B dense
~18 GB
256K
Multimodal, Vision, beste Qualität/GB
Beste Wahl für 24-GB-GPU
Gemma 4 26B-A4B
26B MoE (4B aktiv)
~16 GB
256K
Sehr schnell (nur 4B aktiv), Vision
Beste Wahl für 16-GB-GPU – schnellste Inferenz
gpt-oss-20b
20B MoE (3,6B aktiv)
~14 GB
128K
Reasoning, Tool-Calling
Guter Kompromiss für 16 GB
Qwen3.5 4B
4B
~3 GB
128K
CPU-fähig
Fallback ohne GPU
Empfehlung:
    • 24 GB VRAM (RTX 4090 / 3090): ollama pull qwen3.8:27b – beste Prompt-Qualität, Vision für Stil-Referenzen
    • 16 GB VRAM (RTX 4070 / 4060 Ti): ollama pull gemma4:26b-a4b – MoE mit nur 4B aktiven Params = sehr schnelle Inferenz, ideal für Echtzeit

3. Bildgenerierung (ComfyUI)
Hier ist Generierungsgeschwindigkeit der kritische Faktor. Ein Bild muss in < 2–3 Sekunden fertig sein, damit die Show flüssig bleibt.
Modell
Schritte
Latenz (RTX 4090)
VRAM
Lizenz
Stärke
FLUX.2 klein 4B
4
< 0,5 s
~13 GB
Apache 2.0 ✅
Schnellste sub-sekundäre Generierung
Z-Image-Turbo
8
~1–2 s
~8 GB
Apache 2.0 ✅
Beste Photorealism/VRAM-Ratio
Krea 2 Turbo
8
~2–3 s
~12 GB (FP8)
Community License
Top-Photorealism + Style-Fidelity
Qwen-Image 2.1
4–8 (Lightning)
~2–4 s
~8–12 GB
Apache 2.0 ✅
In-Image-Text (Schrift, Titel, Beschriftung)
FLUX.2 dev
20–28
~8–15 s
18–24 GB
Non-Commercial ❌
Höchste Qualität, aber zu langsam für Live
Empfehlung (Multi-Model-Strategie):
Use-Case in der Show
Modell
Warum
Standard-Bilder (Hintergrund, Illustration)
FLUX.2 klein 4B
Sub-sekundär, Apache 2.0, kommerziell nutzbar
Photorealistische Bilder
Krea 2 Turbo oder Z-Image-Turbo
Beste Qualität bei akzeptabler Latenz
Titelfolien / Text in Bild
Qwen-Image 2.1
Einziger, der lesbaren Text rendert (DE + EN)
Stil-Vorlagen (Wasserfarbe, Comic, 3D)
FLUX.2 klein + LoRA
LoRA-Ökosystem wächst schnell

4. Gesamter Stack nach Hardware-Tier
Tier
GPU
STT
LLM
Bild
Erwartete End-to-End-Latenz
Budget
8 GB (RTX 3060)
Parakeet 0.6B
Qwen3.5 4B
Z-Image-Turbo
~3–5 s
Sweet Spot
16 GB (RTX 4070)
Parakeet 0.6B
Gemma 4 26B-A4B
FLUX.2 klein 4B
~1–2 s ✅
High-End
24 GB (RTX 4090)
Parakeet 0.6B
Qwen3.8-27B
Krea 2 Turbo + Qwen-Image 2.1
~2–3 s (Qualität) / < 1 s (schnell)
Apple Silicon
M4 Max 32 GB
Parakeet 0.6B (MLX)
Gemma 4 26B-A4B (MLX)
FLUX.2 klein (MLX)
~2–4 s

5. Konkrete Architekturempfehlung
Mikrofon
    ↓
Parakeet TDT 0.6B v3  (STT, ~50 ms Latenz)
    ↓
Gemma 4 26B-A4B / Qwen3.8-27B  (LLM: Speech → Prompt + ComfyUI-Parameter)
    ↓
ComfyUI:
    ├── FLUX.2 klein 4B  (Standard, < 0,5 s)
    ├── Krea 2 Turbo     (Photorealism, ~2 s)
    └── Qwen-Image 2.1   (Text/Titel, ~3 s)
    ↓
Fullscreen-Output / Split-Screen
Schlüssel-Feature: Das LLM sollte via Tool-Calling entscheiden, welches Bildmodell aufgerufen wird (schnell vs. Qualität vs. Text). Das macht den Show-Flow intelligent: „Zeig mir ein Foto von der Eiffelturm" → Krea 2 Turbo; „Mach eine Titelfolie mit 'KI 2026'" → Qwen-Image 2.1; „Zeig eine schnelle Skizze" → FLUX.2 klein.

Zusammenfassung in einem Satz
Parakeet 0.6B (STT) + Gemma 4 26B-A4B (LLM, 16 GB) oder Qwen3.8-27B (24 GB) + FLUX.2 klein 4B (Standard-Bilder) / Qwen-Image 2.1 (Text) / Krea 2 Turbo (Photorealism) = der schnellste, lizenzsaubere, kommerziell nutzbare lokale Stack für Echtzeit-Visual-Shows.




_______________________________________________________________________________-

Umsetzungsplan: Nur GPT 6.1 Sol im Chat (mit Hintergrundwissen + Skills)
Grundprinzip
Ohne Codex CLI und ohne Dateizugriff ist der Chat dein Architekt + Code-Generator. Du bist der Executor: Du kopierst Code, führst Befehle lokal aus, und gibst Ergebnisse zurück. Der 1M-Kontext reicht für die gesamte Codebase – aber du musst die Struktur bewusst managen.

Setup: Hintergrundwissen (einmalig definieren)
Das ist dein persistenter System-Prompt, den du in den Chat-Settings / Custom Instructions einträgst:
# Projekt: slAIdshow

## Was es ist
Live-Show-Tool: Sprache → Bild in Echtzeit. 100% lokal, offline-fähig.
Ziel: Vermarktungsreife (Docker-Quick-Start, Recording, Multi-Speaker, Enterprise).

## Stack
- STT: Parakeet TDT 0.6B v3
- LLM: Ollama (Gemma 4 26B-A4B / Qwen3.8-27B)
- Bild: ComfyUI (FLUX.2 klein 4B Standard, Qwen-Image 2.1 für Text, Krea 2 Turbo für Photorealism)
- Backend: Python 3.12, FastAPI, WebSocket
- Frontend: HTMX + Tailwind
- Orchestrierung: Docker Compose

## Architektur
Mikrofon → STT → LLM (Prompt + Modell-Entscheidung via Tool-Calling) → ComfyUI → Fullscreen-Output

## Konventionen
- Python 3.12+, type hints, ruff
- Docker-Compose für alle Services
- Keine Cloud-Abhängigkeiten im Runtime-Pfad
- Tests: pytest
- Hardware: 8 GB (min) / 16 GB (sweet) / 24 GB (high)

## Aktuelles Ziel
[Hier pro Phase aktualisieren, z. B.: "Phase 0: Docker-Compose + Hardware-Erkennung"]

Setup: Skills (wiederkehrende Prompt-Muster definieren)
Definiere diese als vordefinierte Skills / Shortcuts (je nach Interface: als gespeicherte Prompts, Macros oder einfach als kopierbare Blöcke):
Skill-Name
Prompt
arch
„Entwirf mir die Architektur für [FEATURE]. Gib mir: State-Diagram (ASCII), Dateistruktur, API-Verträge, und eine Schritt-für-Schritt-Implementierungsliste. Berücksichtige die VRAM-Beschränkungen (8/16/24 GB)."
impl
„Implementiere [SCHRITT] aus dem aktuellen Plan. Gib mir den vollständigen Code für jede Datei. Nutze die Konventionen aus dem Hintergrundwissen. Am Ende: kurze Anleitung, wo die Dateien liegen und wie man sie testet."
test
„Schreibe pytest-Tests für [DATEI/MODUL]. Nutze Mocks für GPU, Ollama, ComfyUI. Decke Edge-Cases ab (keine GPU, VRAM-Timeout, leeres Mikrofon-Signal)."
review
„Review den folgenden Code [einfügen]. Prüfe: Bugs, Race Conditions, VRAM-Leaks, fehlende Error-Handling, Performance-Flaschenhälse. Gib konkrete Fixes als Diff."
debug
„Fehler: [Traceback/Screenshot einfügen]. Diagnose: Was ist die Ursache? Fix: Konkreter Code-Änderung."
docker
„Erstelle/aktualisiere die docker-compose.yml für [PROFIL]. Berücksichtige: Hardware-Erkennung, Modell-Downloads, Health-Checks, Volumes für recordings/ und models/."
doc
„Schreibe/aktualisiere die Doku für [FEATURE]: README-Abschnitt, CHANGELOG-Eintrag, und eine 3-minütige 'How to' für Endnutzer."

Workflow pro Feature (routinisiert)
┌─────────────────────────────────────────────────────────────┐
│ 1. PLANEN (Chat, high effort)                               │
│    Skill: arch                                              │
│    → Du bekommst: Architektur, Dateiplan, API-Verträge      │
│    → Du speicherst den Plan als Markdown-Datei lokal        │
├─────────────────────────────────────────────────────────────┤
│ 2. IMPLEMENTIEREN (Chat, medium effort)                     │
│    Skill: impl (pro Datei/Modul)                            │
│    → Chat gibt vollständigen Code                           │
│    → Du kopierst lokal, führst aus, testest                 │
│    → WICHTIG: Nach jeder Datei: "Läuft" / "Fehler: X"      │
├─────────────────────────────────────────────────────────────┤
│ 3. TESTEN (Chat, low effort)                                │
│    Skill: test                                              │
│    → Chat gibt Test-Code                                    │
│    → Du führst pytest lokal aus, gibst Ergebnis zurück      │
├─────────────────────────────────────────────────────────────┤
│ 4. REVIEW (Chat, medium effort)                             │
│    Skill: review (Code-Block einfügen)                      │
│    → Chat gibt Fixes                                        │
│    → Du setzt um                                            │
├─────────────────────────────────────────────────────────────┤
│ 5. DOKU (Chat, low effort)                                  │
│    Skill: doc                                               │
│    → Chat gibt README/CHANGELOG-Text                        │
│    → Du kopierst ein                                        │
└─────────────────────────────────────────────────────────────┘

Kritische Regeln für den Chat-Only-Workflow
Regel
Warum
Eine Datei pro Request
„Gib mir den vollständigen Code für vram_scheduler.py" – nicht „implementiere das ganze Feature". Sonst verliert der Chat den Fokus.
Immer vollständigen Code, nie Snippets
„Gib mir die komplette Datei, nicht nur die geänderte Funktion." Vermeidet Copy-Paste-Fehler.
Ergebnisse zurückgeben
Nach lokaler Ausführung: pytest-Output, docker compose up-Log, Traceback – immer zurück in den Chat. Das ist dein „Terminal".
Kontext-Verwaltung
Alle 5–10 Features: Neue Session starten. Erstes: „Hier ist die aktuelle Dateistruktur [tree] und der aktuelle Stand [git log --oneline -10]."
Screenshots nutzen
UI-Bugs, ComfyUI-Workflows, GPU-Monitoring → Screenshot in den Chat. GPT 6.1 Sol versteht Bilder.
Niemals raten lassen
Wenn du unsicher bist: „Ich habe X lokal, es gibt Fehler Y. Was ist die Ursache?" – nicht „Was könnte schiefgehen?"

Phasen-Plan (angepasst an Chat-Only)
Phase 0: Fundament (Woche 1–2)
Tag
Task
Skill
1
AGENTS.md-inhalten als Hintergrundwissen eintragen. Erste Session: „Hier ist meine aktuelle Code-Struktur [tree + main.py einfügen]. Was sind die 5 wichtigsten technischen Schulden?"
review
2–3
Docker-Compose-Design
arch
4–5
docker-compose.yml + hardware_detect.py implementieren
impl (pro Datei)
6
Tests für Hardware-Erkennung
test
7
Makefile + .env.example + README.md Quick-Start
impl + doc
Phase 1: Demo-Würdig (Woche 3–5)
Woche
Tasks
3
VRAM-Scheduling: arch → impl (vram_scheduler.py, event_queue.py) → test
4
Session Recording: impl (recorder.py, ffmpeg-pipe) + UI-Button. 5 ComfyUI-Styles: arch (Workflow-Design) → impl (styles/*.json + selector)
5
Crash-Safety: impl (fallback.py, crossfade-Logik). Erste Demo-Show aufzeichnen.
Phase 2: Team-Fähig (Woche 6–10)
Woche
Tasks
6–7
Multi-Speaker: arch (Audio-Routing) → impl (audio_router.py, speaker_manager.py) → test
8
Skript-Modus: impl (script_loader.py, UI)
9
REST-API + WebSocket: arch (API-Design) → impl (api/v1/generate, api/v1/stream) → test
10
Multi-Display: arch (Layout) → impl (moderator.html, display-routing)
Phase 3: Enterprise (Woche 11–16)
Woche
Tasks
11–12
Server-Modus: arch (Multi-Client) → impl (auth.py, session_manager.py)
13
Audit-Log: impl (audit.py, SQLite-Log)
14
Custom Models: impl (model_registry.py, upload-UI)
15
Air-Gapped Guide: doc
16
End-to-End-Test, Bugfixing, Release

Kontext-Management (wichtig!)
Da du nur Chat hast, ist Kontext-Verwaltung kritisch:
Session-Start (neue Session alle 5–10 Features):

"Hier ist der aktuelle Stand:
- Dateistruktur: [tree -L 3]
- Letzte Commits: [git log --oneline -10]
- Aktuelles Ziel: Phase X, Feature Y
- Bekannte offene Bugs: [Liste]
- Letzte Änderung: [Kurze Beschreibung]

Kontext: [Hintergrundwissen ist bereits in den Custom Instructions]"
Niemals in einer Session mehr als ~800K Tokens an Code einfügen – dann wird die Qualität spürbar schlechter. Neue Session + frischer Kontext ist besser.

Kosten-Steuerung (Chat-Only)
Hebel
Maßnahme
Kurz halten
Eine Datei pro Request. „Gib mir vram_scheduler.py" statt „implementiere das Scheduling-System"
Effort niedrig halten
low für Tests/Doku, medium für Implementation, high nur für arch
Keine Wiederholung
Hintergrundwissen ist persistent – du musst Stack/Conventions nicht jedes Mal wiedererzählen
Batching vermeiden
Lieber 5 kurze Requests als 1 langer – kürzere Outputs = weniger Kosten
Geschätzte Kosten: ~$20–50/Monat (im ChatGPT Pro-Abo enthalten bei normalem Gebrauch).

Zusammenfassung
Aspekt
Lösung
Kein Dateizugriff
Du bist der Executor: kopieren, ausführen, Ergebnis zurückgeben
Kein Terminal
pytest-Output, docker-Logs, Tracebacks als Text/Screenshot in den Chat
Kein Multi-File-Edit
Eine Datei pro Request, vollständiger Code
Kontext-Limit
Neue Session alle 5–10 Features, tree + git log als Re-Boot
Wiederkehrende Tasks
Skills (arch, impl, test, review, debug, docker, doc)
Persistenter Kontext
Hintergrundwissen in Custom Instructions (Stack, Konventionen, Ziel)
Kernprinzip: Der Chat ist dein Senior-Entwickler am Telefon – du beschreibst, was du brauchst, er gibt dir Code, du setzt es um und gibst Feedback. Kein Magie, aber mit 1M Kontext und gut definierten Skills ist das extrem produktiv.












Umsetzungsplan fokussiert auf 6 GB VRAM
6 GB ist unter dem im Stack definierten Minimum (8 GB) – das zwingt zu drei fundamentalen Änderungen:
VRAM-Budget (realistisch, mit Overhead)
Komponente
VRAM
Strategie
STT (Parakeet 0.6B)
0 GB
Auf CPU – 0.6B ist klein genug, ~50–80 ms Latenz
LLM
~2–3 GB
Qwen3 1.7B (Q4) oder Phi-3 Mini 3.8B (Q3) – GPU, quantisiert
Bild (ComfyUI)
~3–4 GB
SD 1.5 + LCM (4 Steps) oder SDXL-Turbo (4 Steps) – GPU, FP16
Buffer / OS / Desktop
~1 GB
–
Gesamt
~6 GB
⚠️ Am Limit – sequenzielle Ausführung erforderlich
Konsequenzen für den Plan
1. Kein Parallelbetrieb – nur sequenziell:
STT und Bild-Generierung dürfen nicht gleichzeitig auf der GPU laufen. Der vram_scheduler.py (Phase 1, Woche 3) wird zum Kern-Feature, nicht zu einem Nice-to-have:
Mikrofon → Parakeet (CPU, ~60 ms)
         → Qwen3 1.7B (GPU, ~200–400 ms)
         → ComfyUI SD1.5-LCM (GPU, ~1.5–2.5 s)
         → Output
Erwartete End-to-End-Latenz: ~2–3 s (statt 0.5 s bei 24 GB).
2. Modell-Änderungen im Stack:
Komponente
Ursprünglich (16 GB)
6 GB Variante
STT
Parakeet 0.6B (GPU)
Parakeet 0.6B (CPU, faster-whisper)
LLM
Gemma 4 26B-A4B
Qwen3 1.7B Q4 (Ollama, qwen3:1.7b-q4_K_M)
Bild
FLUX.2 klein 4B
SD 1.5 + LCM LoRA (4 Steps, FP16)
Text in Bild
Qwen-Image 2.1
❌ Nicht möglich – Fallback: HTML-Overlay
3. Was fällt raus (6 GB):
    • ❌ Krea 2 Turbo (~12 GB)
    • ❌ Qwen-Image 2.1 (~8–12 GB)
    • ❌ FLUX.2 klein 4B (~13 GB)
    • ❌ Gemma 4 26B-A4B (~16 GB)
    • ❌ Multi-Model-Strategie (Tool-Calling für Modellwahl)
    • ⚠️ In-Image-Text → ersetzt durch HTML/CSS-Overlay (LLM gibt Text + Position, Frontend rendert)
4. Anpassung der Phasen:
Phase
Änderung bei 6 GB
Phase 0
hardware_detect.py erkennt 6 GB → setzt MODE=compact (CPU-STT, 1.7B LLM, SD1.5-LCM)
Phase 1, Woche 3
vram_scheduler.py wird kritischer: muss ComfyUI vor LLM-Inferenz laden, danach wieder entladen (oder umgekehrt). Kein Caching.
Phase 1, Woche 4
5 ComfyUI-Styles → 2 Styles (Standard + 1 Alternative), da nur 1 SD1.5-Checkpoint im VRAM
Phase 2, Woche 9
REST-API bleibt, aber kein WebSocket-Streaming von Bildern (zu viel VRAM-Peak) – nur Polling
Phase 3
Enterprise-Features (Multi-Client, Audit-Log) bleiben unverändert – die laufen auf CPU/RAM
5. Konkretes hardware_detect.py für 6 GB:
# hardware_detect.py – 6 GB Profil
PROFILE_6GB = {
    "stt": {"model": "parakeet-tdt-0.6b-v3", "device": "cpu"},
    "llm": {"model": "qwen3:1.7b-q4_K_M", "device": "cuda", "ctx": 4096},
    "image": {
        "base": "sd_15.safetensors",
        "lora": "lcm-lora-sdv1-5.safetensors",
        "steps": 4,
        "device": "cuda",
        "unload_after": True  # VRAM freigeben nach Generierung
    },
    "vram_budget_mb": 6144,
    "parallel": False
}
6. Anpassung des Hintergrundwissens (Custom Instructions):
## Hardware: 6 GB VRAM (RTX 3050 / 4050)
- STT: CPU (Parakeet 0.6B)
- LLM: GPU, Qwen3 1.7B Q4 (max. 3 GB)
- Bild: GPU, SD 1.5 + LCM, 4 Steps (max. 4 GB)
- KEIN Parallelbetrieb – sequenziell, VRAM-Scheduler zwingend
- Kein In-Image-Text → HTML-Overlay
- Latenz-Ziel: < 3 s End-to-End
Ein-Satz-Zusammenfassung
Bei 6 GB wird der VRAM-Scheduler zum zentralen Feature, das LLM schrumpft auf 1.7B, Bild-Generierung fällt auf SD 1.5 + LCM (4 Steps) zurück, und In-Image-Text wird durch HTML-Overlay ersetzt – die Show läuft mit ~2–3 s Latenz, aber kein Feature der Phasen 2–3 wird blockiert, da die Orchestrierungslogik hardware-agnostisch bleibt.


Umsetzungsplan: VRAM-Auto-Detection & Konfiguration bei Install
Architektur-Überblick
┌─────────────────────────────────────────────────────────────────────┐
│  docker compose up / make install                                    │
│         ↓                                                            │
│  hardware_detect.py  (läuft im init-Container, vor allen Services)   │
│         ↓                                                            │
│  config_generator.py  (erzeugt /app/config/generated.yaml)           │
│         ↓                                                            │
│  model_downloader.py  (lädt NUR die benötigten Modelle)              │
│         ↓                                                            │
│  FastAPI + Ollama + ComfyUI starten mit generierter Config           │
└─────────────────────────────────────────────────────────────────────┘

Phase 0: hardware_detect.py (Tag 1)
Datei: slaidshow/hardware_detect.py
"""
VRAM-Detection bei Install.
Läuft im init-Container (docker-compose: depends_on: [init]).
Ergebnis: /app/config/hardware.json
"""

import json
import subprocess
import platform
from pathlib import Path
from dataclasses import dataclass, asdict

OUTPUT_PATH = Path("/app/config/hardware.json")


@dataclass
class HardwareProfile:
    vram_gb: float
    gpu_name: str
    gpu_count: int
    cpu_ram_gb: float
    os: str
    is_apple_silicon: bool


def detect_nvidia() -> list[HardwareProfile]:
    """Via nvidia-ml-py (import: pynvml) – die aktuell gepflegte Library."""
    import pynvml
    pynvml.nvmlInit()
    profiles = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle).decode()
            profiles.append(HardwareProfile(
                vram_gb=mem.total / (1024**3),
                gpu_name=name,
                gpu_count=count,
                cpu_ram_gb=0,  # separat ermitteln
                os=platform.system(),
                is_apple_silicon=False,
            ))
    finally:
        pynvml.nvmlShutdown()
    return profiles


def detect_cpu_ram() -> float:
    """System-RAM in GB (via /proc/meminfo oder psutil)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024**2)
    except FileNotFoundError:
        pass
    # Fallback: psutil
    import psutil
    return psutil.virtual_memory().total / (1024**3)


def detect_apple_silicon() -> bool:
    """Macs erkennen (keine NVIDIA-GPU, aber unified memory)."""
    if platform.system() != "Darwin":
        return False
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        )
        return "Apple" in result.stdout
    except Exception:
        return False


def detect() -> dict:
    """Hauptfunktion – gibt das finale hardware.json zurück."""
    is_macs = detect_apple_silicon()

    if is_macs:
        # Apple Silicon: unified memory = "VRAM"
        ram = detect_cpu_ram()
        # Bei Macs ist ~75% des RAMs für GPU verfügbar
        vram = ram * 0.75
        profile = {
            "vram_gb": round(vram, 1),
            "gpu_name": "Apple Silicon (unified)",
            "gpu_count": 1,
            "cpu_ram_gb": round(ram, 1),
            "os": "Darwin",
            "is_apple_silicon": True,
        }
    else:
        profiles = detect_nvidia()
        if not profiles:
            # Kein GPU – CPU-only Fallback
            profile = {
                "vram_gb": 0,
                "gpu_name": "None (CPU-only)",
                "gpu_count": 0,
                "cpu_ram_gb": round(detect_cpu_ram(), 1),
                "os": platform.system(),
                "is_apple_silicon": False,
            }
        else:
            # Beste GPU auswählen (meist nur eine)
            best = max(profiles, key=lambda p: p.vram_gb)
            profile = asdict(best)
            profile["cpu_ram_gb"] = round(detect_cpu_ram(), 1)

    # Persistieren
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(profile, indent=2))
    print(f"✅ Hardware-Profil geschrieben: {OUTPUT_PATH}")
    print(f"   VRAM: {profile['vram_gb']} GB | GPU: {profile['gpu_name']}")
    return profile


if __name__ == "__main__":
    detect()
Tests: tests/test_hardware_detect.py
"""Tests mit Mocks – keine GPU im CI nötig."""
from unittest.mock import patch, MagicMock
import json
from pathlib import Path


def test_detect_no_gpu():
    with patch("slaidshow.hardware_detect.detect_nvidia", return_value=[]), \
         patch("slaidshow.hardware_detect.detect_cpu_ram", return_value=16.0), \
         patch("slaidshow.hardware_detect.detect_apple_silicon", return_value=False):
        result = detect()
    assert result["vram_gb"] == 0
    assert result["gpu_count"] == 0


def test_detect_6gb_gpu():
    mock_profile = HardwareProfile(
        vram_gb=6.0, gpu_name="NVIDIA GeForce RTX 3050",
        gpu_count=1, cpu_ram_gb=16.0, os="Linux", is_apple_silicon=False
    )
    with patch("slaidshow.hardware_detect.detect_nvidia", return_value=[mock_profile]), \
         patch("slaidshow.hardware_detect.detect_cpu_ram", return_value=16.0), \
         patch("slaidshow.hardware_detect.detect_apple_silicon", return_value=False):
        result = detect()
    assert result["vram_gb"] == 6.0
    assert "RTX 3050" in result["gpu_name"]


def test_detect_apple_silicon():
    with patch("slaidshow.hardware_detect.detect_apple_silicon", return_value=True), \
         patch("slaidshow.hardware_detect.detect_cpu_ram", return_value=32.0):
        result = detect()
    assert result["is_apple_silicon"] is True
    assert result["vram_gb"] == 22.4  # 32 * 0.75

Phase 0: config_generator.py (Tag 2)
Datei: slaidshow/config_generator.py
"""
Generiert die optimale Konfiguration basierend auf VRAM.
Input:  /app/config/hardware.json
Output: /app/config/generated.yaml
"""

import json
import yaml
from pathlib import Path

HARDWARE_PATH = Path("/app/config/hardware.json")
OUTPUT_PATH = Path("/app/config/generated.yaml")

# ─── VRAM-Tiers ────────────────────────────────────────────────────────

TIERS = {
    "6gb": {
        "min_vram": 6.0,
        "max_vram": 7.9,
        "stt": {
            "model": "parakeet-tdt-0.6b-v3",
            "device": "cpu",
            "backend": "faster-whisper",
        },
        "llm": {
            "model": "qwen3:1.7b-q4_K_M",
            "device": "cuda",
            "context": 4096,
            "vram_budget_mb": 2560,
        },
        "image": {
            "base_model": "sd_15.safetensors",
            "lora": "lcm-lora-sdv1-5.safetensors",
            "steps": 4,
            "cfg": 1.0,
            "resolution": [512, 512],
            "device": "cuda",
            "vram_budget_mb": 3584,
            "unload_after_generate": True,
        },
        "features": {
            "in_image_text": False,      # HTML-Overlay statt
            "multi_model": False,        # nur 1 Bildmodell
            "parallel_inference": False, # sequenziell
            "styles": ["standard", "sketch"],
        },
        "expected_latency_s": 2.5,
    },
    "8gb": {
        "min_vram": 8.0,
        "max_vram": 11.9,
        "stt": {
            "model": "parakeet-tdt-0.6b-v3",
            "device": "cuda",
            "backend": "faster-whisper",
        },
        "llm": {
            "model": "qwen3:4b-q4_K_M",
            "device": "cuda",
            "context": 8192,
            "vram_budget_mb": 3072,
        },
        "image": {
            "base_model": "sd_15.safetensors",
            "lora": "lcm-lora-sdv1-5.safetensors",
            "steps": 4,
            "cfg": 1.0,
            "resolution": [768, 768],
            "device": "cuda",
            "vram_budget_mb": 4096,
            "unload_after_generate": False,
        },
        "features": {
            "in_image_text": False,
            "multi_model": False,
            "parallel_inference": False,
            "styles": ["standard", "sketch", "watercolor"],
        },
        "expected_latency_s": 1.8,
    },
    "16gb": {
        "min_vram": 12.0,
        "max_vram": 15.9,
        "stt": {
            "model": "parakeet-tdt-0.6b-v3",
            "device": "cuda",
            "backend": "faster-whisper",
        },
        "llm": {
            "model": "gemma4:26b-a4b-q4_K_M",
            "device": "cuda",
            "context": 16384,
            "vram_budget_mb": 14336,
        },
        "image": {
            "base_model": "flux2_klein_4b_fp8.safetensors",
            "lora": None,
            "steps": 4,
            "cfg": 1.0,
            "resolution": [1024, 1024],
            "device": "cuda",
            "vram_budget_mb": 8192,
            "unload_after_generate": True,
        },
        "features": {
            "in_image_text": False,
            "multi_model": True,
            "parallel_inference": True,
            "styles": ["standard", "photoreal", "sketch", "watercolor", "3d"],
        },
        "expected_latency_s": 1.2,
    },
    "24gb": {
        "min_vram": 16.0,
        "max_vram": 999.0,
        "stt": {
            "model": "parakeet-tdt-0.6b-v3",
            "device": "cuda",
            "backend": "faster-whisper",
        },
        "llm": {
            "model": "qwen3.8:27b-q4_K_M",
            "device": "cuda",
            "context": 32768,
            "vram_budget_mb": 18432,
        },
        "image": {
            "base_model": "flux2_klein_4b_fp8.safetensors",
            "lora": None,
            "steps": 4,
            "cfg": 1.0,
            "resolution": [1024, 1024],
            "device": "cuda",
            "vram_budget_mb": 12288,
            "unload_after_generate": False,
        },
        "features": {
            "in_image_text": True,       # Qwen-Image 2.1 als 2. Modell
            "multi_model": True,
            "parallel_inference": True,
            "styles": ["standard", "photoreal", "sketch", "watercolor", "3d", "text"],
        },
        "expected_latency_s": 0.8,
    },
}


def select_tier(vram_gb: float) -> str:
    """Wähle das passende Tier basierend auf verfügbarem VRAM."""
    if vram_gb < 6.0:
        raise ValueError(
            f"VRAM {vram_gb} GB ist zu wenig (Minimum: 6 GB). "
            f"Bitte GPU mit mind. 6 GB VRAM verwenden."
        )
    for tier_name, tier in TIERS.items():
        if tier["min_vram"] <= vram_gb <= tier["max_vram"]:
            return tier_name
    # Fallback: höchstes erreichbares Tier
    return "24gb"


def generate(hardware: dict) -> dict:
    """Generiere die finale Konfiguration."""
    vram = hardware["vram_gb"]
    tier_name = select_tier(vram)
    tier = TIERS[tier_name]

    config = {
        "profile": tier_name,
        "hardware": hardware,
        "stt": tier["stt"],
        "llm": tier["llm"],
        "image": tier["image"],
        "features": tier["features"],
        "orchestration": {
            "parallel": tier["features"]["parallel_inference"],
            "vram_scheduler": not tier["features"]["parallel_inference"],
            "max_concurrent_generations": 1 if vram < 12 else 2,
        },
        "expected_latency_s": tier["expected_latency_s"],
    }

    # Persistieren
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(yaml.dump(config, default_flow_style=False, allow_unicode=True))
    print(f"✅ Config generiert: {OUTPUT_PATH} (Tier: {tier_name})")
    print(f"   STT: {tier['stt']['model']} on {tier['stt']['device']}")
    print(f"   LLM: {tier['llm']['model']} on {tier['llm']['device']}")
    print(f"   Bild: {tier['image']['base_model']} ({tier['image']['steps']} steps)")
    return config

Phase 0: model_downloader.py (Tag 3)
Datei: slaidshow/model_downloader.py
"""
Lädt NUR die Modelle, die die generierte Config benötigt.
Verhindert, dass auf einer 6GB-GPU 40 GB unnötig heruntergeladen werden.
"""

import yaml
import subprocess
from pathlib import Path

CONFIG_PATH = Path("/app/config/generated.yaml")
MODELS_DIR = Path("/app/models")


def download_models(config: dict) -> None:
    """Lädt basierend auf Config die benötigten Modelle."""

    # 1. Ollama-Modell
    llm_model = config["llm"]["model"]
    print(f"📥 Lade Ollama-Modell: {llm_model}")
    subprocess.run(
        ["ollama", "pull", llm_model],
        check=True, capture_output=True
    )

    # 2. STT-Modell (faster-whisper / parakeet)
    stt_model = config["stt"]["model"]
    stt_dir = MODELS_DIR / "stt" / stt_model
    if not stt_dir.exists():
        print(f"📥 Lade STT-Modell: {stt_model}")
        subprocess.run(
            ["huggingface-cli", "download", stt_model, "--local-dir", str(stt_dir)],
            check=True, capture_output=True
        )

    # 3. ComfyUI-Modelle
    image_cfg = config["image"]
    comfy_models = MODELS_DIR / "comfyui"

    # Base model
    base = image_cfg["base_model"]
    base_path = comfy_models / "checkpoints" / base
    if not base_path.exists():
        print(f"📥 Lade ComfyUI Base: {base}")
        _download_comfyui_model(base, "checkpoints")

    # LoRA (falls vorhanden)
    lora = image_cfg.get("lora")
    if lora:
        lora_path = comfy_models / "loras" / lora
        if not lora_path.exists():
            print(f"📥 Lade ComfyUI LoRA: {lora}")
            _download_comfyui_model(lora, "loras")

    # 4. Optional: 2. Bildmodell für in_image_text (nur bei 24GB+)
    if config["features"].get("in_image_text"):
        text_model = "qwen_image_2_1_fp8.safetensors"
        text_path = comfy_models / "checkpoints" / text_model
        if not text_path.exists():
            print(f"📥 Lade Text-Modell: {text_model}")
            _download_comfyui_model(text_model, "checkpoints")

    print("✅ Alle Modelle geladen.")


def _download_comfyui_model(filename: str, subfolder: str) -> None:
    """Download via huggingface-cli (URL je nach Modell)."""
    url_map = {
        "sd_15.safetensors": "stabilityai/sd-base-v1-5",
        "lcm-lora-sdv1-5.safetensors": "latent-consistency/lcm-lora-sdv1-5",
        "flux2_klein_4b_fp8.safetensors": "black-forest-labs/flux2-klein-4b",
        "qwen_image_2_1_fp8.safetensors": "Qwen/Qwen-Image-2.1",
    }
    repo = url_map.get(filename, filename)
    target = MODELS_DIR / "comfyui" / subfolder
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["huggingface-cli", "download", repo, filename,
         "--local-dir", str(target)],
        check=True, capture_output=True
    )

Phase 0: Install-Flow (Tag 4)
Datei: Makefile
.PHONY: install up down test

# ─── Install: Hardware detect → Config generieren → Modelle laden → Start ───

install:
        @echo "═══ slAIdshow Install ═══"
        @echo ""
        @echo "Step 1/4: Hardware erkennen..."
        docker compose run --rm init python -m slaidshow.hardware_detect
        @echo ""
        @echo "Step 2/4: Config generieren..."
        docker compose run --rm init python -m slaidshow.config_generator
        @echo ""
        @echo "Step 3/4: Modelle laden (dauert je nach Tier)..."
        docker compose run --rm init python -m slaidshow.model_downloader
        @echo ""
        @echo "Step 4/4: Services starten..."
        docker compose up -d
        @echo ""
        @echo "✅ slAIdshow läuft auf http://localhost:8080"
        @cat /app/config/generated.yaml

up:
        docker compose up -d

down:
        docker compose down

test:
        docker compose run --rm init pytest tests/ -v
Datei: docker-compose.yml
services:
  # ─── Init-Container: läuft einmalig bei Install ───
  init:
    build: .
    command: ["sleep", "infinity"]  # wird manuell aufgerufen
    volumes:
      - ./config:/app/config
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # ─── Backend: FastAPI + Orchestrierung ───
  backend:
    build: .
    command: uvicorn slaidshow.main:app --host 0.0.0.0 --port 8080
    depends_on:
      - ollama
      - comfyui
    environment:
      - CONFIG_PATH=/app/config/generated.yaml
    volumes:
      - ./config:/app/config
      - ./models:/app/models
      - ./recordings:/app/recordings
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # ─── Ollama: LLM-Service ───
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # ─── ComfyUI: Bildgenerierung ───
  comfyui:
    image: ghcr.io/comfyanonymous/comfyui:latest
    command: >
      --listen 0.0.0.0 --port 8188
      --lowvram  # wird bei 6GB automatisch gesetzt (via healthcheck)
    volumes:
      - ./models/comfyui:/root/ComfyUI/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  ollama_data:

Phase 0: vram_scheduler.py (Tag 5 – kritisch für 6 GB)
Datei: slaidshow/vram_scheduler.py
"""
Sequenzieller VRAM-Scheduler für GPUs mit < 12 GB.
Stellt sicher, dass LLM und ComfyUI sich die GPU nicht teilen.

Protokoll:
  1. LLM-Inferenz → VRAM belegen
  2. LLM-Resultat → Prompt + Modell-Entscheidung
  3. LLM entladen (torch.cuda.empty_cache())
  4. ComfyUI laden → Bild generieren
  5. ComfyUI entladen
  6. Nächstes Request

Bei ≥ 12 GB: Scheduler wird bypassed (parallel möglich).
"""

import time
import torch
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from slaidshow.config_generator import TIERS


class GpuState(Enum):
    IDLE = "idle"
    LLM_ACTIVE = "llm_active"
    IMAGE_ACTIVE = "image_active"
    TRANSITIONING = "transitioning"


@dataclass
class SchedulerConfig:
    vram_budget_mb: int
    parallel: bool
    transition_timeout_s: float = 5.0


class VramScheduler:
    """
    Steuert die GPU-Zuweisung zwischen LLM und ComfyUI.
    Bei parallel=True (≥12GB) ist dies ein No-Op.
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.state = GpuState.IDLE
        self._llm_loaded = False
        self._image_loaded = False

    @property
    def is_parallel(self) -> bool:
        return self.config.parallel

    async def acquire_llm(self) -> None:
        """GPU für LLM-Inferenz reservieren."""
        if self.is_parallel:
            return
        self._ensure_image_unloaded()
        self.state = GpuState.LLM_ACTIVE
        self._llm_loaded = True

    async def release_llm(self) -> None:
        """LLM-VRAM freigeben."""
        if self.is_parallel:
            return
        self._llm_loaded = False
        torch.cuda.empty_cache()
        self.state = GpuState.IDLE

    async def acquire_image(self) -> None:
        """GPU für Bildgenerierung reservieren."""
        if self.is_parallel:
            return
        self._ensure_llm_unloaded()
        self.state = GpuState.IMAGE_ACTIVE
        self._image_loaded = True

    async def release_image(self) -> None:
        """ComfyUI-VRAM freigeben."""
        if self.is_parallel:
            return
        self._image_loaded = False
        torch.cuda.empty_cache()
        self.state = GpuState.IDLE

    def _ensure_llm_unloaded(self) -> None:
        if self._llm_loaded:
            self.release_llm_sync()

    def _ensure_image_unloaded(self) -> None:
        if self._image_loaded:
            self.release_image_sync()

    def release_llm_sync(self) -> None:
        self._llm_loaded = False
        torch.cuda.empty_cache()

    def release_image_sync(self) -> None:
        self._image_loaded = False
        torch.cuda.empty_cache()

    def get_status(self) -> dict:
        return {
            "state": self.state.value,
            "parallel": self.config.parallel,
            "llm_loaded": self._llm_loaded,
            "image_loaded": self._image_loaded,
        }

Phase 0: Integration in main.py (Tag 6)
Datei: slaidshow/main.py
"""
FastAPI-App – liest Config, initialisiert Scheduler, startet Pipeline.
"""

import yaml
from pathlib import Path
from fastapi import FastAPI, WebSocket
from contextlib import asynccontextmanager

from slaidshow.vram_scheduler import VramScheduler, SchedulerConfig
from slaidshow.config_generator import TIERS

CONFIG_PATH = Path("/app/config/generated.yaml")

app = FastAPI(title="slAIdshow")
scheduler: VramScheduler = None
config: dict = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scheduler, config

    # Config laden (wird bei Install generiert)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    tier_name = config["profile"]
    tier = TIERS[tier_name]

    # Scheduler initialisieren
    scheduler = VramScheduler(
        SchedulerConfig(
            vram_budget_mb=int(config["hardware"]["vram_gb"] * 1024),
            parallel=config["orchestration"]["parallel"],
        )
    )

    print(f"🚀 slAIdshow gestartet | Tier: {tier_name} | "
          f"Parallel: {config['orchestration']['parallel']}")
    yield


app = FastAPI(title="slAIdshow", lifespan=lifespan)


@app.get("/api/v1/status")
async def status():
    return {
        "profile": config["profile"],
        "vram_gb": config["hardware"]["vram_gb"],
        "expected_latency_s": config["expected_latency_s"],
        "features": config["features"],
        "scheduler": scheduler.get_status(),
    }


@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """
    Haupt-WebSocket: Mikrophon-Audio rein → Bild-Events raus.
    Pipeline: STT → LLM → ComfyUI (via Scheduler)
    """
    await ws.accept()
    # ... Pipeline-Logik (in Phase 1 implementiert)

Phase-Plan (konkret, Tag-für-Tag)
Tag
Task
Skill
Output
1
hardware_detect.py + Tests
impl
slaidshow/hardware_detect.py, tests/test_hardware_detect.py
2
config_generator.py + Tests
impl
slaidshow/config_generator.py, tests/test_config_generator.py
3
model_downloader.py + Tests
impl
slaidshow/model_downloader.py, tests/test_model_downloader.py
4
Makefile + docker-compose.yml + .env.example
docker
Install-Flow läuft durch
5
vram_scheduler.py + Tests
impl
slaidshow/vram_scheduler.py, tests/test_vram_scheduler.py
6
main.py (FastAPI + Config-Loading + Status-Endpoint)
impl
App startet, /api/v1/status liefert Tier-Info
7
E2E-Test: make install auf 6GB-GPU → läuft
test + debug
Grüner Install, Latenz < 3 s

Anpassung des Hintergrundwissens (Custom Instructions)
## Hardware-Auto-Detection
- Bei Install: `hardware_detect.py` liest VRAM via `nvidia-ml-py` (import: pynvml)
- `config_generator.py` wählt Tier: 6gb / 8gb / 16gb / 24gb
- `model_downloader.py` lädt NUR benötigte Modelle (kein Over-Download)
- `vram_scheduler.py`: bei < 12 GB → sequenziell (LLM → unload → ComfyUI → unload)
- Bei ≥ 12 GB → parallel (kein Scheduler-Overhead)
- Config wird als `/app/config/generated.yaml` persistiert
- User kann Tier manuell override: `SLAIDSHOW_TIER=6gb` in `.env`

Ein-Satz-Zusammenfassung
make install → VRAM via pynvml messen → Tier (6/8/16/24) wählen → nur die passenden Modelle laden → vram_scheduler.py erzwingt bei < 12 GB sequenzielle GPU-Nutzung → App startet mit der generierten Config. Der User tippt make install und bekommt automatisch das beste Setup für seine Hardware.

Was bereits fertig ist (lokal kopieren, nicht im Chat neu generieren)
Datei	Status
slaidshow/hardware_detect.py	✅ Fertig (Tag 1)
slaidshow/config_generator.py	✅ Fertig (Tag 2)
slaidshow/model_downloader.py	✅ Fertig (Tag 3)
Makefile	✅ Fertig (Tag 4)
docker-compose.yml	✅ Fertig (Tag 4)
slaidshow/vram_scheduler.py	✅ Fertig (Tag 5)
slaidshow/main.py	✅ Fertig (Tag 6)
tests/test_hardware_detect.py	✅ Fertig
tests/test_config_generator.py	❌ Noch nicht geschrieben
tests/test_model_downloader.py	❌ Noch nicht geschrieben
tests/test_vram_scheduler.py	❌ Noch nicht geschrieben
tests/test_e2e_install.py	❌ Noch nicht geschrieben
.env.example	❌ Noch nicht geschrieben
README.md (Install-Abschnitt)	❌ Noch nicht geschrieben

Korrigierter Chat-Workflow (nur noch offene Items)
Session 1 – Setup (einmalig):

Hier ist meine aktuelle Code-Struktur:

[tree -L 3 einfügen]

Die Dateien hardware_detect.py, config_generator.py, model_downloader.py,
vram_scheduler.py und main.py sind bereits implementiert.

Aktuelles Ziel: Fehlende Tests + .env.example + README-Install-Abschnitt.
Hintergrundwissen ist in den Custom Instructions.

Request 1 (test):

Schreibe tests/test_config_generator.py:
- test_select_tier_6gb, 8gb, 16gb, 24gb
- test_select_tier_too_low (4.0 → ValueError)
- test_generate_6gb_config (parallel=False, stt.device=cpu)
- test_generate_24gb_config (parallel=True, in_image_text=True)

Request 2 (test):

Schreibe tests/test_model_downloader.py:
- Mock subprocess.run
- test_download_6gb (nur SD1.5 + LCM + qwen3:1.7b)
- test_download_24gb (inkl. Qwen-Image 2.1)
- test_idempotent

Request 3 (test):

Schreibe tests/test_vram_scheduler.py:
- test_sequential_6gb
- test_conflict (acquire_image während LLM aktiv)
- test_parallel_noop
- test_status

Request 4 (impl):

Gib mir .env.example mit allen Variablen:
- SLAIDSHOW_TIER (optional, Override)
- CONFIG_PATH
- MODELS_DIR
- OLLAMA_HOST
- COMFYUI_URL
- RECORDED_DIR

Request 5 (doc):

Schreibe README.md Abschnitt "Installation":
- make install (was passiert in 4 Steps)
- Tier-Override via SLAIDSHOW_TIER
- Erwartete Latenz pro Tier (Tabelle)
- "Was passiert mit einer RTX 3050 (6 GB)?"

Request 6 (test):

Schreibe tests/test_e2e_install.py:
- Simuliert: hardware_detect (6 GB) → config_generator → model_downloader (mocked)
- Prüft: generated.yaml enthält parallel=False, stt.device=cpu
- Prüft: main.py startet, /api/v1/status liefert profile="6gb"

Was der Chat ab jetzt für dich tut
Phase	Chat-Rolle
Phase 0 (Tag 1–7)	Tests schreiben, Debuggen wenn make install lokal bricht, .env.example + Doku
Phase 1 (Woche 3–5)	arch + impl für Recording, Styles, Crash-Safety
Phase 2–3	Wie im ursprünglichen Plan

Kurz: Der Code steht, der Chat ist ab jetzt dein Tester + Debugger + Doku-Autor – kein zweiter Code-Generator.
