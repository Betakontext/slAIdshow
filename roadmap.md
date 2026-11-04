### slAIdshow — Roadmap zur produktionsreifen Show-Engine

Planungsgrundlage: dokumentierter Projektstand, Commit `4caa19f`, bisherige Roadmap und berichtete native CPU-/Lowend-Einsätze auf ThinkPad X260 und ThinkPad E14.

Produktziel: Sprache wird live in passende Bilder übersetzt — vollständig lokal, mit entferntem ComfyUI oder mit ausdrücklich gewähltem Pollinations-Cloudbackend.

Aktuelle Priorität:

1. Docker- und GPU-Select-Integration abschließen und durch Tests abnehmen.
2. Unmittelbar danach Pollinations einschließlich „Bring your own Pollen“ integrieren.
3. Anschließend Parakeet als zusätzliches lokales STT-Backend einbinden.
4. Danach Pipeline-Stabilität, Recording, Team-Betrieb und Enterprise ausbauen.

### 1. Arbeitsregeln und Status

- **Vorhanden:** Datei oder Implementierung nachgewiesen.
- **Im Einsatz berichtet:** Funktioniert laut praktischer Nutzung.
- **Teilweise:** Integration oder Funktionsumfang noch unvollständig.
- **Offen:** Umsetzung noch nicht nachgewiesen.
- **Umgesetzt:** Vereinbarter Funktionsumfang integriert; keine automatische Aussage über Testabnahme.
- **Abgenommen:** Docker/GPU-Schritt anhand dokumentierter Test- und Laufnachweise bestätigt.

Eine Datei im Repository und ein erfolgreicher Commit belegen keine vollständige Laufzeitintegration.

#### 1.1 Vorläufige Testregel

Eine verpflichtende testbasierte Abnahme gilt aktuell ausschließlich für M0: Docker- und GPU-Select-Integration.

Für nachfolgende Arbeiten:

- Tests und manuelle Prüfungen können bei Bedarf ergänzt werden.
- Sie sind vorerst keine verpflichtenden Meilenstein- oder Release-Gates.
- Fehlende Testnachweise werden transparent benannt.
- Ungeprüfte Funktionen werden nicht als getestet oder abgenommen bezeichnet.
- Verifizierung externer Schnittstellen, sichere Zugangsdatenverwaltung und ausdrückliche Cloud-Freigabe bleiben erforderlich.

#### 1.2 Technische Leitplanken

- Python 3.12+, Type Hints und ruff.
- FastAPI, Ollama, ComfyUI und Docker Compose.
- Native Installation bleibt ein gleichwertiger Startpfad.
- Bestehende Whisper-/Faster-Whisper- und Lowend-Funktionen erhalten.
- SSE für bestehende Status-, Transkriptions- und Bildereignisse erhalten.
- Keine Cloud-Abhängigkeit im vollständig lokalen Runtime-Pfad.
- Pollinations nur nach ausdrücklicher Auswahl.
- pynvml zur NVIDIA-Erkennung.
- JSON für Hardwaredaten, YAML für Konfiguration.
- Eine gemeinsame Konfigurationsquelle.
- Neue Abhängigkeiten erst nach ausdrücklicher technischer Entscheidung.

### 2. Dokumentierter Ausgangspunkt

#### 2.1 Bestehende Anwendung

Vorhanden beziehungsweise im Einsatz berichtet:

- Anwendungseinstieg `app.py`.
- Mikrofon, VAD und Transkription.
- Whisper/Faster-Whisper.
- Ollama-Prompt-Erzeugung.
- Lokale und entfernte ComfyUI-Bildgenerierung.
- Pollinations als alternatives Bildbackend.
- Galerie, Vollbild, Stilsteuerung und SSE.
- Nativer CPU-/Lowend-Betrieb auf ThinkPad X260 und ThinkPad E14.

Teilweise integriert:

- Hardware-Erkennung und GPU-Auswahl.
- Gemeinsame Konfiguration.
- Modell-Downloader.
- VRAM-Scheduler.
- Docker-Startpfad.

#### 2.2 Commit `4caa19f`

Commit: `Added Docker files and GPU selection`, Branch `1.0_vram-selection`.

Durch Commit-Ausgabe nachgewiesen:

- `Dockerfile` aufgenommen.
- `config_generator.py` aufgenommen.
- `runtime_config.py` aufgenommen.
- Tests für Hardware-Erkennung und Konfigurationsgenerator aufgenommen.
- Zusätzliche ComfyUI-Workflows aufgenommen.
- Weitere Änderungen an der Anwendung versioniert.

Docker und GPU-Auswahl sind implementierter Projektstand, aber noch kein vollständig abgenommener Betriebsweg.

#### 2.3 Bekannter Laufzeitbefund

Der zuletzt gezeigte Health-Aufruf lieferte HTTP 200 mit:

- `ollama_ok:false`
- `image_backend:"comfyui"`
- `allow_cloud:true`
- vorhandenem Ausgabeordner

Daraus folgt:

- HTTP-Anwendung erreichbar.
- Ollama-Bereitschaft in diesem Lauf nicht bestätigt.
- Vollständige Bildpipeline nicht nachgewiesen.
- Cloud-Freigabe muss zur ausdrücklich gewählten Betriebsart passen.
- `allow_cloud:true` allein beweist keinen tatsächlich ausgeführten Cloud-Aufruf.

### 3. Unterstützte Betriebsarten

#### 3.1 Vollständig lokal

- Anwendung, STT, Ollama und ComfyUI auf demselben Rechner.
- Nach Modellbereitstellung ohne Internet nutzbar.
- Gemeinsame GPU-Ressourcen koordiniert verwenden.
- Keine stillen Cloud-Fallbacks.

#### 3.2 CPU-/Lowend mit Remote-ComfyUI

- Anwendung, Whisper/Faster-Whisper und Ollama auf dem ThinkPad.
- Bilderzeugung durch ComfyUI auf einem anderen Rechner.
- Keine lokale NVIDIA-GPU erforderlich.
- Bestehende Lowend-Option erhalten.
- LAN und internetabhängige Tunnel unterscheiden.
- Offline-fähig nur bei einer ohne Internet nutzbaren Verbindung.
- Remote-Unloads nur bei ausdrücklich vereinbarter Verwaltungshoheit.

#### 3.3 Pollinations

- Optionales Bildbackend mit ausdrücklicher Cloud-Freigabe.
- Eigene Zugangsdaten und eigene Abrechnung nach dem verifizierten Pollinations-Verfahren.
- Kein automatischer kostenpflichtiger Backendwechsel.
- Kein Offline-Versprechen.
- Lokale STT-/LLM-Verarbeitung bleibt möglich.

### 4. Architektur und Ressourcen

#### 4.1 Bestehende Anwendung integrieren

- `app.py` bleibt Einstiegspunkt.
- Docker, Makefile und native Startskripte darauf ausrichten.
- Paketmigration und Frontend-Neuentwicklung sind keine Voraussetzung.
- SSE erhalten; WebSocket nur bei konkretem Bedarf ergänzen.

#### 4.2 Gemeinsame Konfiguration

Containerpfade:

- `/app/config/hardware.json`
- `/app/config/generated.yaml`
- `/app/models`
- `/app/outputs`
- `/app/recordings`

Native Pfade separat berücksichtigen. Bestehende Verwendung von `CONFIG_PATH` mit dem Code abgleichen.

Getrennt konfigurieren:

- Native Installation oder Docker.
- CPU-/Lowend oder lokaler GPU-Betrieb.
- STT-Backend, Modell und Gerät.
- Ollama-Endpunkt und Modell.
- Bildbackend und lokaler oder entfernter Endpunkt.
- Cloud-Freigabe.
- GPU-Auswahl und Ressourcenprofil.

Anwendung, Downloader und Scheduler verwenden dieselbe effektive Konfiguration.

Secrets gehören nicht in Hardwareberichte, öffentlich sichtbare Konfiguration oder Statusantworten.

#### 4.3 Hardwareprofile

- **CPU-/Lowend:** kein lokaler Mindest-VRAM; bestehendes Whisper/Faster-Whisper und kleines Ollama-Modell; Remote-ComfyUI oder Pollinations.
- **6gb:** 6 bis unter 8 GiB; CPU-STT, kleines quantisiertes LLM, SD1.5+LCM bei 512 px, sequenziell.
- **8gb:** 8 bis unter 16 GiB; sichere Basis, Erweiterungen nach Ressourcenprüfung.
- **16gb:** 16 bis unter 24 GiB; größere kompatible Modelle.
- **24gb:** ab 24 GiB; Qualitätsoptionen und kontrollierter Parallelbetrieb.

Regeln:

- Unter 6 GiB kein automatisches lokales GPU-Bildprofil freigeben.
- CPU-/Remote und Pollinations bleiben nutzbar.
- VRAM mehrerer GPUs nicht addieren.
- Gesamt-VRAM bestimmt Profilfähigkeit, freier VRAM aktuelle Ausführbarkeit.
- `SLAIDSHOW_TIER=6gb` erlaubt ein kleineres Profil.
- Zu große Overrides verständlich ablehnen.
- Parallelbetrieb auf gemeinsamer GPU nur mit passendem Gesamtbudget.
- Keine pauschale No-Op-Regel ab einer festen VRAM-Grenze.
- Lokale CPU-Arbeit und Remote-ComfyUI nicht aufgrund lokaler VRAM-Werte global serialisieren.
- Keine ungeprüften Latenzversprechen.

### 5. M0 — Docker und GPU-Select abschließen

**Aktueller Arbeitsschritt und einziger vorläufig verpflichtend testbasiert abzunehmender Meilenstein.**

Ziel: Docker und gemeinsame Hardware-Konfiguration funktionieren, ohne native Betriebsarten zu beschädigen.

#### 5.1 Repository und Startpfade

- [ ] Repository-Stand mit Commit `4caa19f` abgleichen.
- [ ] Bereits getrackte Modelle, Outputs, Caches und Secrets prüfen.
- [ ] `.gitignore` und `.dockerignore` getrennt prüfen.
- [ ] `app.py`, Imports und Einstiegspunkte konsolidieren.
- [ ] Dockerfile und Compose auf tatsächliche Anwendung abstimmen.
- [ ] Makefile, `run.sh` und `run.ps1` konsistent halten.
- [ ] Native CPU-/Lowend-Startpfade erhalten.
- [ ] Tatsächlich verwendete Abhängigkeiten festhalten.

#### 5.2 Hardware und Konfiguration

- [ ] `hardware_detect.py`, `config_generator.py` und `runtime_config.py` abgleichen.
- [ ] Fehlende NVML, Treiberfehler und fehlende NVIDIA-GPU kontrolliert behandeln.
- [ ] GPU-Auswahl auf Host und im Container eindeutig zuordnen.
- [ ] Tier-Grenzen und Overrides validieren.
- [ ] Persistierte Nutzerauswahl respektieren.
- [ ] Gemeinsame Konfiguration in Anwendung und Downloader verwenden.
- [ ] CPU-/Remote-Modus ohne lokale GPU initialisieren.
- [ ] Effektive Konfiguration ohne Secrets anzeigen.

#### 5.3 Docker-Laufzeit

- [ ] Ursache für `ollama_ok:false` untersuchen.
- [ ] Container-, Host- und Remote-Endpunkte korrekt zuordnen.
- [ ] HTTP-Liveness und Backend-Readiness unterscheiden.
- [ ] Mikrofonzugriff und Geräteberechtigungen prüfen.
- [ ] Konfiguration, Modelle und Ausgaben persistent speichern.
- [ ] Gewählte GPU tatsächlich den vorgesehenen Diensten zuweisen.
- [ ] Für dieselbe lokale GPU eine sichere sequenzielle Ausführung sicherstellen.
- [ ] Frontend-Abhängigkeiten für lokalen Offline-Betrieb berücksichtigen.

#### 5.4 Verpflichtende Abnahme

- [ ] Relevante pytest-Tests und ruff erfolgreich ausführen.
- [ ] Fehlende GPU/NVML und gültigen CPU-/Remote-Betrieb mit Mocks prüfen.
- [ ] Profilgrenzen, GPU-Auswahl und ungültige Overrides prüfen.
- [ ] Docker-Build und dokumentierten Containerstart ausführen.
- [ ] Native Startpfade auf Regressionen prüfen.
- [ ] Persistenz und Konfigurationsnutzung nach Neustart bestätigen.
- [ ] Sichere 6-GB-Konfiguration auf passender Referenzhardware bestätigen.
- [ ] Ergebnisse, Umgebung und verbleibende Grenzen dokumentieren.

Nach Abschluss folgt unmittelbar M1. Recording, große Modelle und weiterführende Scheduler-Optimierungen werden nicht vorgezogen.

### 6. M1 — Pollinations und „Bring your own Pollen“

**Nächster priorisierter Implementierungspunkt nach M0.**

Ziel: Bestehendes Pollinations-Backend erhalten und um eine sichere Nutzung eigener Zugangsdaten und eigener Abrechnung ergänzen.

Todos:

- [ ] Offizielle Dokumentation zu API, Authentifizierung, Pollen und Nutzungsbedingungen prüfen.
- [ ] Bestehenden Backendpfad mit dem tatsächlichen API-Vertrag abgleichen.
- [ ] Pollinations als `image_backend` ausdrücklich auswählbar halten.
- [ ] Eigene Zugangsdaten nach dem verifizierten Verfahren serverseitig konfigurierbar machen.
- [ ] „Bring your own Pollen“ in Oberfläche und Dokumentation erklären.
- [ ] Cloud-Freigabe und Backend-Auswahl getrennt validieren.
- [ ] Fehlende oder ungültige Zugangsdaten verständlich melden.
- [ ] Guthaben-/Abrechnungsfehler, Rate Limits und Timeouts behandeln.
- [ ] Keine Zugangsdaten in Browserantworten, URLs, Logs oder Git ausgeben.
- [ ] Übertragene Prompts, Parameter und mögliche Referenzbilder transparent ausweisen.
- [ ] Keine lokalen Bildgewichte für Pollinations herunterladen.
- [ ] Pollinations vom lokalen Bild-VRAM-Scheduling ausnehmen.
- [ ] Retry-Verhalten auf mögliche Doppelaufträge und Mehrkosten abstimmen.
- [ ] Keine automatischen kostenpflichtigen Backendwechsel ausführen.
- [ ] Nutzungs- und Budgetanzeigen nur bei tatsächlich unterstützter API anbieten.

Vorläufiger Abschluss:

- Backend-Auswahl, serverseitige Zugangsdatenverwaltung und Fehlerpfade sind integriert.
- Dokumentation entspricht der implementierten Schnittstelle.
- Tatsächliche Prüfungen und noch ungeprüfte Pfade sind benannt.
- Keine verpflichtende Testabnahme als Voraussetzung für M2.
- Reale kostenpflichtige Requests nur nach ausdrücklicher Freigabe.

### 7. M2 — Parakeet als zusätzliches STT-Backend

**Unmittelbar nach der Pollinations-Integration priorisiert; nicht mehr zurückgestellt.**

Ziel: Parakeet TDT 0.6B v3 als zusätzliches lokales STT-Backend einbinden, ohne Whisper/Faster-Whisper zu entfernen.

Todos:

- [ ] Modellquelle, Lizenz, unterstützte Sprachen und Laufzeitvoraussetzungen verifizieren.
- [ ] Tatsächlich kompatiblen Inferenzpfad ermitteln.
- [ ] Parakeet nicht ungeprüft als Faster-Whisper-kompatibel behandeln.
- [ ] Notwendige neue Abhängigkeiten vor Umsetzung ausdrücklich entscheiden.
- [ ] Bestehende STT-Schnittstelle in `transcription_backend.py` erhalten.
- [ ] Backend-Auswahl in gemeinsamer Konfiguration abbilden.
- [ ] Modellbereitstellung in den Downloader integrieren.
- [ ] Parakeet nur bei entsprechender Auswahl herunterladen.
- [ ] Modelle, Cache und Ladeverhalten für native Installation und Docker berücksichtigen.
- [ ] Audioformat, Samplingrate, VAD-Anbindung und Transkriptionsausgabe abstimmen.
- [ ] Ressourcenbedarf für CPU und GPU ermitteln und transparent dokumentieren.
- [ ] GPU-STT in das lokale VRAM-Budget aufnehmen.
- [ ] Auf 6 GB GPU-STT nicht ungeprüft zusätzlich zu LLM und Bildmodell aktivieren.
- [ ] Falls Parakeet im CPU-/Lowend-Profil ungeeignet ist, Whisper/Faster-Whisper als ressourcenschonende Variante erhalten.
- [ ] Fehlende Modelle, inkompatible Laufzeit und Initialisierungsfehler verständlich melden.
- [ ] Keine Cloud-STT als stillen Fallback ergänzen.

Vorläufiger Abschluss:

- Parakeet ist über die gemeinsame STT-Schnittstelle auswählbar.
- Bestehende Whisper-/Faster-Whisper-Betriebsarten bleiben erhalten.
- Unterstützte Geräte und Ressourcenbeschränkungen sind dokumentiert.
- Ungeprüfte Hardwarepfade sind ausdrücklich benannt.
- Keine verpflichtende Testabnahme als Voraussetzung für Folgearbeiten.

### 8. M3 — Pipeline und Installation weiter stabilisieren

Folgt auf M1 und M2.

Todos:

- [ ] Scheduler und gemeinsame GPU-Sperren vervollständigen.
- [ ] Ollama und lokales ComfyUI über unterstützte Schnittstellen entladen.
- [ ] Speicherfreigabe mit Timeout bestätigen.
- [ ] Abbruch, Shutdown und Ausfälle mit Cleanup behandeln.
- [ ] Remote-Unload nur bei entsprechender Verwaltungshoheit zulassen.
- [ ] Fehlgeschlagene lokale Entladung vor unsicherer Folgearbeit behandeln.
- [ ] Native Installation auf X260 und E14 dokumentieren.
- [ ] Modellquellen und Lizenzen dokumentieren.
- [ ] Unvollständige Downloads erkennen.
- [ ] Prüfsummen oder gepinnte Revisionen verwenden, soweit verfügbar.
- [ ] Nur für die gewählte Betriebsart benötigte Modelle bereitstellen.
- [ ] Installations- und Update-Anleitung vervollständigen.

Prüfungen können bedarfsbezogen ergänzt werden; vorerst keine verpflichtende Testabnahme.

### 9. M4 — Demo-Release und Show-Stabilität

Todos:

- [ ] Begrenzte Queue und Überlaststrategie implementieren.
- [ ] Veraltete Aufträge kontrolliert verwerfen.
- [ ] Fehlerhafte Bilder nicht anzeigen.
- [ ] Letztes gültiges Bild bei Backendfehlern erhalten.
- [ ] Echten Crossfade ergänzen.
- [ ] SSE-Wiederverbindung und Zustandsabgleich verbessern.
- [ ] Workflow-Presets und Style-Grenzen dokumentieren.
- [ ] Lokale Titel und Text-Overlays ergänzen.
- [ ] STT-, LLM-, Netzwerk-, Lade-, Render- und Gesamtzeit erfassen.
- [ ] Demo für lokale GPU, CPU-/Remote und Pollinations vorbereiten.

Optionale spätere Qualitätsnachweise:

- Längere Live-Läufe auf 6-GB-Hardware und ThinkPads.
- Messung von Latenzen, Fehlerrate und Ressourcenverbrauch.
- Kontrollierte Ausfallversuche.

Die bisherigen 0,8 bis 2,5 Sekunden sind keine bestätigte Produkteigenschaft.

### 10. M5 — Recording und Export

- [ ] Recording-Verfahren und Abhängigkeiten gesondert entscheiden.
- [ ] Start, Stopp und Aufzeichnungsstatus integrieren.
- [ ] Bildwechsel, Overlays und optionales Audio synchron erfassen.
- [ ] MP4-/WebM-Ausgabe gemäß gewähltem Verfahren unterstützen.
- [ ] Speicherplatzmangel und Abbruch behandeln.
- [ ] Lokale Ablage und Löschfunktion anbieten.
- [ ] Ressourcenschonende Variante für CPU-/Lowend und 6 GB vorsehen.
- [ ] Keine zusätzlichen Cloud-Aufrufe ohne Freigabe erzeugen.

### 11. M6 — Moderation und Team-Betrieb

Reihenfolge:

1. Separate Moderator- und Display-Ansicht.
2. Skript-Modus mit manuellen und automatischen Triggern.
3. Multi-Speaker mit eindeutiger Quellenzuordnung.
4. Versionierte externe API und Ereignisverträge.
5. Optionale Publikumsinteraktion.

Todos:

- [ ] Sitzungszustände zwischen Clients abstimmen.
- [ ] Sprecherwechsel nachvollziehbar machen.
- [ ] Queue und Scheduler zentral durchsetzen.
- [ ] Zugangsdaten und Nutzung eindeutig zuordnen.
- [ ] Wiederverbindung nach Netzwerkunterbrechungen ermöglichen.

### 12. M7 — Enterprise und Pilotbetrieb

- [ ] Sitzungs- und Benutzerisolierung.
- [ ] Rollen, Berechtigungen und Authentifizierung.
- [ ] SSO nach Architekturentscheidung.
- [ ] Audit-Log und Aufbewahrung.
- [ ] Verwaltete Modell-, Workflow- und Backend-Freigaben.
- [ ] Administrativ erzwungener lokaler Modus.
- [ ] Sichere Verwaltung und Rotation von Provider-Zugangsdaten.
- [ ] On-Premise-Deployment und Updates.
- [ ] Backup und Wiederherstellung.
- [ ] Air-Gapped-Anleitung.
- [ ] Pilotbetrieb für Bildungs- und kreative Live-Anwendungen.

Lokaler Betrieb allein ist keine Datenschutz-Zertifizierung. Compliance- und Air-Gapped-Aussagen benötigen eigene Nachweise.

### 13. Zurückgestellte Erweiterungen

- Große LLMs und FLUX-basierte Qualitätsprofile.
- Automatische Wahl mehrerer Bildmodelle.
- Generierter Text direkt im Bild.
- Pauschaler Wechsel zu WebSocket.
- Vollständige Frontend-Neuentwicklung.
- Native Desktop-App.
- Apple-Silicon-Unterstützung.
- Workflow-Marktplatz und Lizenzierung.

Pollinations und Parakeet sind ausdrücklich nicht zurückgestellt.

### 14. Vorläufige Abschlussregeln

Für M0 gilt die verpflichtende Testabnahme aus Abschnitt 5.4.

Für M1 und folgende gilt vorerst:

- Vereinbarter Umfang integriert.
- Bestehende Betriebsarten bewusst berücksichtigt.
- Konfiguration und Dokumentation angepasst.
- Bekannte Grenzen und offene Punkte benannt.
- Keine erfundenen Test- oder Hardwareergebnisse.
- Keine unbeabsichtigten Cloud-Aufrufe oder Kosten.
- Keine verpflichtende Testabnahme vor dem nächsten Arbeitsschritt.

Eine spätere Verschärfung der Qualitäts- und Release-Gates wird separat entschieden.

### 15. Unmittelbar nächste Aufgaben

1. Dockerfile, Compose und Laufzeitkonfiguration gegen native Startpfade abgleichen.
2. Ollama-Erreichbarkeit, GPU-Zuordnung und Konfigurationsnutzung schließen.
3. Docker/GPU-Schritt durch die vorgesehenen Tests und Laufnachweise abnehmen.
4. Pollinations einschließlich „Bring your own Pollen“ integrieren.
5. Parakeet als zusätzliches lokales STT-Backend einbinden.
6. Anschließend Pipeline-Stabilität und weitere Produktfunktionen ausbauen.

Release-Reihenfolge:

Docker/GPU integrieren und testbasiert abnehmen → Pollinations/BYOP → Parakeet → Pipeline und Installation stabilisieren → Live-Demo → Recording → Team → Enterprise.
