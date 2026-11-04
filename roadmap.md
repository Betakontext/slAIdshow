### slAIdshow — Roadmap zur produktionsreifen Show-Engine

Planungsgrundlage: bisherige Roadmap, dokumentierter Projektstand, Commit `4caa19f` sowie die berichteten erfolgreichen nativen CPU-/Lowend-Einsätze auf ThinkPad X260 und ThinkPad E14.

Produktziel: Sprache wird live in passende Bilder übersetzt — wahlweise vollständig lokal, mit entferntem ComfyUI-Rechner oder mit ausdrücklich gewähltem Pollinations-Cloudbackend.

Priorität: Bestehende Betriebsarten erhalten, gemeinsame Konfiguration integrieren und Installation sowie Laufzeit stabilisieren, bevor große Modelle, Recording oder Enterprise-Funktionen hinzukommen.

### 1. Arbeitsregeln und Status

- **Vorhanden:** Implementierung oder Datei nachgewiesen; vollständige Integration nicht automatisch bestätigt.
- **Im Einsatz berichtet:** Funktioniert laut praktischer Nutzung; reproduzierbare Testbedingungen und Abnahme werden ergänzt.
- **Teilweise:** Bausteine vorhanden, Integration oder Funktionsnachweis unvollständig.
- **Offen:** Keine vollständige Implementierung nachgewiesen.
- **Abgenommen:** Akzeptanzkriterien erfüllt und Testergebnis dokumentiert.

Eine Datei im Repository oder ein erfolgreicher Commit ist kein abgeschlossener Meilenstein.

Technische Leitplanken:

- Python 3.12+, Type Hints und ruff.
- FastAPI, Ollama, ComfyUI und Docker Compose.
- Native Installation bleibt ein gleichwertiger Startpfad.
- Bestehende Whisper-/Faster-Whisper- und Lowend-Funktionen erhalten.
- Keine Cloud-Abhängigkeiten im vollständig lokalen Runtime-Pfad.
- Cloud-Nutzung bleibt als explizite Option erhalten.
- pynvml zur NVIDIA-Erkennung, JSON für Hardwaredaten, YAML für Konfiguration.
- pytest mit Mocks für GPU, Ollama, ComfyUI und Pollinations.
- Eine gemeinsame Konfigurationsquelle statt widersprüchlicher Defaults.
- Neue Abhängigkeiten erst nach ausdrücklicher technischer Entscheidung.

### 2. Dokumentierter Ausgangspunkt

#### 2.1 Bestehende Anwendung

- FastAPI-Anwendung in `app.py`: vorhanden.
- Mikrofon, VAD und Transkription: vorhanden.
- Whisper-/Faster-Whisper-Backends: vorhanden; erhalten und absichern.
- Ollama-Prompt-Erzeugung: vorhanden.
- ComfyUI-Bildgenerierung: vorhanden.
- Pollinations als alternatives Bildbackend: bestehende Option; erhalten und systematisch testen.
- Galerie, Vollbild und Stilsteuerung: vorhanden.
- Live-Ereignisse über SSE: vorhanden; erhalten.
- Modell-Downloader und VRAM-Scheduler: teilweise; Integration und Fehlerfälle prüfen.

#### 2.2 Nativer CPU-/Lowend-Betrieb

Laut bisheriger Nutzung funktioniert slAIdshow außerhalb von Docker auf ThinkPad X260 und ThinkPad E14 mit:

- CPU-Modus und Lowend-Option.
- Whisper oder Faster-Whisper für lokale Transkription.
- Ollama auf dem ThinkPad.
- Bildgenerierung durch ComfyUI auf einem anderen Rechner über eine Remote-Verbindung.

Dieser Betriebsweg ist ausdrücklich Bestandteil des Produkts und kein zu entfernender Übergangszustand.

Noch zu dokumentieren:

- Konkrete CPU-/RAM-Ausstattung und Betriebssysteme.
- Verwendete STT- und Ollama-Modelle.
- Lowend-Einstellungen, Verbindungsart und ComfyUI-Workflows.
- Gemessene Latenzen und bekannte Grenzen.
- Reproduzierbare Start- und Prüfschritte.

#### 2.3 Commit `4caa19f`

Commit: `Added Docker files and GPU selection` auf Branch `1.0_vram-selection`.

Durch Commit-Ausgabe nachgewiesen:

- `Dockerfile` neu aufgenommen.
- `config_generator.py` neu aufgenommen.
- `runtime_config.py` neu aufgenommen.
- `tests/test_config_generator.py` neu aufgenommen.
- `tests/test_hardware_detect.py` neu aufgenommen.
- Zusätzliche ComfyUI-Workflow-Dateien aufgenommen.
- Weitere Änderungen an der bestehenden Anwendung versioniert.

GPU-Auswahl und Docker-Unterstützung sind damit als aktueller Implementierungsstand festgehalten. Vollständigkeit, Container-Gerätezuordnung und tatsächliche Nutzung der gewählten GPU müssen noch getestet werden.

Die aktualisierte `.gitignore` ist für den Repository-Stand zu prüfen. Modelle, Outputs und Secrets dürfen nicht versehentlich versioniert sein. Bereits getrackte Dateien werden durch Ignore-Regeln nicht automatisch entfernt.

#### 2.4 Bekannter Laufzeitbefund

Der zuletzt gezeigte Health-Aufruf unter `127.0.0.1:8080` liefert HTTP 200 mit:

- `ollama_ok:false`
- `image_backend:"comfyui"`
- `allow_cloud:true`
- vorhandenem Ausgabeordner

Bewertung:

- Die HTTP-Anwendung ist erreichbar.
- Ollama-Erreichbarkeit ist in diesem Lauf nicht bestätigt.
- Eine vollständige Bildpipeline ist damit nicht nachgewiesen.
- `allow_cloud:true` ist nicht grundsätzlich ein Fehler: Die Einstellung muss zum ausdrücklich gewählten Betriebsmodus passen.
- Im vollständig lokalen Modus müssen Cloud-Aufrufe ausgeschlossen sein.
- Bei ausdrücklich gewähltem Pollinations-Backend muss Cloud-Nutzung erlaubt und sichtbar sein.

### 3. Unterstützte Betriebsarten

#### 3.1 Vollständig lokal mit NVIDIA-GPU

- STT, Ollama und ComfyUI auf demselben Rechner.
- Hardwareprofile ab 6 GiB VRAM für lokale GPU-Bildgenerierung.
- Nach Installation und Modellbereitstellung ohne Internet betreibbar.
- Native Installation und Docker als getrennt zu prüfende Startpfade.

#### 3.2 CPU-/Lowend-Client mit entferntem ComfyUI

- Mikrofon, STT, Anwendung und Ollama auf dem Client.
- Keine lokale NVIDIA-GPU erforderlich.
- ComfyUI übernimmt die Bildgenerierung auf einem anderen Rechner.
- Bestehende Remote-Verbindung erhalten und dokumentieren.
- LAN-Verbindung und internetabhängige Tunnel unterscheiden.
- Offline-fähig nur, wenn auch die Verbindung zum Bildrechner ohne Internet funktioniert.
- Remote-Modellverwaltung darf andere Nutzer des Bildservers nicht unbeabsichtigt beeinflussen.

#### 3.3 Pollinations als optionales Cloudbackend

- Auswahl über `image_backend`.
- Ausdrückliches Opt-in statt automatischem Cloud-Fallback.
- Nutzer verwenden eigene Pollinations-Zugangsdaten und eigenes Pollen-Guthaben im Sinne der gewünschten „Bring your own Pollen“-Strategie.
- Konkretes Authentifizierungsverfahren, API-Vertrag und Nutzungsbedingungen vor Implementierung anhand offizieller Dokumentation prüfen.
- Internet erforderlich; keine Offline-Zusage für diese Betriebsart.
- Lokale STT-/LLM-Verarbeitung kann erhalten bleiben; der konkrete ausgehende Bildrequest muss transparent sein.

### 4. Verbindliche Architekturentscheidungen

#### 4.1 Bestehende Anwendung zuerst integrieren

`app.py` bleibt zunächst der Anwendungseinstieg.

Docker Compose, Makefile und native Startskripte werden daran ausgerichtet.

Eine spätere Paketmigration erfolgt separat mit Regressionstests. Docker-Einführung und Paketmigration werden nicht gleichzeitig umgebaut.

#### 4.2 Eine gemeinsame Konfigurationsquelle

Standardpfade im Container:

- `/app/config/hardware.json`
- `/app/config/generated.yaml`
- `/app/models`
- `/app/outputs`
- `/app/recordings`

`CONFIG_PATH` ermöglicht abweichende Pfade außerhalb von Docker; bestehende Implementierung und Dokumentation sind abzugleichen.

Konfigurationsreihenfolge:

1. Verfügbare lokale Hardware erfassen.
2. Bestehende Nutzerauswahl und Betriebsart berücksichtigen.
3. Passendes lokales Ressourcenprofil bestimmen.
4. Explizite Overrides prüfen.
5. Konfiguration validieren und atomar speichern.
6. Dieselbe effektive Konfiguration für Downloader und Laufzeit verwenden.

Getrennt abbilden:

- Native Installation oder Docker.
- CPU-/Lowend- oder lokaler GPU-Betrieb.
- STT-Backend und STT-Gerät.
- Ollama-Endpunkt und Modell.
- Bildbackend und lokaler oder entfernter Endpunkt.
- Cloud-Freigabe.
- Lokale GPU-Auswahl und VRAM-Profil.

Secrets gehören weder in Hardwareberichte noch in öffentlich sichtbare Konfiguration oder Statusantworten.

Fehlende NVML oder eine fehlende NVIDIA-GPU dürfen einen gültigen CPU-/Remote- oder Pollinations-Betrieb nicht verhindern.

#### 4.3 Sichere und gerätebezogene VRAM-Nutzung

Parallelbetrieb nur freigeben, wenn Modelle, Kontextspeicher, STT, Bildauflösung und Sicherheitsreserve gemeinsam in das Budget der jeweiligen GPU passen.

Bis zur Abnahme gilt bei gemeinsam genutzter lokaler GPU sequenzieller Modellbetrieb.

Keine pauschale globale Sequenzialisierung unabhängiger Rechner:

- Lokale CPU-Arbeit und entferntes ComfyUI teilen kein lokales VRAM-Budget.
- Für Remote-ComfyUI keine lokalen NVML-Werte als Remote-Messung ausgeben.
- Remote-Unloads nur bei ausdrücklich vereinbarter Verwaltungshoheit.
- Pollinations benötigt keine lokale Bildmodell-Entladung.

#### 4.4 Bestehendes SSE bleibt erhalten

SSE bleibt der Kanal für Status, Transkription und Bildereignisse.

WebSocket nur ergänzen, wenn bidirektionales Audio oder eine konkrete neue Funktion es benötigt.

#### 4.5 Modelle werden verifiziert

Modellnamen, Quantisierungen, Quellen, Lizenzen und Backend-Kompatibilität vor Integration prüfen.

Parakeet bleibt ein eigener Integrationspunkt. Bestehendes Whisper/Faster-Whisper bleibt bis zur erfolgreichen Abnahme einer Alternative verfügbar.

#### 4.6 Cloud nur nach bewusster Auswahl

- Vollständig lokaler Modus: keine Cloud-Aufrufe.
- Remote-ComfyUI: nur der konfigurierte Bildserver; kein stiller Wechsel zu Pollinations.
- Pollinations: ausdrückliche Cloud-Freigabe und gültige eigene Zugangsdaten erforderlich.
- Fehler eines Backends lösen keinen kostenpflichtigen Backendwechsel aus.
- Cloud-Modus und ausgehende Datenarten in Oberfläche und Dokumentation sichtbar machen.
- Pollinations-Zugangsdaten ausschließlich serverseitig verwenden.
- Keine fremden Tokens in URLs, Frontend-Code, Logs oder Git aufnehmen.

### 5. Ressourcenprofile und Hardwaregrenzen

#### 5.1 CPU-/Lowend-Profil

Eigenständiger unterstützter Betriebsweg ohne Mindestanforderung von 6 GiB lokalem VRAM.

- Whisper/Faster-Whisper auf CPU.
- Kleines, auf der jeweiligen Hardware geprüftes Ollama-Modell.
- Bestehende Lowend-Option erhalten und dokumentieren.
- Bildgenerierung durch Remote-ComfyUI oder ausdrücklich gewähltes Pollinations.
- Lokale CPU-Bildgenerierung ist kein zugesichertes Echtzeitprofil.
- Keine pauschale Latenzzusage.

#### 5.2 Lokale NVIDIA-Profile

- **6gb:** 6 bis unter 8 GiB; CPU-STT, kleines quantisiertes LLM, SD1.5+LCM bei 512 px, sequenziell.
- **8gb:** 8 bis unter 16 GiB; zunächst sichere Basis, größere Modelle und Auflösungen erst nach Messung.
- **16gb:** 16 bis unter 24 GiB; geprüfte größere Modelle, standardmäßig sequenziell.
- **24gb:** ab 24 GiB; Qualitätsoptionen und kontrollierter Parallelbetrieb nach Budgetprüfung.

Zusätzliche Regeln:

- Unter 6 GiB keine automatische Freigabe des lokalen GPU-Bildprofils.
- CPU-/Remote- und Cloud-Profile bleiben trotzdem nutzbar.
- Bei mehreren GPUs eine GPU ausdrücklich auswählen; VRAM nicht addieren.
- Gesamt-VRAM bestimmt Profilfähigkeit, freier VRAM die aktuelle Ausführbarkeit.
- Desktopbelegung und Spitzen durch gemessene Reserve berücksichtigen.
- `SLAIDSHOW_TIER=6gb` ermöglicht ein kleineres lokales GPU-Profil.
- Zu große Overrides verständlich ablehnen.
- GPU-Zuordnung zwischen Host, Container und Modellservern dokumentieren.
- Apple Silicon bleibt außerhalb der ersten NVIDIA-/Docker-Abnahme.

Vollbild, Galerie, Stilsteuerung und geplante Overlays bleiben auch für CPU-/Lowend- und 6-GB-Betrieb vorgesehen.

### 6. M0 — Bestand und Startfähigkeit konsolidieren

Ziel: Eindeutiger Ausgangsstand ohne Regression bestehender Betriebsarten.

Todos:

- [ ] Repository mit dokumentiertem Dateibaum und Commit `4caa19f` abgleichen.
- [ ] Bereits getrackte Modelle, Outputs, Caches und Secrets ausschließen beziehungsweise bereinigen.
- [ ] `app.py`, Imports und Modulpfade konsolidieren.
- [ ] Vorhandenen Dockerfile-Build prüfen.
- [ ] Compose, Makefile, `run.sh` und `run.ps1` konsistent ausrichten.
- [ ] Native CPU-/Lowend-Startpfade unverändert funktionsfähig halten.
- [ ] Tatsächlich verwendete Abhängigkeiten festhalten.
- [ ] API-Routen, Health-Semantik und SSE-Ereignisse dokumentieren.
- [ ] Smoke-Tests für App-Start, Health und Status ergänzen.
- [ ] Ursache für `ollama_ok:false` im zuletzt gezeigten Lauf untersuchen.

Abnahme:

- Native Anwendung und Container starten über dokumentierte Einstiegspunkte.
- HTTP-Erreichbarkeit und Backend-Bereitschaft sind unterscheidbar.
- Galerie, Vollbild, Styles und CPU-/Lowend-Option bleiben erreichbar.
- Testergebnisse von pytest und ruff sind dokumentiert.

### 7. M1 — Hardware, GPU-Auswahl und Konfiguration integrieren

Abhängigkeit: M0.

Vorhandene Grundlage: Hardware-Erkennung, neue Konfigurationsmodule und Tests aus dem aktuellen Commit.

Todos:

- [ ] `hardware_detect.py`, `config_generator.py` und `runtime_config.py` abgleichen.
- [ ] Fehlende NVML, Treiberfehler und fehlende GPU kontrolliert behandeln.
- [ ] CPU-/Lowend-Modus ohne NVIDIA-Abhängigkeit initialisieren.
- [ ] GPU-Auswahl auf Host und im Container eindeutig testen.
- [ ] Tier-Grenzen und `SLAIDSHOW_TIER` validieren.
- [ ] Betriebsart, Ressourcenprofil und Bildbackend getrennt konfigurieren.
- [ ] Bestehende Lowend-Einstellungen kontrolliert in die gemeinsame Konfiguration übernehmen.
- [ ] Konfigurationsschema versionieren und ungültige Kombinationen früh ablehnen.
- [ ] Persistierte Konfiguration tatsächlich in `app.py` und Downloader verwenden.
- [ ] Effektive Konfiguration ohne Secrets im Status anzeigen.
- [ ] Bestehende Tests ausführen und um CPU-/Remote-Fälle erweitern.

Abnahme:

- Gemockte Hardwarefälle und Grenzwerte liefern reproduzierbare Ergebnisse.
- Ein gültiger CPU-/Remote-Modus startet ohne NVIDIA-GPU.
- 6gb aktiviert die sichere CPU-STT-/Sequenzstrategie.
- Neustarts respektieren persistierte Nutzerauswahl.
- Gewählte GPU wird tatsächlich verwendet.
- Unzulässige Overrides erzeugen klare Fehler.

### 8. M2 — Lokale und entfernte Modellorchestrierung absichern

Abhängigkeit: M1.

Todos:

- [ ] Scheduler in die LLM-/Bildpipeline integrieren.
- [ ] Gemeinsame lokale GPU-Zugriffe über eine gemeinsame Sperre koordinieren.
- [ ] Ollama serverseitig entladen.
- [ ] Lokales ComfyUI über unterstützte Schnittstellen freigeben.
- [ ] Entladung mit Timeout und gemessener Speicherfreigabe bestätigen.
- [ ] Abbrüche und Ausfälle mit garantiertem Cleanup behandeln.
- [ ] GPU-STT budgetieren oder auf CPU zurückfallen.
- [ ] Remote-ComfyUI ohne Zugriff auf dessen NVML nutzbar halten.
- [ ] Unload-Rechte bei gemeinsam genutztem Remote-Server ausdrücklich regeln.
- [ ] Pollinations vom lokalen Bild-VRAM-Scheduling ausnehmen.
- [ ] Zustände, Wartezeiten und Fehler sichtbar machen.

Lokale sequenzielle Pipeline:

Sprache → CPU-STT → LLM → bestätigtes LLM-Unload → ComfyUI → bestätigtes Bildmodell-Unload → Ausgabe.

Abnahme:

- Konkurrenz, Timeout, Abbruch und Serverausfall sind getestet.
- Fehlgeschlagenes lokales Unload blockiert unsichere Folgearbeit.
- Remote-Betrieb wird nicht durch fehlende lokale GPU-Messwerte blockiert.
- Das letzte gültige Bild bleibt bei Fehlern sichtbar.

### 9. M3 — Reproduzierbare Installation und Backend-Auswahl

Abhängigkeit: M1; M2 für lokale gemeinsame GPU-Nutzung.

#### 9.1 Native Installation und Docker

- [ ] Docker-Build und Startreihenfolge abnehmen.
- [ ] Native Installation auf X260 und E14 dokumentieren.
- [ ] Container-zu-Host- und Container-zu-Remote-Verbindungen dokumentieren.
- [ ] Mikrofonzugriff und Geräteberechtigungen für unterstützte Startpfade prüfen.
- [ ] Ollama und gewähltes Bildbackend zuverlässig erreichbar machen.
- [ ] Service-Readiness von reiner HTTP-Liveness unterscheiden.
- [ ] Konfiguration, Modelle, Ausgaben und Recordings persistent speichern.
- [ ] Geprüfte Container-Versionen verwenden.
- [ ] Frontend-Abhängigkeiten für lokalen Betrieb lokal ausliefern.

#### 9.2 Modellbereitstellung

- [ ] Modellartefakte, Quellen und Lizenzen verifizieren.
- [ ] Nur lokal benötigte Modelle herunterladen.
- [ ] Bei Remote-ComfyUI oder Pollinations keine lokalen Bildgewichte erzwingen.
- [ ] Unvollständige Downloads erkennen.
- [ ] Prüfsummen oder gepinnte Revisionen nutzen.
- [ ] Wiederholte Installation ohne unnötigen erneuten Download ermöglichen.

#### 9.3 Pollinations und „Bring your own Pollen“

- [ ] Offizielle Dokumentation zu API, Authentifizierung, Pollen und Nutzungsbedingungen prüfen.
- [ ] Bestehenden Pollinations-Backendpfad mit dem verifizierten API-Vertrag abgleichen.
- [ ] Eigene Zugangsdaten bei Auswahl von Pollinations sicher konfigurierbar machen.
- [ ] Eigene Nutzung beziehungsweise Abrechnung transparent erläutern.
- [ ] Fehlende oder ungültige Zugangsdaten verständlich melden.
- [ ] Guthaben-/Abrechnungsfehler, Rate Limits, Timeouts und Provider-Ausfälle behandeln.
- [ ] Budget-/Nutzungsanzeigen nur auf Grundlage tatsächlich unterstützter API-Daten anbieten.
- [ ] Keine erfundenen Preise oder garantierten Kosten pro Bild anzeigen.
- [ ] Retry-Regeln auf mögliche zusätzliche Kosten oder Doppelaufträge prüfen.
- [ ] Keine stillen Wiederholungen oder Backendwechsel mit unklaren Kostenfolgen.
- [ ] Zugangsdaten weder an Browser noch an Status- oder Logausgaben weitergeben.
- [ ] Übertragene Prompts, Parameter und gegebenenfalls Referenzbilder transparent ausweisen.
- [ ] Mocktests ohne echte Zugangsdaten und ohne kostenpflichtige Requests ergänzen.

Abnahme:

- Lokaler Modus funktioniert nach initialer Installation ohne Internet.
- CPU-/Remote-Modus benötigt keine lokale NVIDIA-GPU.
- Pollinations funktioniert nach ausdrücklicher Auswahl mit eigenen gültigen Zugangsdaten.
- Ohne Cloud-Freigabe werden keine Pollinations-Requests ausgelöst.
- Backendfehler führen nicht zu einem stillen Cloud-Fallback.
- Gemockter Installations-E2E-Test benötigt weder GPU noch Internet.

### 10. M4 — Demo-Release und Show-Stabilität

Abhängigkeit: M3 für die jeweils geprüfte Betriebsart.

Todos:

- [ ] Begrenzte Queue und Überlaststrategie implementieren.
- [ ] Veraltete Generierungen kontrolliert verwerfen.
- [ ] Ungültige Bilder nicht anzeigen.
- [ ] Echten Crossfade ergänzen.
- [ ] SSE-Wiederverbindung und Zustandsabgleich absichern.
- [ ] Workflow-Presets und Backend-spezifische Style-Grenzen dokumentieren.
- [ ] Titel und Beschriftungen als lokale Overlays anbieten.
- [ ] STT-, LLM-, Netzwerk-, Lade-, Render- und Gesamtzeiten getrennt messen.
- [ ] Median, hohe Perzentile, Fehlerrate und Ressourcenverbrauch dokumentieren.
- [ ] Reproduzierbare Demo-Show erstellen.

Abnahme je Betriebsart:

- Mindestens 30 Minuten auf einer 6-GB-Referenzkarte.
- Mindestens 30 Minuten CPU-/Lowend-Betrieb auf X260 oder E14 mit Remote-ComfyUI.
- Regressionstest auf dem jeweils anderen ThinkPad.
- Pollinations-Funktionstest mit ausdrücklich freigegebenem Testbudget.
- Keine unkontrollierte Queue oder fehlerbedingter Blackout.
- Netzwerk- und Backendausfälle sind sichtbar und kontrolliert behandelbar.

Latenzangaben immer mit Hardware, Modellen, Backend und Einstellungen veröffentlichen. Die bisherigen 0,8 bis 2,5 Sekunden sind keine bestätigte Produkteigenschaft.

### 11. M5 — Recording und Export

Abhängigkeit: M4.

- [ ] Recording-Verfahren und benötigte Abhängigkeiten gesondert entscheiden.
- [ ] Start, Stopp und Aufzeichnungsstatus integrieren.
- [ ] Bildwechsel, Overlays und optionales Audio synchron erfassen.
- [ ] MP4-/WebM-Ausgabe gemäß gewähltem Verfahren unterstützen.
- [ ] Speicherplatzmangel und Abbrüche behandeln.
- [ ] Lokale Ablage und Löschfunktion anbieten.
- [ ] Zusatzlast auf CPU-/Lowend- und 6-GB-Systemen messen.

Abnahme:

- Export in üblichen Playern abspielbar.
- Recording blockiert die Live-Ausgabe nicht.
- Ressourcenschonende Variante für schwache Clients dokumentiert.
- Recording erzeugt keine zusätzlichen Cloud-Aufrufe ohne ausdrückliche Freigabe.

### 12. M6 — Moderation und Team-Betrieb

Abhängigkeit: M4; Recording optional.

Reihenfolge:

1. Separate Moderator- und Display-Ansicht.
2. Skript-Modus mit manuellen und automatischen Bildtriggern.
3. Multi-Speaker mit eindeutiger Quellen- und Rollenzuordnung.
4. Versionierte externe API und Ereignisverträge.
5. Optionale Publikumsinteraktion.

Abnahme:

- Konsistente Sitzungszustände.
- Nachvollziehbare Sprecherwechsel.
- Mehrere Clients umgehen weder Queue noch Scheduler.
- Pollinations-Zugangsdaten und Nutzung sind eindeutig einer Konfiguration beziehungsweise Sitzung zugeordnet.
- Netzwerkunterbrechungen zerstören nicht die gesamte Sitzung.

### 13. M7 — Enterprise und Pilotbetrieb

Abhängigkeit: stabiler Team-Betrieb und konkrete Pilotanforderungen.

- [ ] Sitzungs- und Benutzerisolierung.
- [ ] Rollen, Berechtigungen und Authentifizierung.
- [ ] SSO nach gesonderter Architekturentscheidung.
- [ ] Audit-Log mit konfigurierbarer Aufbewahrung.
- [ ] Verwaltete Modell-, Workflow- und Backend-Freigaben.
- [ ] Administrativ erzwungener lokaler Modus ohne Cloudzugriff.
- [ ] Sichere Verwaltung und Rotation eigener Provider-Zugangsdaten.
- [ ] On-Premise-Deployment, Updates, Backup und Wiederherstellung.
- [ ] Air-Gapped-Anleitung für vollständig lokale Installation.
- [ ] Pilotabnahme für Bildungs- und kreative Live-Anwendungsfälle.

Lokaler Betrieb allein ist keine Datenschutz-Zertifizierung. Remote-, Cloud- und Air-Gapped-Aussagen benötigen jeweils passende technische und organisatorische Nachweise.

### 14. Zurückgestellte Erweiterungen

Diese Themen blockieren den ersten stabilen Release nicht:

- Parakeet als zusätzlicher STT-Backendpfad.
- Große LLMs und FLUX-basierte Qualitätsprofile.
- Automatische Wahl mehrerer Bildmodelle.
- Generierter Text direkt im Bild.
- Pauschaler Wechsel von SSE zu WebSocket.
- Vollständige Frontend-Neuentwicklung.
- Native Desktop-App.
- Apple-Silicon-Unterstützung.
- Workflow-Marktplatz und Lizenzierung.

Nicht zurückgestellt werden der bestehende CPU-/Lowend-Betrieb, Remote-ComfyUI sowie die Absicherung des optionalen Pollinations-Backends einschließlich eigener Zugangsdaten.

### 15. Definition of Done

Ein Meilenstein ist abgeschlossen, wenn:

- Implementierung und Laufzeitintegration vollständig sind.
- Relevante Fehlerfälle getestet sind.
- pytest und ruff erfolgreich laufen.
- Betroffene Betriebsarten auf Referenzsystemen geprüft wurden.
- Native Startpfade nicht durch Docker-Änderungen beschädigt werden.
- Dokumentation und Konfiguration dem Code entsprechen.
- Secrets nicht in Git, Browserantworten oder Logs gelangen.
- Keine unbemerkte Cloud-Abhängigkeit entsteht.
- CPU-/Lowend- und 6-GB-Grenzen klar dokumentiert sind.
- Kostenpflichtige Requests nur nach ausdrücklicher Auswahl erfolgen.
- Reproduzierbare Abnahmeschritte und Ergebnisse vorliegen.

### 16. Unmittelbar nächste Aufgaben nach `4caa19f`

1. Repository-Hygiene, vorhandene Tests und ruff prüfen.
2. `Dockerfile`, `docker-compose.yml` und `runtime_config.py` gegen native Startpfade abgleichen.
3. Ursache für `ollama_ok:false` untersuchen.
4. Betriebsarten ausdrücklich unterscheiden: vollständig lokal, CPU-/Remote-ComfyUI, Pollinations.
5. Bestehende Lowend-Einstellungen und Backend-Auswahl in die gemeinsame Konfiguration integrieren.
6. GPU-Auswahl, Tier-Grenzen und Overrides auf Host und im Container testen.
7. Lokalen Scheduler mit bestätigten serverseitigen Unloads absichern.
8. Pollinations-API und „Bring your own Pollen“-Umsetzung verifizieren und integrieren.
9. Native CPU-/Remote-Installation und 6-GB-Docker-Installation getrennt abnehmen.
10. Erst danach Recording, Team-Funktionen und große Qualitätsmodelle ausbauen.

Release-Reihenfolge:

Bestehende Betriebsarten erhalten → gemeinsame Konfiguration und GPU-Auswahl absichern → lokale und entfernte Pipeline stabilisieren → native Installation und Docker reproduzierbar machen → optionales Pollinations mit eigenen Zugangsdaten absichern → Live-Demo abnehmen → Recording → Team → Enterprise.
