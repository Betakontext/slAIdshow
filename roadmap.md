Planungsgrundlage: dokumentierter Stand aus slAIdshow_summary.md und Abgleich mit der bisherigen Roadmap; vorhandener Code gilt erst nach erfolgreichem Test als abgenommen.
slAIdshow — Roadmap zur produktionsreifen lokalen Show-Engine

Produktziel: Sprache wird live in passende Bilder übersetzt — vollständig lokal betreibbar, mit zuverlässiger Ausgabe und automatischer Hardware-Konfiguration ab 6 GB VRAM.

Priorität: Bestehende Komponenten integrieren und stabilisieren, bevor neue Modelle, Recording oder Enterprise-Funktionen hinzukommen.
1. Arbeitsregeln und Status

    Vorhanden: Implementierung im dokumentierten Stand enthalten.
    Teilweise: Bausteine vorhanden, aber Integration oder Funktionsnachweis fehlt.
    Offen: Keine vollständige Implementierung nachgewiesen.
    Abgenommen: Akzeptanzkriterien erfüllt und Testergebnis dokumentiert.

Eine Datei im Repository ist kein abgeschlossener Meilenstein. Installation, Laufzeitintegration und Fehlerfälle gehören zur Abnahme.

Technische Leitplanken:

    Python 3.12+, Type Hints, ruff.
    FastAPI, Ollama, ComfyUI und Docker Compose.
    Bestehende Funktionen erhalten; keine unnötige Komplettmigration.
    Keine Cloud-Abhängigkeiten im lokalen Runtime-Pfad.
    pynvml zur NVIDIA-Erkennung, JSON für Hardwaredaten, YAML für Konfiguration.
    pytest mit Mocks für GPU und externe Modellserver.
    Eine gemeinsame Konfigurationsquelle statt widersprüchlicher Defaults.
    Neue Abhängigkeiten erst nach ausdrücklicher technischer Entscheidung.

2. Dokumentierter Ausgangspunkt
Bereich	Stand	Nächster Schritt
FastAPI-Anwendung in app.py	Vorhanden	Als verbindlichen Einstiegspunkt konsolidieren
Mikrofon, VAD und Transkription	Vorhanden	Bestehende Whisper-Backends absichern
Ollama-Prompt-Erzeugung	Vorhanden	Mit Konfiguration und Scheduler verbinden
ComfyUI-Bildgenerierung	Vorhanden	Workflow-Parameter und Fehlerfälle testen
Galerie, Vollbild und Stilsteuerung	Vorhanden	Regressionstests und Show-Abnahme
Live-Ereignisse über SSE	Vorhanden	Beibehalten und Wiederverbindung absichern
Hardware-Erkennung und Tier-Konfiguration	Teilweise	Fehlerfälle, Grenzwerte und Integration korrigieren
Modell-Downloader	Teilweise	Quellen, Vollständigkeit und Wiederholbarkeit prüfen
VRAM-Scheduler	Teilweise	Tatsächliche serverseitige Entladung implementieren
Docker-Quick-Start	Teilweise	Build, Pfade und Startreihenfolge schließen
Recording	Offen	Nach stabiler Basispipeline
Multi-Speaker und Skript-Modus	Offen	Nach Demo-Release
Enterprise-Funktionen	Offen	Nach pilotiertem Team-Betrieb

Nicht nachgewiesen: erfolgreicher Gesamtinstallationslauf, belastbare VRAM-Abnahme und gemessene End-to-End-Latenzen pro Hardwareprofil.
3. Verbindliche Architekturentscheidungen
3.1 Bestehende Anwendung zuerst integrieren

app.py bleibt zunächst der Anwendungseinstieg. Docker Compose, Makefile und Startskripte werden daran ausgerichtet.

Eine spätere Migration in ein slaidshow/-Paket erfolgt separat und nur mit Regressionstests. Installation und Paketmigration werden nicht gleichzeitig umgebaut.
3.2 Eine gemeinsame Konfigurationsquelle

Standardpfade im Container:

    /app/config/hardware.json
    /app/config/generated.yaml
    /app/models
    /app/recordings

CONFIG_PATH erlaubt einen abweichenden Konfigurationspfad außerhalb von Docker.

Konfigurationsreihenfolge:

    Hardware erfassen.
    Automatisches Profil auswählen.
    Explizite Overrides prüfen.
    Konfiguration validieren und atomar speichern.
    Dieselbe Konfiguration beim Download und zur Laufzeit verwenden.

Bestehende Umgebungsvariablen werden dokumentiert zugeordnet oder kontrolliert abgelöst. Es darf keine zweite, widersprüchliche Tier-Definition geben.
3.3 Sichere VRAM-Nutzung statt pauschalem Parallelbetrieb

Die bisherige Regel „ab 12 GB ist der Scheduler ein No-Op“ entfällt.

Parallelbetrieb wird nur freigegeben, wenn Modelle, Kontextspeicher, STT, Bildauflösung und Sicherheitsreserve gemeinsam in das verfügbare Speicherbudget passen.

Bis zur Abnahme gilt für jede Modellkombination sequenzieller GPU-Betrieb.
3.4 Bestehendes SSE bleibt erhalten

SSE bleibt zunächst der Kanal für Status, Transkription und Bildereignisse. WebSocket wird nur ergänzt, wenn bidirektionales Audio oder eine konkrete neue Funktion es benötigt.

Ein Wechsel zu WebSocket ist kein Voraussetzungsschritt für Vermarktungsreife.
3.5 Modelle werden verifiziert, nicht vorausgesetzt

Modellnamen, Quantisierungen, Downloadquellen, Lizenzen und Backend-Kompatibilität müssen vor Integration geprüft werden.

Insbesondere ist Parakeet ein eigener Integrationspunkt: Ein Modellname in der Konfiguration ersetzt keinen kompatiblen STT-Backendpfad. Bestehendes Whisper bleibt bis zur erfolgreichen Abnahme verfügbar.
4. Hardwareprofile

Profilnamen entsprechen künftig der tatsächlichen Mindestkapazität. Kleinere Karten werden nicht automatisch einem nominell größeren Profil zugeordnet.
Profil	Auswahlbereich	Sichere Ausgangsstrategie
6gb	6 bis unter 8 GiB	CPU-STT, kleines quantisiertes LLM, SD1.5+LCM bei 512 px, sequenziell
8gb	8 bis unter 16 GiB	Zunächst gleiche sichere Basis; größere Modelle oder 768 px erst nach Messung
16gb	16 bis unter 24 GiB	Geprüfte größere Modelle; standardmäßig sequenziell
24gb	Ab 24 GiB	Qualitätsoptionen und kontrollierter Parallelbetrieb nach Budgetprüfung

Zusätzliche Regeln:

    Unter 6 GiB: klare Meldung; kein ungeprüftes Echtzeitversprechen für CPU-only.
    Mehrere GPUs: eine GPU explizit auswählen; VRAM nicht addieren.
    Gesamt-VRAM bestimmt die Profilfähigkeit, freier VRAM die aktuelle Ausführbarkeit.
    Desktopbelegung und Laufzeitspitzen werden durch eine messwertbasierte Reserve berücksichtigt.
    SLAIDSHOW_TIER=6gb ermöglicht ein kleineres Profil.
    Ein zu großes Override wird mit verständlicher Fehlermeldung abgelehnt.
    Apple Silicon ist kein Bestandteil der ersten NVIDIA-/Docker-Abnahme.

6-GB-Funktionsumfang: Vollbild, Galerie, Stilsteuerung, Text-Overlays und später Recording bleiben vorgesehen. Große Bildmodelle und parallele GPU-Modellresidenz gehören nicht zur 6-GB-Basis.
5. Meilenstein M0 — Bestand und Startfähigkeit konsolidieren

Ziel: Ein eindeutiger, reproduzierbarer Ausgangsstand.

    Dokumentierten Dateibaum mit dem tatsächlichen Repository abgleichen.
    Anwendungseinstieg, Imports und Modulpfade vereinheitlichen.
    Dockerfile ergänzen beziehungsweise vorhandenen Build prüfen.
    Compose, Makefile, run.sh und run.ps1 konsistent ausrichten.
    Abhängigkeiten auf den tatsächlich verwendeten Stack reduzieren und festhalten.
    Bestehende API-Routen und Ereignisformate dokumentieren.
    Minimalen Smoke-Test für App-Start, Health und Status ergänzen.
    Bekannte Fehler als nachvollziehbare Aufgaben erfassen.

Abnahme:

    Anwendung startet über den dokumentierten Einstiegspunkt.
    Health- und Statusantworten sind getestet.
    Kein Startpfad verweist auf ein nicht vorhandenes Paket.
    Bestehende Galerie, Vollbild- und Stilfunktionen bleiben erreichbar.

6. Meilenstein M1 — Hardware und Konfiguration integrieren

Abhängigkeit: M0.

    hardware_detect.py gegen fehlende NVML, Treiberfehler und fehlende GPU absichern.
    GPU-Auswahl und Speicherwerte eindeutig dokumentieren.
    config_generator.py als eigenständiges, ausführbares Modul bereitstellen.
    Tier-Grenzen ohne Lücken implementieren.
    SLAIDSHOW_TIER validieren.
    Konfigurationsschema mit Versionsfeld einführen.
    Ungültige oder unvollständige Konfigurationen früh ablehnen.
    Generierte Konfiguration in app.py tatsächlich laden.
    STT, Ollama und Bildbackend aus dieser Konfiguration initialisieren.
    Effektive Konfiguration ohne Geheimnisse im Status ausgeben.

Abnahme:

    Gemockte Hardwarefälle liefern reproduzierbare Profile.
    Grenzwerte einschließlich knapp darunter und darüber sind getestet.
    Ein 6-GB-Profil aktiviert CPU-STT und sequenzielle GPU-Nutzung.
    Neustarts verwenden die persistierte Konfiguration.
    Unbekannte Profilnamen und unzulässige Overrides führen zu klaren Fehlern.

7. Meilenstein M2 — Echte VRAM-Orchestrierung

Abhängigkeit: M1.

    Scheduler direkt in die LLM-/Bildpipeline integrieren.
    Gleichzeitige GPU-Zugriffe mit einer gemeinsamen Sperre verhindern.
    Ollama-Modelle über den Modellserver entladen.
    ComfyUI-Modelle über die unterstützte Server-Schnittstelle freigeben.
    Entladung mit Timeout und gemessener Speicherfreigabe bestätigen.
    Fehler und Request-Abbrüche mit garantiertem Cleanup behandeln.
    GPU-residentes STT in die Budgetplanung aufnehmen oder auf CPU zurückfallen.
    Zustände, Wartezeiten und Fehler im Status sichtbar machen.
    Bei unbestätigter Entladung keine nächste GPU-Phase starten.

Sequenzielle Standardpipeline:

Sprache → CPU-STT → LLM → bestätigtes LLM-Unload → ComfyUI → bestätigtes Bildmodell-Unload → Ausgabe.

Abnahme:

    Tests decken Konkurrenz, Timeout, Abbruch und Serverausfall ab.
    Ein fehlgeschlagenes Unload blockiert unsichere Folgearbeit.
    Ein 6-GB-Dauerlauf erzeugt keine unkontrollierte Modellüberlappung.
    Das zuletzt gültige Bild bleibt bei Fehlern sichtbar.

8. Meilenstein M3 — Reproduzierbarer Docker-Quick-Start

Abhängigkeit: M1 und M2.

    Konkrete Modellartefakte und deren Lizenzen verifizieren.
    Downloader lädt ausschließlich die ausgewählten, benötigten Dateien.
    Unvollständige Downloads erkennen; nicht als fertige Modelle behandeln.
    Downloads über verfügbare Prüfsummen oder gepinnte Revisionen absichern.
    Ollama für Modellinstallation zuverlässig erreichbar machen.
    Service-Readiness und Init-Reihenfolge definieren.
    Konfiguration, Modelle, Ausgaben und Aufzeichnungen persistent speichern.
    Geprüfte Container-Versionen statt ungeprüfter latest-Stände verwenden.
    Lokalen Modus als Standard konfigurieren; Cloudpfade explizit deaktivieren.
    Frontend-Abhängigkeiten lokal ausliefern.
    Installations- und Update-Anleitung vervollständigen.

Abnahme:

    Dokumentierter Installationslauf funktioniert auf sauberer unterstützter Umgebung.
    Wiederholung lädt vollständige Modelle nicht erneut.
    Nach dem initialen Download funktioniert die Show ohne Internet.
    Fehlende Modelle erzeugen eine konkrete Fehlermeldung statt stiller Cloud-Ausweichlogik.
    Gemockter Installations-E2E-Test läuft ohne GPU und Internet.

9. Meilenstein M4 — Demo-Release und Show-Stabilität

Abhängigkeit: M3.

    Begrenzte Queue und dokumentierte Überlaststrategie implementieren.
    Veraltete Generierungen kontrolliert verwerfen.
    Fehlerhafte Bilder nicht an die Ausgabe übergeben.
    Echten Crossfade zwischen gültigen Bildern ergänzen.
    SSE-Wiederverbindung und Zustandsabgleich absichern.
    Unterstützte Styles als geprüfte Workflow-Presets ausliefern.
    Titel und Beschriftungen als lokale Text-Overlays anbieten.
    STT-, LLM-, Lade-, Render- und Gesamtzeiten getrennt messen.
    Median, hohe Perzentile, Fehlerrate und VRAM-Spitzen dokumentieren.
    Eine reproduzierbare Demo-Show aufzeichnen.

Abnahme:

    Mindestens 30 Minuten Live-Betrieb auf einer 6-GB-Referenzkarte.
    Keine unbehandelte Ausnahme, unkontrollierte Queue oder fehlerbedingter Blackout.
    Ollama-/ComfyUI-Ausfälle sind sichtbar und kontrolliert behandelbar.
    Latenzangaben werden nur mit Hardware, Modellen und Einstellungen veröffentlicht.

Keine pauschale Zusage: Die bisherigen Angaben von 0,8 bis 2,5 Sekunden gelten nicht als bestätigte Produkteigenschaft.
10. Meilenstein M5 — Recording und Export

Abhängigkeit: M4.

    Recording-Architektur einschließlich benötigter Abhängigkeiten gesondert entscheiden.
    Start, Stopp und Aufzeichnungsstatus integrieren.
    Bildwechsel, Overlays und optionales Audio synchron erfassen.
    MP4-/WebM-Ausgabe gemäß gewähltem Verfahren unterstützen.
    Speicherplatzmangel und abgebrochene Aufzeichnungen behandeln.
    Lokale Ablage und Löschfunktion bereitstellen.
    Zusätzliche CPU-, RAM- und GPU-Last messen.

Abnahme:

    Export ist in üblichen Playern abspielbar.
    Recording blockiert die Live-Ausgabe nicht.
    6-GB-Betrieb besitzt eine geprüfte ressourcenschonende Variante.

11. Meilenstein M6 — Moderation und Team-Betrieb

Abhängigkeit: M4; Recording ist optional.

Reihenfolge:

    Separate Moderator- und Display-Ansicht.
    Skript-Modus mit manuellem Weiter und automatischen Bildtriggern.
    Multi-Speaker mit expliziter Quellen- und Rollenzuordnung.
    Versionierte externe API mit dokumentierten Ereignisverträgen.
    Optionale Publikumsinteraktion.

Abnahme:

    Moderator und Display besitzen konsistente Sitzungszustände.
    Sprecherwechsel ist eindeutig nachvollziehbar.
    Mehrere Clients umgehen weder Queue noch VRAM-Scheduler.
    Netzwerkunterbrechungen führen nicht zum Verlust der gesamten Sitzung.

12. Meilenstein M7 — Enterprise und Pilotbetrieb

Abhängigkeit: stabiler Team-Betrieb und konkrete Pilotanforderungen.

    Sitzungs- und Benutzerisolierung.
    Rollen, Berechtigungen und Authentifizierung.
    SSO nach gesonderter Architekturentscheidung.
    Audit-Log mit konfigurierbarer Aufbewahrung.
    Verwaltete Modell-/Workflow-Freigaben.
    On-Premise-Deployment- und Update-Verfahren.
    Backup, Wiederherstellung und Support-Diagnose.
    Air-Gapped-Installationsanleitung und Netzwerktest.
    Pilotabnahme mit Bildungseinrichtung und kreativem Live-Anwendungsfall.

Grenze der Produktkommunikation: Lokaler Betrieb allein ist keine Datenschutz-Zertifizierung. Compliance- und Air-Gapped-Aussagen benötigen dokumentierte technische und organisatorische Nachweise.
13. Zurückgestellte Erweiterungen

Diese Themen blockieren den ersten stabilen Release nicht:

    Parakeet als zusätzlicher STT-Backendpfad.
    Große LLMs und FLUX-basierte Qualitätsprofile.
    Automatische Wahl mehrerer Bildmodelle.
    Generierter Text direkt im Bild.
    Pauschaler Wechsel von SSE zu WebSocket.
    Vollständige Frontend-Neuentwicklung.
    Native Desktop-App.
    Apple-Silicon-Unterstützung.
    Workflow-Marktplatz und Lizenzierung.

Jede Erweiterung benötigt einen messbaren Nutzen sowie Tests für Speicherbedarf, Kompatibilität und Lizenzbedingungen.
14. Definition of Done

Ein Meilenstein ist abgeschlossen, wenn:

    Implementierung und Laufzeitintegration vollständig sind.
    Relevante Fehlerfälle getestet sind.
    pytest und ruff erfolgreich laufen.
    Betroffene GPU-Funktionen auf Referenzhardware geprüft wurden.
    Dokumentation und Konfigurationsbeispiele dem Code entsprechen.
    Keine unbemerkte Cloud-Abhängigkeit eingeführt wurde.
    Die 6-GB-Basis funktioniert oder eine klare Feature-Grenze dokumentiert ist.
    Reproduzierbare Abnahmeschritte und Ergebnisse vorliegen.

15. Unmittelbar nächste Aufgaben

    Tatsächlichen Repository-Stand erfassen und M0 schließen.
    Gemeinsame Konfiguration bis in die laufende Anwendung integrieren.
    Tier-Grenzen und Overrides testen.
    Scheduler mit echten serverseitigen Unloads implementieren.
    Erst danach den vollständigen 6-GB-Docker-Installationslauf abnehmen.

Release-Reihenfolge: integrierte Basis → sichere 6-GB-Pipeline → reproduzierbare Installation → stabile Live-Demo → Recording → Team → Enterprise.
