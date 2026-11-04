.DEFAULT_GOAL := help

COMPOSE ?= docker compose

.PHONY: help build install hardware config validate download-policy up down logs test

help:
	@echo "slAIdshow"
	@echo ""
	@echo "make build            Container bauen"
	@echo "make install          Hardware erfassen und Konfiguration prüfen"
	@echo "make hardware         Hardware-Datei erzeugen"
	@echo "make config           Konfiguration erzeugen"
	@echo "make validate         Persistierte Konfiguration prüfen"
	@echo "make download-policy  Downloadrichtlinie prüfen"
	@echo "make up               Laufzeitdienste explizit starten"
	@echo "make down             Dienste stoppen"
	@echo "make logs             Laufzeitprotokolle anzeigen"
	@echo "make test             pytest ausführen"
	@echo ""
	@echo "Hinweis: install bestätigt keine Modell- oder Show-Startfähigkeit."

build:
	$(COMPOSE) build init backend

# Keep initialization sequential, including when make is invoked with -j.
# Do not start runtime services before model and workflow acceptance.
install:
	@echo "═══ slAIdshow: Konfiguration vorbereiten ═══"
	$(COMPOSE) build init
	@echo ""
	@echo "Schritt 1/4: NVIDIA-Hardware erfassen"
	$(COMPOSE) run --rm --no-deps --workdir /app init python hardware_detect.py
	@echo ""
	@echo "Schritt 2/4: Konfiguration erzeugen"
	$(COMPOSE) run --rm --no-deps --workdir /app init python config_generator.py
	@echo ""
	@echo "Schritt 3/4: Konfiguration validieren"
	$(COMPOSE) run --rm --no-deps --workdir /app init python runtime_config.py
	@echo ""
	@echo "Schritt 4/4: Downloadrichtlinie prüfen"
	$(COMPOSE) run --rm --no-deps --workdir /app init python model_downloader.py
	@echo ""
	@echo "Konfigurationsvorbereitung abgeschlossen."
	@echo "Downloads bleiben deaktiviert; Laufzeitdienste wurden nicht gestartet."
	@echo "Modell-, Workflow-, Scheduler- und Hardware-Abnahme stehen noch aus."

hardware:
	$(COMPOSE) run --rm --no-deps --workdir /app init python hardware_detect.py

config:
	$(COMPOSE) run --rm --no-deps --workdir /app init python config_generator.py

validate:
	$(COMPOSE) run --rm --no-deps --workdir /app init python runtime_config.py

download-policy:
	$(COMPOSE) run --rm --no-deps --workdir /app init python model_downloader.py

# Configuration validation does not establish backend or model readiness.
# Start only runtime services; the init service must remain one-shot.
up:
	$(COMPOSE) run --rm --no-deps --workdir /app init python runtime_config.py
	$(COMPOSE) up -d ollama comfyui backend
	@echo "Start angefordert; Betriebsbereitschaft ist noch nicht bestätigt."

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs --follow --tail=100 backend ollama comfyui

# Unit tests must mock GPU access and external model servers.
test:
	$(COMPOSE) run --rm --no-deps --workdir /app init python -m pytest tests/ -v
