.PHONY: install up down test

install:
	@echo "═══ slAIdshow Install ═══"
	@echo ""
	@echo "Step 1/4: Detecting hardware..."
	docker compose run --rm init python -m slaidshow.hardware_detect
	@echo ""
	@echo "Step 2/4: Generating config..."
	docker compose run --rm init python -m slaidshow.config_generator
	@echo ""
	@echo "Step 3/4: Downloading models (duration depends on tier)..."
	docker compose run --rm init python -m slaidshow.model_downloader
	@echo ""
	@echo "Step 4/4: Starting services..."
	docker compose up -d
	@echo ""
	@echo "✅ slAIdshow is running at http://localhost:8080"
	@cat /app/config/generated.yaml

up:
	docker compose up -d

down:
	docker compose down

test:
	docker compose run --rm init pytest tests/ -v
