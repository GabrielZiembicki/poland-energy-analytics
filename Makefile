ANALYSIS ?= 02_capture_factor

SLUG := $(shell echo "$(ANALYSIS)" | sed 's/^[0-9]*_//' | tr '_' '-')
OUTPUT_JSON := outputs/$(SLUG)/variables.json
DEST_DIR := /Users/gziembicki/repo/clarwise/landing_page/src/content/metrics
DEST_JSON := $(DEST_DIR)/$(SLUG).json

.PHONY: run

run:
	uv run python notebooks/$(ANALYSIS).py
	mkdir -p $(DEST_DIR)
	cp $(OUTPUT_JSON) $(DEST_JSON)
	@echo "Copied $(OUTPUT_JSON) → $(DEST_JSON)"
