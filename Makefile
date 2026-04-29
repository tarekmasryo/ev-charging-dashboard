.PHONY: run lint format format-check test

APP_FILE := EV-Charging-Analytics.py

run:
	python -m streamlit run $(APP_FILE)

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

test:
	pytest -q
