.PHONY: run lint format test

run:
	streamlit run app.py

lint:
	ruff check .

format:
	ruff format .

test:
	pytest
