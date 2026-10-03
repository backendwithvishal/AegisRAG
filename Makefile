# ==============================================================================
# AegisRAG - Makefile Command Automation
# ==============================================================================

.PHONY: install ingest run test eval benchmark clean docker-build docker-up

install:
	pip install -r requirements-prod.txt
	pip install pytest pytest-asyncio httpx

ingest:
	python -m app.ingestion.processor --dir DATA --incremental

ingest-wipe:
	python -m app.ingestion.processor --dir DATA --wipe

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -v tests/

eval:
	python -m evals.cli --mode all

benchmark:
	python -m benchmarks.run_benchmark
	python -m benchmarks.ablation

docker-build:
	docker build -t aegis-rag:latest .

docker-up:
	docker-compose up -d

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__
	rm -f state_checkpoint.db feedback.db
