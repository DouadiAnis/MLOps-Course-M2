PY=python
VENV=.venv
APP=app.py

init:
	python3 -m venv $(VENV) && . $(VENV)/bin/activate && pip install -U pip && pip install -r requirements.txt

run:
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q

lint:
	flake8 .

format:
	ruff check --fix . || true

build:
	docker build -t mlops-microservice:latest .

train:
	python3 src/train.py --config config/config.yaml
