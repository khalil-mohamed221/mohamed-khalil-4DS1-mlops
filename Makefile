#SHELL=/bin/bash
# ============================
# VARIABLES
# ============================
PYTHON=python3
ENV_NAME=venv
REQUIREMENTS=requirements.txt
SHELL=/bin/bash


# ============================
# 1. ENVIRONMENT CONFIGURATION
# ============================
.PHONY: setup
setup:
	@echo "Creating the virtual environment and installing dependencies..."
	@$(PYTHON) -m venv $(ENV_NAME)
	@source $(ENV_NAME)/bin/activate && pip install -r $(REQUIREMENTS)


# ============================
# 2. CODE QUALITY / FORMAT / SECURITY
# ============================
.PHONY: format
format:
	@echo "🧽 Formatting code with Black..."
	@black *.py

.PHONY: lint
lint:
	@echo "🔍 Checking code quality with Pylint..."
	@pylint *.py || true

.PHONY: flake
flake:
	@echo "🧹 Running Flake8 style checks..."
	@flake8 *.py

.PHONY: security
security:
	@echo "🛡️ Checking code security..."
	@bandit -r .

.PHONY: quality
quality:
	@echo "🔎 Running Quality Gate..."
	@$(PYTHON) quality_gate.py



# ============================
# 3. DATA PREPARATION
# ============================
.PHONY: data
data: quality
	@echo "Data preparation..."
	@$(PYTHON) main.py --prepare


# ============================
# 4. MODEL TRAINING
# ============================
.PHONY: train
train: quality
	@echo "Model training..."
	@$(PYTHON) main.py --train


# ============================
# 5. UNIT TESTS
# ============================
.PHONY: test
test: quality
	@echo "🧪 Running tests..."
	@$(PYTHON) test_environment.py


# ============================
# 6. FULL PIPELINE
# ============================
.PHONY: pipeline
pipeline:
	@echo "🚀 Running full ML pipeline..."
	@$(PYTHON) main.py --pipeline


# ============================
# 7. MODEL DEPLOYMENT
# ============================
deploy:
	@echo "🚀 Starting FastAPI server..."
	@source venv/bin/activate && uvicorn app:app --reload



# ============================
# 8. JUPYTER NOTEBOOK
# ============================
.PHONY: notebook
notebook:
	@echo "Starting Jupyter Notebook..."
	@source $(ENV_NAME)/bin/activate && jupyter notebook

# ============================
# 9. DOCKER COMMANDS
# ============================

DOCKER_USER = mohamed1khalil
IMAGE_NAME = mohamed_khalil_4ds1_mlops

.PHONY: docker-build
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t $(IMAGE_NAME) .

.PHONY: docker-run
docker-run:
	@echo "🚀 Running Docker container with MLflow connection..."
	docker run --rm \
		-p 8000:8000 \
		--add-host=host.docker.internal:host-gateway \
		-e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
		$(IMAGE_NAME)


.PHONY: docker-tag
docker-tag:
	@echo "🏷️ Tagging Docker image..."
	docker tag $(IMAGE_NAME) $(DOCKER_USER)/$(IMAGE_NAME):latest

.PHONY: docker-push
docker-push:
	@echo "📤 Pushing image to Docker Hub..."
	docker push $(DOCKER_USER)/$(IMAGE_NAME):latest


# ================================
#  Run MLflow UI
# ================================
# ================================
# MLflow LOCAL DEV (NO Docker, NO Registry)
# ================================
.PHONY: mlflow-ui-local
mlflow-ui-local:
	@echo "⚠️ Starting MLflow UI (LOCAL DEV ONLY)"
	@echo "⚠️ Do NOT use with Docker"
	mlflow ui --host 0.0.0.0 --port 5000


# ================================
# MLflow SERVER (Registry + Docker)
# ================================
.PHONY: mlflow-server
mlflow-server:
	@echo "🚀 Starting MLflow Server (Docker-safe, artifacts served)..."
	mlflow server \
		--host 0.0.0.0 \
		--port 5000 \
		--backend-store-uri sqlite:///mlflow.db \
		--artifacts-destination ./mlruns \
		--serve-artifacts \
		--allowed-hosts '*' \
		--cors-allowed-origins '*'



# ================================
# Pipeline — LOCAL MODE
# ================================
.PHONY: mlflow-pipeline-local
mlflow-pipeline-local:
	@echo "🧪 Running pipeline in LOCAL mode"
	python main.py --mlflow


# ================================
# Pipeline — SERVER MODE
# ================================
.PHONY: mlflow-pipeline
mlflow-pipeline:
	@echo "🚀 Running pipeline with MLflow SERVER"
	MLFLOW_TRACKING_URI=http://localhost:5000 \
	python main.py --mlflow
