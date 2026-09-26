PY := .venv/bin/python

# Set up Python virtual environment with dependencies
setup:
	uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -r requirements.txt

# Download and prepare dataset, generate split summary
data:
	$(PY) -m fraud.cli data

# Train all baseline models with all imbalance strategies
train:
	$(PY) -m fraud.cli train

# Evaluate best model: anomaly detection, clustering, calibration, SHAP, stability
evaluate:
	$(PY) -m fraud.cli evaluate

# Train and evaluate DP-SGD models at multiple privacy budgets
privacy:
	$(PY) -m fraud.cli privacy

# Export to ONNX and Core ML, benchmark latency
onnx:
	$(PY) -m fraud.cli onnx

# Detect feature and score drift using PSI
drift:
	$(PY) -m fraud.cli drift

# Run pytest on the test suite
test:
	$(PY) -m pytest -q

# Start Docker Compose services for streaming
stream-up:
	docker compose -f streaming/docker-compose.yml up -d --build --wait

# Run streaming experiment
stream:
	$(PY) streaming/run_experiment.py

# Stop Docker Compose services
stream-down:
	docker compose -f streaming/docker-compose.yml down -v

# Run all pipeline stages
all: data train evaluate privacy onnx drift

# Run CI pipeline: sampled data, reduced epochs, minimal configuration
ci:
	$(PY) -m fraud.cli all --sample 0.05 --ae-epochs 2 --dp-epochs 1 --skip-coreml

# Remove build artifacts and cache
clean:
	rm -rf results/artifacts .pytest_cache
