PY := .venv/bin/python
GCP_PROJECT ?= your-gcp-project
VPY := .venv-vertex/bin/python

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

# Set up Pub/Sub topic, subscription, and copy model to GCS
cloud-stream-up:
	gcloud pubsub topics create transactions --project $(GCP_PROJECT) || true
	gcloud pubsub subscriptions create transactions-beam --topic transactions --project $(GCP_PROJECT) --ack-deadline 60 || true
	gcloud storage cp results/artifacts/best_tree.onnx gs://$(GCP_PROJECT)-fraud/model/best_tree.onnx
	bq --project_id $(GCP_PROJECT) mk --dataset --location US fraud || true

# Run cloud streaming experiment
cloud-stream:
	$(VPY) streaming/cloud/run_experiment.py --project $(GCP_PROJECT) --rate 2000 --seconds 120

# Cancel Dataflow jobs and tear down Pub/Sub topic and subscription
cloud-stream-down:
	-gcloud dataflow jobs list --project $(GCP_PROJECT) --region us-central1 --status=active --format='value(id)' | xargs -n1 -I{} gcloud dataflow jobs cancel {} --project $(GCP_PROJECT) --region us-central1
	-gcloud pubsub subscriptions delete transactions-beam --project $(GCP_PROJECT) --quiet
	-gcloud pubsub topics delete transactions --project $(GCP_PROJECT) --quiet

# Run all pipeline stages
all: data train evaluate privacy onnx drift

# Run CI pipeline: sampled data, reduced epochs, minimal configuration
ci:
	$(PY) -m fraud.cli all --sample 0.05 --ae-epochs 2 --dp-epochs 1 --skip-coreml

# Set up Vertex AI virtual environment with dependencies
setup-vertex:
	uv venv -q -p 3.11 .venv-vertex && uv pip install -q -p .venv-vertex/bin/python -r requirements-vertex.txt

# Build and push Vertex AI container image
vertex-build:
	gcloud builds submit --config vertex/cloudbuild.yaml --project $(GCP_PROJECT) --substitutions=_IMAGE=us-central1-docker.pkg.dev/$(GCP_PROJECT)/fraud/pipeline:v1 .

# Run Vertex AI pipeline
vertex-run:
	$(VPY) vertex/pipeline.py run --project $(GCP_PROJECT)

# Schedule Vertex AI pipeline
vertex-schedule:
	$(VPY) vertex/pipeline.py schedule --project $(GCP_PROJECT)

# Tear down Vertex AI pipeline
vertex-down:
	$(VPY) vertex/pipeline.py teardown --project $(GCP_PROJECT)

# Remove build artifacts and cache
clean:
	rm -rf results/artifacts .pytest_cache
