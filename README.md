# CS3081 Cloud Computing Mini Project — Knowledge Distillation

Repository: https://github.com/Dushyant0608/cs3081-cloud-miniproject

## What this project does

Uses phi3.5 (served via Ollama) as a teacher model to label 3000
natural-language commands with structured JSON (intent/date/time/place),
then fine-tunes EleutherAI/pythia-70m as a student model via sequence-level
knowledge distillation to reproduce that behavior at a fraction of the
model size.

## Repository structure

cluster/ k3d cluster config
k8s/ Kubernetes manifests (namespaces, SeaweedFS)
prompts/ Teacher prompt template
worker/ Sharded labeling Jobs + cleaning script
scripts/ Setup/upload utility scripts
data/ Cleaned training/validation data + checksums
training/ Training, prediction, scoring scripts + explanation.md
results/ Evaluation results (teacher / untrained / trained student)
report/ LaTeX report


## Prerequisites

- Docker Desktop
- k3d, kubectl
- Python 3.10+ with a virtual environment
- Ollama (for serving the teacher model locally)

## Setup and run

```bash
# 1. Create the cluster and namespaces
k3d cluster create --config cluster/k3d.yaml
kubectl apply -f k8s/namespaces.yaml

# 2. Deploy object storage
kubectl create secret generic seaweedfs-credentials \
  --namespace data --from-literal=ACCESS_KEY=labuser \
  --from-literal=SECRET_KEY=localdevkey123456
kubectl apply -f k8s/seaweedfs.yaml

# 3. Set up Python environment
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 4. Pull and serve the teacher model
ollama pull phi3.5

# 5. Run the labeling jobs (sharded)
kubectl apply -f worker/job-0.yaml -f worker/job-1.yaml -f worker/job-2.yaml

# 6. Clean and prepare training data
python worker/clean.py

# 7. Train the student model
python training/train.py --config training/config.yaml --version v1

# 8. Evaluate
python training/predict_teacher.py
python training/predict_hf.py --model EleutherAI/pythia-70m --out results/predictions_untrained.jsonl
python training/predict_hf.py --model models_local/v1 --out results/predictions_student_v1.jsonl
python tools/score.py
```

## Model artifacts

Trained model weights are not committed to git. They are available via:
- SeaweedFS object storage (`s3://models/v1/`, `s3://models/v2/`), used by
  the pipeline itself
- GitHub Releases (tag `v1-model`) for external download

## Teardown

```bash
k3d cluster delete cloudproj
```

## Notes

Developed and tested on Windows with Docker Desktop.