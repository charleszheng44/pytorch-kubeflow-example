# PyTorch with KubeFlow

This repository contains an end-to-end example of how to use PyTorch with 
KubeFlow for my blog post [From Training to Serving: AI with PyTorch & Kubeflow on K8s](https://personal-blog-cyan-eight.vercel.app/posts/e2e-ai-workflow-on-k8s/).

## Run the pytorch example locally

### Install the dependencies

```bash
python -m venv .venv 
source .venv/bin/activate 
pip install -r requirements.txt
```

### Run the training script

```bash
MODEL_DIR=/tmp/models python train.py
```

### Run the inference script

```bash
MODEL_PATH=/tmp/models/mnist_cnn.pt uvicorn inference:app --host 0.0.0.0 --port 8080 --reload
```

### Try the model

```bash
curl -X POST "http://localhost:8080/predict" -F "file=@/Users/charlesz/Works/pytorch-kubeflow-example/8.png"
``` 
If you see the following output, then the model is working correctly.

```bash
{"prediction":8}
```
