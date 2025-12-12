#!/bin/bash

# === Project Name ===
PROJECT_NAME="gesture-ml-project_V1"

# === Start ===
echo "Creating Machine Learning Gesture Recognition project structure..."
mkdir -p $PROJECT_NAME

# === DATA ===
mkdir -p $PROJECT_NAME/data/{raw,processed,train,eval,test}

# === SOURCE CODE ===
mkdir -p $PROJECT_NAME/src/{dataset,preprocessing,models,training,evaluation,utils}
mkdir -p $PROJECT_NAME/src/models/{tflite,onnx,checkpoints}

# === NOTEBOOKS ===
mkdir -p $PROJECT_NAME/notebooks

# === CONFIG FILES ===
mkdir -p $PROJECT_NAME/configs

# === LOGS ===
mkdir -p $PROJECT_NAME/logs

# === OUTPUTS ===
mkdir -p $PROJECT_NAME/outputs/{models,plots,reports}

# === SCRIPTS (automation tools) ===
mkdir -p $PROJECT_NAME/scripts

# === EXTRA: README & .gitignore ===
echo "# Gesture Machine Learning Project" > $PROJECT_NAME/README.md
echo -e "logs/\noutputs/\n*.tflite\n*.onnx\n*.h5" > $PROJECT_NAME/.gitignore

echo "Project structure created successfully!"
