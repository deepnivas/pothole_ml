Multimodal Pothole Severity Scoring System
Fusing ESP32-CAM Vision + IMU Telemetry via Cross-Attention.

📌 Project Overview
This repository contains a state-of-the-art Multimodal Pothole Severity Scoring System that combines visual cues (camera frames) and kinematic cues (IMU telemetry) to predict a continuous pothole severity score in the range [1.0, 10.0].

🧠 Model Architecture
Vision Backbone: Pre-trained ConvNeXt-Tiny with projection heads fine-tuned at a stable differential learning rate (2e-5).
IMU Encoder: Two-layer residual Multi-Layer Perceptron (MLP) extracting temporal, spectral, and energy-based motion features from 6-axis telemetry.
Fusion Module: Bidirectional Multi-Head Cross-Attention where visual features attend to motion events and kinematic features attend to road surface structures.
Regressor: Linear layers with a scaled sigmoid activation outputting a continuous score.
🛠️ Hardware Integration (ESP32-CAM + IMU)
Because deep learning models (like ConvNeXt-Tiny with ~29M parameters) require significant RAM and compute, the system operates on a Client-Server Architecture:

Client Edge (ESP32-CAM + MPU6050 IMU): Mounted on a vehicle, the microcontroller captures a high-contrast road frame (JPG) and a synchronized window of 50 samples of raw accelerometer (a 
x
​
 ,a 
y
​
 ,a 
z
​
 ) and gyroscope (g 
x
​
 ,g 
y
​
 ,g 
z
​
 ) data. The client transmits this data via Wi-Fi/HTTP to the central server.
Inference Server (PC/Edge Server/Cloud): Runs the Python PotholeScorer on a GPU to process the multimodal inputs, calculate the severity score in real-time, and store or map the road hazard.
📁 Repository Structure
pothole_multimodal_system.py – The main system pipeline containing network declarations (ConvNeXt, MLP, Cross-Attention), RobustScaler processing, loss configurations (Huber, Focal MSE, Wing Loss), training loops, and inference wrappers.
prepare_dataset.py – Parser that processes Pascal VOC bounding box annotations (XML files) to calculate visual severity scores and generates correlated synthetic IMU telemetry.
requirements.txt – The project dependency configuration list.
.gitignore – Pre-configured git ignore file that excludes virtual environments (venv/), TensorBoard logs (logs/), and large model checkpoints (checkpoints/).
🚀 Quick Start
1. Installation & Environment Setup
Create a virtual environment and install the required dependencies (using the fast uv package manager):

powershell

# Create a Python 3.12 virtual environment
python -m uv venv venv --python 3.12
# Activate the virtual environment
venv\Scripts\activate
# Install dependencies with PyTorch CUDA 12.4 support for GPU acceleration
python -m uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
2. Dataset Preparation
If you have a set of road images and Pascal VOC XML bounding box annotations (e.g., in an annotations/ and images/ directory), place them under the target paths and run the parser to generate the structured telemetry database:

powershell

python prepare_dataset.py
This generates data/telemetry.csv with structured labels and simulated IMU acceleration spikes synchronized to the pothole sizes.

3. Training the Model
To start training the cross-attention network on the GPU:

powershell

python pothole_multimodal_system.py --mode train
Note: The script uses layer-wise differential learning rates and Cosine Annealing scheduler. The default configuration runs for 150 epochs.

4. Running Production Inference
Load the saved best checkpoint and perform real-time severity scoring:

python

from pothole_multimodal_system import PotholeScorer
import numpy as np
# 1. Initialize the scorer
scorer = PotholeScorer("checkpoints/best_model.pt")
# 2. Score a raw trigger (JPG/PNG path + 50x6 IMU array)
fake_imu = np.random.normal(0, 0.1, (50, 6))
result = scorer.predict("path_to_image.png", fake_imu)
print(f"Severity Score: {result['score']} / 10.0")
print(f"Severity Class: {result['severity']}")
print(f"Description: {result['description']}")
📊 Model Evaluation Metrics
After completing a deep 150-epoch training curriculum on the GPU, the model achieved the following performance metrics on the holdout test set:

Metric	Value	Description
Test MAE	1.7704	Average absolute deviation on the [1.0, 10.0] scale
Within-1 Accuracy	37.3%	Percentage of test samples scored within ±1.0 of the true label
Test RMSE	2.3225	Root Mean Squared Error
Spearman Rank Correlation (ρ)	0.044	Order agreement between true and predicted severity
