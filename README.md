# Challenge 6 — Advanced Unsupervised Learning: Anomaly Detection with AutoEncoders and Representation Learning

**Course:** Machine Learning — Universidad Distrital Francisco José de Caldas  
**Group:** 2 — Environment & Air Quality  
**Authors:** Angelo Ibañez · David Ariza · Cristhian Truque  

---

## Overview

AutoEncoder (AE) and Variational AutoEncoder (VAE) anomaly detection pipeline applied to **144,815 EPA county-day air quality records**. Reconstruction error is used as an unsupervised anomaly score; an Isolation Forest provides a classical baseline. t-SNE and UMAP visualise the latent spaces. Results are connected to the clustering findings of Challenge 5 through a cross-challenge synthesis.

| Model | Config | Anomaly rate | Threshold | Spearman vs IF |
|---|---|---|---|---|
| AutoEncoder | latent=16, seed=42 | 4.98% | 0.03857 | 0.486 |
| VAE | β=1.0, latent=16, seed=42 | 5.03% | 0.30939 | 0.627 |
| Isolation Forest | n=200, contamination=0.05 | 4.97% | 0.49444 | 1.000 |

Key finding: the top anomalies include the **March 14, 2025 multi-state atmospheric event** (Arkansas, Missouri, Colorado, Iowa) and a **recurring El Paso dust intrusion sequence** (January–April 2025), both confirmed by AE, VAE, and IF simultaneously.

---

## Repository Structure

```
challenge-6_group2/
├── challenge6_group2.ipynb         # Main notebook (all experiments)
├── final_dataset_all_elements.csv  # Preprocessed input (from Challenge 2)
├── requirements.txt
├── README.md
├── CHECKLIST.md
├── artifacts/
│   ├── km_labels_full.npy              # ← REQUIRED: copy from Challenge 5 artifacts/
│   ├── cluster_labels_ch5.csv          # ← REQUIRED: copy from Challenge 5 artifacts/
│   ├── ae_best_latent16_seed42.pt      # AE model weights
│   ├── vae_best_beta1_latent16_seed42.pt  # VAE model weights
│   ├── scaler_challenge6.pkl           # Fitted StandardScaler
│   ├── top50_anomalies.csv             # Top-50 anomalies with domain features
│   └── metrics_summary.csv            # Summary metrics table (all seeds)
└── figures/
    ├── fig1_ae_training_loss.png
    ├── fig2_vae_training_loss.png
    ├── fig3_error_distributions.png
    ├── fig4_tsne_vae_latent.png
    ├── fig5_umap_ae_latent.png
    ├── fig6_detector_agreement.png
    └── fig_beta_vae_umap.png
```

---

## Dependencies on Challenge 5

Before running this notebook, copy the following files from Challenge 5's `artifacts/` folder:

```bash
cp ../challenge-5_group2/artifacts/km_labels_full.npy   artifacts/
cp ../challenge-5_group2/artifacts/cluster_labels_ch5.csv artifacts/
```

These are used in Figure 5 (UMAP coloured by Challenge 5 cluster labels) and in the cross-challenge synthesis.

---

## Setup and Reproduction

### 1. Clone and enter the repository
```bash
git clone https://github.com/<your-org>/challenge-6_group2.git
cd challenge-6_group2
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** After installing PyTorch for the first time, restart the Jupyter kernel before running any cells.

### 4. Place required files
```
challenge-6_group2/
├── final_dataset_all_elements.csv   ← from Challenge 2
└── artifacts/
    ├── km_labels_full.npy           ← from Challenge 5
    └── cluster_labels_ch5.csv       ← from Challenge 5
```

### 5. Run the notebook
```bash
jupyter lab challenge6_group2.ipynb
```

Run all cells in order. Total estimated runtime on CPU: **45–75 minutes**.

---

## Reproducing Individual Experiments

| Section | Cell range | Runtime |
|---|---|---|
| Data loading & split | Cells 7–8 | < 1 min |
| AE training (3 dims × 3 seeds × 150 epochs) | Cell 14 | ~15–25 min |
| VAE training (3 β × 3 seeds × 150 epochs) | Cell 16 | ~15–25 min |
| Isolation Forest (3 seeds) | Cell 18 | ~2 min |
| Anomaly scoring & thresholds | Cell 20 | < 1 min |
| Figure 1 — AE loss curves | Cell 22 | < 1 min |
| Figure 2 — VAE loss curves | Cell 24 | < 1 min |
| Figure 3 — Error distributions | Cell 26 | < 1 min |
| Figure 4 — t-SNE (10k, ~3–5 min) | Cell 28 | ~5 min |
| Figure 5 — UMAP | Cell 30 | ~5 min |
| Figure 6 — Spearman scatter | Cell 32 | < 1 min |
| Anomaly case study (top-50) | Cell 34 | < 1 min |
| Silhouette scores | Cell 36 | ~2 min |
| Metrics summary table | Cell 38 | < 1 min |
| β-VAE UMAP ablation | Cell 40 | ~5 min |
| Save checkpoints | Cell 42 | < 1 min |

---

## Architecture Summary

### AutoEncoder
```
Input (25) → Linear(128) → ReLU → Linear(64) → ReLU
           → Linear(latent_dim)                          ← bottleneck
           → Linear(64) → ReLU → Linear(128) → ReLU → Linear(25)
```
- Loss: MSELoss
- Optimizer: Adam (lr=1e-3)
- Epochs: 150, batch size: 256
- Latent dims tested: {8, 16, 32}

### Variational AutoEncoder
```
Input (25) → Linear(128) → ReLU → Linear(64) → ReLU
           → fc_mu(16), fc_logvar(16)                    ← probabilistic bottleneck
           → reparameterise → Linear(64) → ReLU → Linear(128) → ReLU → Linear(25)
```
- Loss: ELBO = MSE(sum) + β·KL(q‖p)
- KL warm-up: linear 0→β over first 30 epochs
- β values tested: {0.5, 1.0, 4.0}
- Primary config: β=1.0, latent=16

---

## Random Seeds

All stochastic operations use seeds `{42, 7, 123}`. Set via `set_seed(s)` which fixes `random`, `numpy.random`, and `torch.manual_seed`. Primary seed for single-run figures and saved checkpoints: **42**.

---

## Loading Saved Models

```python
import torch
from challenge6_group2 import AutoEncoder, VAE  # or copy class definitions

# Load AE
ae = AutoEncoder(input_dim=25, hidden_dims=[128, 64], latent_dim=16)
ae.load_state_dict(torch.load("artifacts/ae_best_latent16_seed42.pt"))
ae.eval()

# Load VAE
vae = VAE(input_dim=25, hidden_dims=[128, 64], latent_dim=16)
vae.load_state_dict(torch.load("artifacts/vae_best_beta1_latent16_seed42.pt"))
vae.eval()
```

---


## Cross-Challenge Data Flow

```
Challenge 2  ──► final_dataset_all_elements.csv ──► Challenge 5
                                                 └──► Challenge 6

Challenge 5  ──► artifacts/km_labels_full.npy   ──► Challenge 6 (Figure 5, synthesis)
```

## Video

Link to video: https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQBEy2Wr0sQKS4aalJwq4EklAWOLnAylBsHtSGu2Dcq0tnk?e=VPN9ce&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D



## Timestamps
[0:00](https://udistritaleduco-my.sharepoint.com/:v:/r/personal/aaibanezh_udistrital_edu_co/Documents/challenge_6-2_video_final.mp4?csf=1&web=1&e=3Xe28q&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7fX0%3D) - Greetings

[0:35](https://udistritaleduco-my.sharepoint.com/:v:/r/personal/aaibanezh_udistrital_edu_co/Documents/challenge_6-2_video_final.mp4?csf=1&web=1&e=SIIVaV&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MzYuNTF9fQ%3D%3D) - Introduction and Objectives

[1:35](https://udistritaleduco-my.sharepoint.com/:v:/r/personal/aaibanezh_udistrital_edu_co/Documents/challenge_6-2_video_final.mp4?csf=1&web=1&e=ymEtoa&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6OTUuNzl9fQ%3D%3D) - Methodology

[2:33](https://udistritaleduco-my.sharepoint.com/:v:/r/personal/aaibanezh_udistrital_edu_co/Documents/challenge_6-2_video_final.mp4?csf=1&web=1&e=P3bBPt&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MTUzLjc2fX0%3D) - Training Convergence and Ablation 

[3:40](https://udistritaleduco-my.sharepoint.com/:v:/r/personal/aaibanezh_udistrital_edu_co/Documents/challenge_6-2_video_final.mp4?csf=1&web=1&e=NrXgS0&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MjIwLjExfX0%3D) - Representation Learning

[4:33](https://udistritaleduco-my.sharepoint.com/:v:/r/personal/aaibanezh_udistrital_edu_co/Documents/challenge_6-2_video_final.mp4?csf=1&web=1&e=0vQmo2&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MjczLjA1fX0%3D) - Anomaly Detection Results

[5:06](https://udistritaleduco-my.sharepoint.com/:v:/r/personal/aaibanezh_udistrital_edu_co/Documents/challenge_6-2_video_final.mp4?csf=1&web=1&e=YrHD0E&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MzA1LjExfX0%3D) - t-SNE and UMAP

[7:26](https://udistritaleduco-my.sharepoint.com/:v:/r/personal/aaibanezh_udistrital_edu_co/Documents/challenge_6-2_video_final.mp4?csf=1&web=1&e=pFIXLf&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6NDQ2Ljc4fX0%3D) - Cross Challenge Synthesis

[8:25](https://udistritaleduco-my.sharepoint.com/:v:/r/personal/aaibanezh_udistrital_edu_co/Documents/challenge_6-2_video_final.mp4?csf=1&web=1&e=OKCrh5&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6NTA0LjE5fX0%3D) - Conclusions
