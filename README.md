# Advanced Machine Learning — Python

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-deep_learning-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Grade](https://img.shields.io/badge/Projects-12%2F12_·_30%2F30_cum_laude-brightgreen?style=flat)
![University](https://img.shields.io/badge/SDU_Odense-Erasmus_Exchange-red?style=flat)

**Course:** Advanced Machine Learning  
**Institution:** University of Southern Denmark (SDU), Odense — Erasmus Exchange  
**Year:** Spring 2025  
**Grade:** Both projects individually verified at 12/12 (Danish scale) · converted to 30/30 cum laude

---

## Overview

Two independent projects covering deep learning for computer vision and natural language processing, developed during an Erasmus exchange semester. Each project follows a rigorous experimental methodology: baseline model → ablation study across hyperparameters → best configuration selection → evaluation and interpretability analysis.

---

## Project 1 — Cat & Dog Image Classification (CNN)

Binary image classification on a dataset of 150×150 RGB images using a custom Convolutional Neural Network built in **PyTorch**.

### Architecture

A two-layer CNN with configurable pooling, regularization, and dropout:

```
Input (3 × 150 × 150)
    → Conv2d(3 → 32, kernel=3, padding=1) + ReLU
    → Conv2d(32 → 64, kernel=3, padding=1) + ReLU
    → Pooling layer (configurable)
    → Dropout
    → FC(64 × h × w → 128) + ReLU
    → FC(128 → 2)
    → Softmax
```

### Experimental Methodology

The model was systematically scaled up through four independent ablation studies, each building on the best configuration of the previous:

**1. Pooling layer comparison**

| Configuration | Validation Accuracy |
|---|---|
| Max Pooling (2×2, stride 2) | baseline |
| Average Pooling (2×2, stride 2) | lower |
| **Max Pooling (3×3, stride 2)** | **best** |
| Max Pooling (3×3, stride 3) | lower |

**2. Regularization**

| Configuration | Result |
|---|---|
| No regularization | overfitting observed |
| **L1 regularization** | **best** |
| L2 regularization | slightly lower |

**3. Optimizer**

| Optimizer | Notes |
|---|---|
| **Adam** (lr=0.0005) | **best overall** |
| AdamW | comparable |
| SGD (momentum 0.0 / 0.9) | slower convergence |

**4. Dropout rate**

| Rate | Result |
|---|---|
| 0.00 | overfitting |
| **0.25** | **best** |
| 0.50 | underfitting begins |
| 0.75 / 1.00 | poor performance |

**Best configuration:** Max Pooling (3×3, stride 2) · L1 regularization · Adam (lr=0.0005) · Dropout 0.25

Training used early stopping with patience=3 and a train/val/test split. Final model evaluated via confusion matrix on held-out test set.

### Data Augmentation

Training images augmented with random horizontal flips, affine transforms (±15° rotation, 80–120% zoom scale), and normalised to `mean=0.5, std=0.5` per channel.

---

## Project 2 — Emotion Recognition from Text (NLP)

Six-class emotion classification (`sadness`, `joy`, `love`, `anger`, `fear`, `surprise`) on the `dair-ai/emotion` dataset from HuggingFace, using four progressively more powerful models with full experimental ablation and interpretability analysis.

### Models

**Model 1 — Embedding + Pooling Classifier**  
Custom tokeniser and vocabulary built from scratch. Text sequences embedded via `nn.Embedding`, pooled (average or max), and classified via a single linear layer. Hyperparameter grid searched across: pooling type × regularization × optimizer × dropout × momentum.

**Model 2 — Bag-of-Words MLP**  
Sparse BoW features extracted via `CountVectorizer`, fed into a 3-layer MLP. Same ablation grid as Model 1. Serves as a strong non-sequence baseline showing what can be achieved without any positional or contextual information.

**Model 3 — Transformer Classifier (from scratch)**  
Custom `TransformerClassifier` built with `nn.TransformerEncoder`. Learnable positional embeddings (`nn.Parameter`), multi-head self-attention, configurable number of layers and heads. Trained end-to-end on the same tokenised sequences as Model 1.

```python
TransformerClassifier(
    vocab_size, embed_dim, num_heads,
    hidden_dim, num_classes, num_layers,
    dropout=0.1, max_len=128
)
```

**Model 4 — Fine-tuned BERT**  
`bert-base-uncased` fine-tuned via HuggingFace `Trainer` API with `BertForSequenceClassification`. Hyperparameter search over: learning rate ∈ {2e-5, 3e-5, 5e-5} × weight decay ∈ {0.0, 0.01, 0.1} × batch size ∈ {16, 32} × epochs ∈ {3, 5}.

### Interpretability

All four models analysed with two complementary techniques:

- **LIME** (`lime.lime_text.LimeTextExplainer`) — local surrogate model explaining which tokens drive each prediction. Applied to all four models via a unified `predict_proba` wrapper interface.
- **Integrated Gradients** (`captum.attr.IntegratedGradients`) — gradient-based attribution method highlighting token-level importance for the Transformer and BERT models.

Each model also includes an **interactive classification mode**: the user can type any sentence and receive a real-time emotion prediction with explanations.

---

## Repository Structure

```
advanced-ml-sdu/
├── project1-cat-dog-cnn/
│   ├── aml_project1.ipynb
│   └── cat_dog_classification_report.html
├── project2-emotion-nlp/
│   └── Emotion_project_aml.ipynb
└── README.md
```

---

## Stack

| Tool | Purpose |
|---|---|
| PyTorch | CNN, Transformer, training loops |
| HuggingFace Transformers | BERT fine-tuning |
| HuggingFace Datasets | `dair-ai/emotion` dataset loading |
| scikit-learn | `CountVectorizer`, metrics, `classification_report` |
| LIME | Local model-agnostic interpretability |
| Captum | Integrated Gradients attribution |
| Matplotlib / Seaborn | Training curves, confusion matrices |
