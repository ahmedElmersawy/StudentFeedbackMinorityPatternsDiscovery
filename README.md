# Student Feedback Minority Pattern Detection & Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmedElmersawy/StudentFeedbackMinorityDetection/blob/main/StudentFeedback_Analysis_Pipeline.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Ensuring every student voice matters in feedback analysis**

Student feedback analysis that ensures minority voices matter. Automatically finds rare, unusual, or underrepresented opinions in course evaluations, then trains an AI classifier that treats all perspectives fairly not just the majority. Perfect for institutions committed to hearing every student and improving courses based on complete feedback.

---

## Quick Start

### **[Open the Main Notebook](StudentFeedback_Analysis_Pipeline.ipynb)**

Click the link above or the "Open in Colab" badge to run the complete 26-cell pipeline in Google Colab.

---

## What's Inside

### Main Components:

1. **StudentFeedback_Analysis_Pipeline.ipynb** (Main File)
   - Complete 26-cell pipeline
   - Data loading, pattern discovery, model training, and evaluation
   - Run this notebook to see the entire project in action

2. **studentdataset.csv**
   - Sample student feedback dataset
   - Contains text feedback and ratings

3. **Output Artifacts:**
   - `final_feedback_classifier/` - Trained DistilRoBERTa model
   - `minority_patterns.json` - Detected minority pattern metadata
   - `minority_patterns.csv` - Exportable rare feedback examples
   - `embeddings.npy` & `embeddings_2d.npy` - Sentence embeddings

---

## Key Features

### Unsupervised Minority Discovery
- Isolation Forest detects outliers (rare feedback patterns)
- HDBSCAN clustering finds minority opinion groups
- Sentence embeddings capture semantic meaning
- PCA visualization shows pattern distribution

### Fair AI Classification
- Class-weighted loss protects minority sentiments
- DistilRoBERTa fine-tuning for accurate predictions
- Minority F1 optimization (not just overall accuracy)
- Adaptive thresholds handle imbalanced datasets

### Comprehensive Analysis
- Auto-detects text and rating columns
- Balances classes with quantile-based splitting
- Visualizes confusion matrices and F1 scores
- Exports all findings for further analysis

---

## How to Run

### Option 1: Google Colab (Recommended)
1. Click [Open in Colab](https://colab.research.google.com/github/ahmedElmersawy/StudentFeedbackMinorityDetection/blob/main/StudentFeedback_Analysis_Pipeline.ipynb)
2. Run all cells sequentially (Cell 1 through Cell 26)
3. Upload your own `studentdataset.csv` or use the provided sample

### Option 2: Local Installation
```bash
