# Multilingual Toxic Comment Detection

This project presents a robust pipeline for **multilingual toxic comment classification** using both **deep learning** and **traditional machine learning** approaches. The system aims to accurately detect six types of offensive content with a key focus on improving performance for the **"toxic"** label through **threshold tuning** and **weighted sampling** strategies.

---

## Dataset Overview

The dataset contains the following:

- **Features**:
  - `id`: Unique identifier
  - `feedback_text`: User-generated comment
  - `lang`: Language of the text
  - **Label columns**: `toxic`, `abusive`, `vulgar`, `menace`, `offense`, `bigotry`
- **Test Set**:
  - Contains `id`, `content`, and `lang` (labels are withheld)
- **Languages**:
  - Multilingual (language preserved for model training)

---

## Problem Statement

Multilabel classification is complex due to:

-  **Severe class imbalance**, especially for toxic comments  
-  **Multilingual text noise** and inconsistent structures  
-  Need for **deep contextual understanding** to detect subtle toxicity  

---

## 🧪 Models and Results

###  1. Transformer-Based Model (`XLM-RoBERTa`)

Fine-tuned multilingual transformer using Hugging Face Transformers.

**Highlights**:
- `xlm-roberta-base` pre-trained model  
- **WeightedRandomSampler** for handling imbalance  
- **Threshold optimization** on logits to boost F1  
- **Early stopping** based on validation loss  

**Training & Validation Metrics (Toxic Label)**:

| Epoch | Train Loss | Val Loss | Precision | Recall | F1    |
|-------|------------|----------|-----------|--------|-------|
| 1     | 0.1747     | 0.0821   | 0.622     | 0.455  | 0.526 |
| 2     | 0.0898     | 0.0907   | 0.675     | 0.403  | 0.505 |
| 3     | 0.0473     | 0.1247   | 0.724     | 0.313  | 0.438 |

 **Best Threshold**: `-0.00`  
 **Best F1 Score (Toxic)**: **0.526**

---

### 2. GRU-Based Neural Network

Compact RNN-based model for multilingual classification.

**Architecture**:
- Embedding layer → GRU → Sigmoid output
- Trained using **Binary Cross Entropy**
- **WeightedRandomSampler** used to balance underrepresented toxic classes
- **Threshold search** applied for optimal F1 on the toxic label

**Performance (Toxic Label)**:

| Epoch | Train Loss | Val F1 |
|-------|------------|--------|
| 1     | 1.1687     | 0.3033 |
| 2     | 0.5155     | 0.3571 |
| 4     | 0.2191     | **0.3580**  |

 **Best Threshold**: `0.51`  
 **Best F1 Score**: **0.362**

---

### 📊 . Logistic Regression (TF-IDF)

Lightweight, interpretable baseline using TF-IDF + Logistic Regression.

**Highlights**:
- One-vs-Rest logistic regression classifier  
- TF-IDF for text representation  
- Evaluated on **toxic** label only  

**Validation Results (Toxic Label)**:

| Class       | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| Toxic (1)   | 0.18      | 0.80   | 0.29     | 134     |
| Non-toxic (0)| 0.89     | 0.30   | 0.45     | 705     |
| **Macro Avg** | 0.53    | 0.55   | 0.37     | 839     |

📝 **Comment**: High recall but low precision → many false positives.

---

###  4. Random Forest

Classic ensemble model using bagging.

**Validation Results (Toxic Label)**:

| Class       | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| Toxic (1)   | 0.29      | 0.27   | 0.28     | 134     |
| Non-toxic (0)| 0.86     | 0.87   | 0.87     | 705     |
| **Macro Avg** | 0.58    | 0.57   | 0.57     | 839     |

 **Comment**: Better balance across classes, but still struggles with toxic detection.

---

##  Final Comparison

| Model            | Best F1 (Toxic) | Precision (Toxic) | Recall (Toxic) | Strengths                            |
|------------------|------------------|--------------------|------------------|----------------------------------------|
| **XLM-RoBERTa**  | **0.526**        | 0.62 – 0.72        | 0.31 – 0.45      | Contextual modeling, multilingual understanding |
| GRU              | 0.358            | Moderate            | Moderate         | Lightweight, fair multilingual handling |
| Random Forest    | 0.28             | 0.29               | 0.27             | Handles non-toxic well, limited toxic generalization |
| Logistic Regression | 0.29          | 0.18               | **0.80**         | High recall but low precision (false positives) |

---

##  Future Improvements

- Advanced multilingual augmentation (back-translation, paraphrasing)  
- Larger transformer models (`xlm-roberta-large`, `mDeBERTa`)  
- Label-wise thresholding  
- Meta-learning or ensembling for robustness  


