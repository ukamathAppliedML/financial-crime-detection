# 🛡️ Financial Crime Detection Framework

A **generic package** for detecting financial crime activities in communications and transactions.

---

## 🚀 1. Framework Overview

This package provides a unified framework for training and deploying ML models to detect various financial crimes. 

### 🔍 1.1 Current Implementation

- **Anti-Money Laundering (AML):** Production-ready detection system  
- **Dataset:** 726 labeled examples with detailed AML risk categories  
- **Performance:** High accuracy with robust evaluation metrics  

### 🧩 1.2 Framework Extensibility

Easily extend the architecture to support:

- Bribery & Corruption Detection  
- Fraud Detection  
- Custom Financial Crime Types  

---

## 🏗️ 2. Technical Architecture

### ⚙️ 2.1 Core Components

- `FinCrimeModel`: Training and model management  
- `FinCrimePredictor`: Inference and prediction  
- `FinCrimeEvaluator`: Evaluation and analysis  
- `utils`: Data loading, preprocessing, and analysis  

### 🧠 2.2 Model Architecture

- **Base Model:** XLM-RoBERTa (multilingual transformer)  
- **Task:** Binary sequence classification  
- **Labels:** `0 = Normal`, `1 = Suspicious`  
- **Input:** Financial communication text  
- **Output:** Prediction + confidence score  

### 📂 2.3 Dataset Format

CSV structure required:

```csv
conversation,label,category
"Financial communication text",0,"Category Name"
"Suspicious activity text",1,"Risk Category"
```

---

## ✨ 3. Key Features

### ⚙️ 3.1 Generic Framework

- ✅ **Modular Design** — Easily extend for new crime types  
- ✅ **Consistent API** — Common interface across all types  
- ✅ **Configurable** — Flexible training/inference parameters  

### 🔄 3.2 Comprehensive Workflow

- 📊 **Data Loading & Analysis**  
- 🔄 **Data Preparation** (train/val/test split)  
- 🤖 **Model Training** (transformer-based)  
- 📈 **Evaluation** (metrics + visualizations)  
- 🧪 **Prediction** (confidence score included)  
- 🔍 **Misclassification Analysis**  

### 🧰 3.3 Interactive Tools

- 📓 Jupyter Notebook demo  
- 💻 Command-line training/eval scripts  
- 🔌 API-ready components  

---

## ⚡ 4. Installation & Quick Start

### ✅ Option 1: Quick Start

```bash
# Clone the repository
git clone https://github.com/smarsh/financial-crime-detection.git
cd financial-crime-detection

# Create virtual environment (Python 3.10 or 3.11)
python3.10 -m venv venv
source venv/bin/activate

# Run setup and demo
python quick_start.py
```

### 🛠 Option 2: Manual Installation

```bash
python3.10 -m venv venv
source venv/bin/activate

pip install torch torchvision torchaudio
pip install -r requirements.txt

# Test installation
python example_usage.py
```

### 📓 Option 3: Interactive Demo

```bash
jupyter notebook
# Open: Financial_Crime_Detection_Demo.ipynb
```

---

## 🧪 5. Usage Examples

### 🔹 5.1 Basic Prediction

```python
from src.predictor import FinCrimePredictor

predictor = FinCrimePredictor('path/to/model')

# Single prediction
result = predictor.predict("Financial communication text")
print(f"Prediction: {result['label']} (Confidence: {result['confidence']:.3f})")

# Batch predictions
texts = ["Normal transaction", "Suspicious activity"]
results = predictor.predict_batch(texts)
```

### 🔹 5.2 Model Training

```python
from src.model import FinCrimeModel
from src.utils import load_fincrime_dataset, prepare_data_splits

df = load_fincrime_dataset('data/your_dataset.csv')
X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(df)

model = FinCrimeModel()
model.train(X_train, y_train, X_val, y_val, X_test, y_test)
model.save_model('./trained_model')
```

### 🔹 5.3 Model Evaluation

```python
from src.evaluator import FinCrimeEvaluator

evaluator = FinCrimeEvaluator(predictor)
evaluator.print_evaluation_summary(X_test, y_test)
evaluator.plot_confusion_matrix(X_test, y_test)
```

---

## 🛠 6. Training Configuration

### 🧾 6.1 Model Parameters

- **Epochs:** 6 (tunable)  
- **Batch Size:** 16 (GPU) / 8 (CPU)  
- **Learning Rate:** Adaptive (AdamW optimizer)  
- **Validation:** Epoch-based with early stopping  

### 🔄 6.2 Data Splits

- Training: 80%  
- Validation: 10%  
- Test: 10%  
- Stratified sampling for balanced classes  

---

## 📓 7. Interactive Demo Notebook

### [`AML_Demo.ipynb`](./AML_Demo.ipynb)

Includes:

- 📊 Full workflow from data loading to prediction  
- 🤖 Interactive training with configuration options  
- 📈 Visual evaluation metrics  
- 🧪 Real-time predictions with confidence  
- 🔍 Misclassification analysis  

To run:

```bash
pip install jupyter notebook
jupyter notebook
```

> Works with your own dataset or built-in sample data for demo purposes.

## 📚 8. Domain-Specific Documentation

### 🧾 Anti-Money Laundering (AML)

📄 [**AntiMoneyLaundering_README.md**](./AntiMoneyLaundering_README.md)

Includes:

- AML risk categories & detection logic  
- Dataset structure & labeled examples  
- Compliance insights and evaluation