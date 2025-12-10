# Policy Optimization for Financial Decision-Making: Deep Learning vs. Offline RL

**Author:** Hammad Shaikh  
**Assessment:** Shodh AI - Machine Learning Internship Task

---

## 📌 Executive Summary

This project explores the transition from **predicting credit risk** to **optimizing financial returns**. Using the LendingClub dataset, I developed and compared two distinct AI systems:

1. **Supervised Deep Learning (DL):** Predicts default probability.  
2. **Offline Reinforcement Learning (RL):** Conservative Q-Learning (CQL) agent that optimizes portfolio value.

### 🔍 Core Finding

The **Optimized Deep Learning Model** outperformed all other policies:

- DL reduced losses by **85%**  
- RL reduced losses by **26%**

This demonstrates that a well-tuned classifier can outperform offline RL when the deny action is safe and well-defined.

---

## 📊 Key Results

| Strategy | Primary Metric | Approval Rate | EPV (Est. Policy Value) | Net Improvement |
|---------|----------------|----------------|---------------------------|-----------------|
| **Human Baseline** | N/A | 100% | **–$1,805** | – |
| **RL Agent (CQL)** | EPV | 95.2% | **–$1,339** | **+26%** |
| **DL Model (Tuned)** | AUC / F1 | 65.6% | **–$273** | **+85%** |

---

## 📂 Repository Structure
```text
├── models/
│   └── cql_loan_agent.pt                 # Trained RL Agent
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb       # Cleaning, leakage removal, scaling
│   ├── 02_deep_learning.ipynb            # MLP training & threshold optimization
│   ├── 03_offline_rl.ipynb               # Reward engineering & CQL training
│   └── 04_analysis_and_comparison.ipynb  # Final EPV & policy comparison
│
├── requirements.txt
├── README.md
└── Hammad_Shaikh_Report_Shodh_AI.pdf
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Hammad-1105/Loan-Approval-Shodh-AI.git
cd Loan-Approval-Shodh-AI
```

### 2️⃣ Create environment (conda or venv)

**Conda:**
```bash
conda create -n shodh_ai python=3.9
conda activate shodh_ai
```

**Venv:**
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📁 Data Setup

**Download:**
```text
accepted_2007_to_2018.csv
```

**Place it in:**
```text
data/accepted_2007_to_2018.csv
```

---

## 🚀 How to Run the Code (Reproducibility Steps)

Run notebooks in this exact order:

### ✅ 1. Data Preprocessing

**📌 `01_data_preprocessing.ipynb`**

- Removes leakage features
- Removes missing-heavy columns
- Encodes categories, scales numerics
- Performs correlation filtering

**Output:** `cleaned_data.csv`

---

### ✅ 2. Deep Learning Model

**📌 `02_deep_learning.ipynb`**

- Trains PyTorch MLP (256→128→64)
- Handles class imbalance
- Tunes threshold to 0.25

**Reproduced Metrics:**

- AUC = 0.741
- F1 = 0.456
- EPV ≈ –$273

---

### ✅ 3. Offline RL Training (CQL)

**📌 `03_offline_rl.ipynb`**

- Adds counterfactual Deny actions
- Defines reward model
- Trains CQL via d3rlpy

**Reproduced Metrics:**

- Approval Rate: 95.2%
- EPV: –$1,339

---

### ✅ 4. Analysis & Comparison

**📌 `04_analysis_and_comparison.ipynb`**

- Computes EPV for all policies
- Compares DL vs RL
- Extracts divergence cases
- Recreates Applicant #45 case

---

## 📌 Reproducibility Notes

Running the four notebooks sequentially will reproduce:

- All metrics in the report
- EPV values
- Threshold tuning behavior
- RL agent decisions
- Divergent case study outputs

---