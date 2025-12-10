# Policy Optimization for Financial Decision-Making: Deep Learning vs. Offline RL

**Author:** Hammad Shaikh  
**Assessment:** Shodh AI - Machine Learning Internship Task

---

## 📌 Executive Summary
This project explores the transition from **predicting credit risk** to **optimizing financial returns**. Using the LendingClub dataset, I developed and compared two distinct AI systems:
1.  **Supervised Deep Learning (DL):** A classifier optimized to predict default probability.
2.  **Offline Reinforcement Learning (RL):** A Conservative Q-Learning (CQL) agent optimized to maximize portfolio value.

**Core Finding:** While both models beat the human baseline, the **Optimized DL Model** proved superior. By strictly tuning the decision threshold to `0.25`, it reduced portfolio losses by **85%**, whereas the RL agent achieved a **26%** reduction.

---

## 📊 Key Results

| Strategy | Primary Metric | Approval Rate | Est. Policy Value (EPV) | Net Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Human Baseline** | N/A | 100% | **-$1,805** per loan | - |
| **RL Agent (CQL)** | EPV | 95.2% | **-$1,339** per loan | +26% |
| **DL Model (Tuned)** | AUC / F1 | 65.6% | **-$273** per loan | **+85%** |

---

## 📂 Repository Structure

```text
├── models/
│   └── cql_loan_agent.pt            # Trained Offline RL Agent (Weights)
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb         # Data cleaning, leakage removal, and scaling
│   ├── 02_deep_learning.ipynb              # Training the MLP & Threshold Optimization
│   ├── 03_offline_rl.ipynb                 # Reward engineering & Training CQL Agent
│   └── 04_analysis_and_comparison.ipynb    # EPV calculation & Divergent Case Analysis
│
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
└── Hammad_Shaikh_Report_Shodh_AI.pdf # Final detailed report

⚙️ Installation & Setup

1. Clone the Repository
git clone [https://github.com/Hammad-1105/Loan-Approval-Shodh-AI.git](https://github.com/Hammad-1105/Loan-Approval-Shodh-AI.git)
cd Loan-Approval-Shodh-AI

2. Set up the Environment
It is recommended to use a virtual environment (Conda or venv).
# Create environment
conda create -n shodh_ai python=3.9
conda activate shodh_ai

# Install dependencies
pip install -r requirements.txt

3. Data Setup
Download the LendingClub Dataset (specifically accepted_2007_to_2018Q4.csv) from Kaggle.

Place the CSV file inside a data/ folder in the root directory.

🚀 How to Run the Code
To reproduce the findings, run the notebooks in the following order:

1. Data Preprocessing (01_data_preprocessing.ipynb)

Goal: Prepare the raw financial data.

Process: Removes leakage columns (e.g., recoveries), handles missing data (>30% threshold), and performs feature selection.

Output: data/processed_loans.csv.

2. Deep Learning Model (02_deep_learning.ipynb)

Goal: Train a predictive classifier.

Process: Trains a Multi-Layer Perceptron (MLP) using PyTorch. Performs a grid search to find the optimal decision threshold (0.25) that maximizes the F1-score.

Output: AUC/F1 metrics and ROC plots.

3. Offline RL Training (03_offline_rl.ipynb)

Goal: Train a policy to optimize rewards.

Process: Augments the dataset with counterfactual "Deny" actions (Reward = $0). Trains a Conservative Q-Learning (CQL) agent using d3rlpy.

Output: Trained agent saved to models/cql_loan_agent.pt.

4. Analysis & Comparison (04_analysis_and_comparison.ipynb)

Goal: Compare the two approaches.

Process: Loads both models and evaluates them on the held-out test set. Calculates Estimated Policy Value (EPV) and identifies divergent cases (where DL says "No" but RL says "Yes").

Output: Final comparison tables and financial metrics.