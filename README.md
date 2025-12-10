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