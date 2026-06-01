# 🛡️ Travel Insurance Prediction — EDA & Preprocessing

> **ML Lab Case Study | MVSR Engineering College**
> Subject: Machine Learning LAB (U21PC682IT) | Year: 2025-2026

---

## 📌 Overview

This project performs end-to-end **data preprocessing** and **exploratory data analysis (EDA)** on a real-world travel insurance dataset containing 63,326 transaction records. The goal is to identify patterns and key features that predict whether a customer will file an insurance claim.

---

## 📂 Project Structure

```
travel-insurance-eda/
│
├── travel_insurance.csv       # Raw dataset
├── travel_insurance_eda.py    # Main analysis script
├── README.md                  # You're here
│
└── figures/                   # Auto-generated on running the script
    ├── fig1_claim_distribution.png
    ├── fig2_age_distribution.png
    ├── fig3_duration_boxplot.png
    ├── fig4_netsales_boxplot.png
    ├── fig5_agency_channel.png
    ├── fig6_correlation_heatmap.png
    └── fig7_destination_claimrate.png
```

---

## 📊 Dataset

| Attribute | Value |
|---|---|
| File | `travel_insurance.csv` |
| Raw Rows | 63,326 |
| Rows after cleaning | 54,632 |
| Columns | 11 |
| Target Variable | `Claim` (Yes / No) |
| Problem Type | Binary Classification |
| Claim Rate | ~1.67% — severe class imbalance |

**Features:** Agency, Agency Type, Distribution Channel, Product Name, Duration, Destination, Net Sales, Commision (in value), Gender, Age

---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/your-username/travel-insurance-eda.git
cd travel-insurance-eda
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run the script
```bash
python travel_insurance_eda.py
```

All 7 figures will be saved as `.png` files in the same directory and displayed inline.

---

## 🔍 What the Script Does

| Section | Description |
|---|---|
| **1 — Load** | Reads CSV, prints shape, column names, dtypes, first 5 rows |
| **2 — Audit** | Checks missing values per column and duplicate row count |
| **3 — Preprocessing** | Deduplication → outlier removal → null fill → label encoding → standard scaling |
| **4 — Stats** | Descriptive statistics + Pearson correlation with target variable |
| **5 — EDA (7 Figures)** | Visual analysis across claim distribution, age, duration, net sales, agency, channel, heatmap, destinations |
| **6 — Product Analysis** | Claim rate breakdown by insurance product name |
| **7 — Summary** | Final console printout of all key findings |

---

## 📈 Key Findings

- **1.67% claim rate** — dataset is heavily imbalanced; SMOTE or class weighting needed for modelling
- **Net Sales** is the strongest numerical predictor of claims (`+0.133` correlation)
- **Trip Duration** shows positive association with claims (`+0.070`) — longer trips = higher risk
- **Annual plans** (Annual Silver, Annual Gold) have claim rates of **10–12%**, roughly **6–7× the average**
- **Singapore** is the highest-volume high-risk destination (556 claims, 4.8% rate)
- **Age** shows negligible correlation (`−0.016`) — not a meaningful predictor

---

## 📦 Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## 🏫 Academic Info

| Field | Detail |
|---|---|
| College | MVSR Engineering College, Hyderabad |
| Department | Information Technology |
| Subject | Machine Learning LAB (U21PC682IT) |
| Assessment | Internal Assessment — Case Study |
| Faculty | P. Sita Sowjanya |
| Academic Year | 2025-2026 |

---

## 📄 License

This project is submitted as part of academic coursework. Dataset sourced from [Kaggle — Travel Insurance](https://www.kaggle.com/datasets/mhdzahier/travel-insurance).
