# 🔐 Career Progression & Promotion Gap Analysis
### HR Intelligence Platform — Palo Alto Networks Workforce Dataset

> An industry-grade, end-to-end HR Analytics system that identifies career stagnation, promotion gaps, employee trajectory patterns, and retention risks using advanced machine learning — built for real business impact.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Key Metrics](#-key-metrics)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Key Business Insights](#-key-business-insights)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Deployment](#-deployment)
- [Tech Stack](#-tech-stack)
- [Contact](#-contact)

---

## 📊 Project Overview

This project analyzes **1,470 Palo Alto Networks employees** to uncover hidden patterns in career progression, promotion fairness, and attrition risk. It combines feature engineering, unsupervised machine learning, and an interactive Streamlit dashboard to give HR teams and executives actionable, data-driven insights.

**What this system solves:**
- Who is at risk of leaving — and why?
- Which employees are stagnating despite high performance?
- Where are promotion gaps most critical?
- How do managers impact team retention?

---

## 📈 Key Metrics

| Metric | Value |
|--------|-------|
| Dataset Size | 1,470 Employees |
| Overall Attrition Rate | **16.1%** |
| Average Promotion Gap | **2.2 years** |
| Employees in Stagnation (index > 50) | **29.2%** |
| Medium-Risk Employees | **471** |
| ML Silhouette Score | **0.154** |
| Employee Clusters Identified | **5 distinct segments** |
| High-Risk Stagnation Segment | **167 employees (24% attrition)** |

---

## 🗂️ Project Structure

```
pan_hr/
├── data/
│   ├── pan_employees.csv          # Raw employee dataset
│   └── processed.csv              # Feature-engineered dataset
├── src/
│   ├── feature_engineering.py     # 15 advanced HR features
│   ├── clustering.py              # K-Means + PCA pipeline
│   └── preprocessing.py           # Sklearn preprocessing pipeline
├── notebooks/
│   └── 01_EDA.ipynb               # Exploratory Data Analysis
├── models/                        # Serialized model artifacts (.pkl)
├── reports/
│   ├── research_paper.md          # Full research documentation
│   └── portfolio_interview_prep.md
├── configs/
│   └── config.yaml                # Project configuration
├── deployment/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/
│   └── test_features.py           # Unit tests
├── .github/
│   └── workflows/ci_cd.yml        # GitHub Actions CI/CD
├── app.py                         # Streamlit application entry point
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Saurabhchatur1/pan_hr_complete_project.git
cd pan_hr_complete_project
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```bash
streamlit run app.py
```
Visit **http://localhost:8501** in your browser.

### 5. Docker (Optional)
```bash
docker build -t pan-hr-analytics -f deployment/Dockerfile .
docker run -p 8501:8501 pan-hr-analytics
```

---

## 🤖 Machine Learning Pipeline

### Feature Engineering — 15 Advanced HR Features

| Feature | Formula | Business Insight |
|---------|---------|-----------------|
| Promotion Gap Ratio | `YearsSincePromotion / (YearsAtCompany + 1)` | How overdue is a promotion |
| Role Stagnation Index | `YearsInCurrentRole / (TotalWorkingYears + 1)` | Career mobility indicator |
| Career Velocity Score | `JobLevel / (TotalWorkingYears + 1)` | Speed of advancement |
| Burnout Risk Proxy | `0.4×OT + 0.3×Travel + 0.3×Distance` | Work-life stress composite |
| Engagement Composite | Weighted avg of 4 satisfaction scores | Overall engagement health |

### K-Means Clustering Results (k=5)

| Cluster | Size | Attrition Rate | Profile |
|---------|------|---------------|---------|
| 🟢 Fast-Track Performers | 463 | 13% | High velocity, highly engaged |
| 🔵 Stable Contributors | 389 | 15% | Consistent, satisfied workers |
| 🟣 Career Explorers | 297 | 19% | Transitioning, seeking growth |
| 🔴 High-Risk Stagnation | 167 | 24% | Urgent intervention required |
| 🟡 Promotion-Stalled | 154 | 17% | Qualified but overlooked |

> PCA reduced features to 2D for visualization. Silhouette Score: **0.154**

---

## 💡 Key Business Insights

1. **Sales attrition at 20.6%** — highest across all departments; driven by promotion gaps and overtime pressure
2. **Overtime employees leave at 2.4× the base rate** — urgent workload redistribution needed
3. **Year 0–1 is the most critical retention window** — 34–36% attrition in the first year alone
4. **167 employees in High-Risk Stagnation** — 49% above baseline attrition; immediate HR action required
5. **Fast-Track Performers are retention anchors** — protect with stretch assignments and equity incentives

---

## 📱 Streamlit Dashboard Pages

| Page | Description |
|------|-------------|
| Executive Summary | KPIs, attrition trends, risk distribution overview |
| Career Path Analysis | Trajectory scatter plots, parallel category flows |
| Promotion Gap Dashboard | Distribution analysis, role-level breakdown |
| Retention Risk Matrix | At-risk employee table with exportable filters |
| Department Analytics | Benchmarking across R&D, Sales, and HR |
| Employee Explorer | Filterable employee table with CSV export |
| ML Cluster Analysis | PCA visualization, cluster profiles, segment insights |

---

## ☁️ Deployment Options

| Platform | Method |
|----------|--------|
| Streamlit Cloud | Connect GitHub repo → auto-deploy |
| Docker | `docker build & run` (see Quick Start) |
| Render | Add `render.yaml` + Procfile |
| AWS EC2 | Docker on t3.medium instance |
| Azure | Container Instances |
| GCP | Cloud Run |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| ML & Data | Pandas, NumPy, Scikit-Learn |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| DevOps | Docker, GitHub Actions |
| Config | YAML, python-decouple |

---

## 📄 Research Abstract

This study develops a multi-layered HR analytics system using unsupervised machine learning to identify employee career stagnation patterns, promotion gaps, and retention risks within a 1,470-employee dataset representative of Palo Alto Networks' workforce composition. We engineer 15 domain-specific features capturing career velocity, burnout risk, promotion momentum, and engagement health. K-Means clustering with PCA dimensionality reduction reveals five distinct employee archetypes with statistically significant attrition differentials (13%–24% across clusters). The High-Risk Stagnation segment (n=167) exhibits 24% attrition — 49% above baseline — and represents the primary retention intervention target.

---

## 💼 Portfolio Summary

Developed an end-to-end HR analytics ML system for workforce retention optimization using Python, Scikit-Learn, and Streamlit. Engineered 15 domain-specific features capturing career velocity, burnout risk, and promotion momentum. Applied K-Means clustering + PCA to segment 1,470 employees into 5 career archetypes with a 24% attrition delta across risk tiers. Deployed an interactive multi-page Streamlit dashboard with real-time filters and executive-level KPIs.

**Tech**: Python · Pandas · Scikit-Learn · Plotly · Streamlit · Docker · GitHub Actions

---

## 📞 Contact

**Built by:** Saurabh Chatur
**GitHub:** [github.com/Saurabhchatur1](https://github.com/Saurabhchatur1)
**LinkedIn:** [linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)

