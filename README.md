# 🔐 Career Progression & Promotion Gap Analysis
## Palo Alto Networks — HR Intelligence Platform

> **Industry-grade HR Analytics system** identifying career stagnation, promotion gaps, employee trajectory patterns, retention opportunities, and managerial impact using advanced machine learning.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/sklearn-1.4-orange.svg)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://docker.com)

---

## 📊 Project Overview

This end-to-end HR analytics project analyzes **1,470 Palo Alto Networks employees** to uncover:

| Metric | Value |
|--------|-------|
| Overall Attrition Rate | **16.1%** |
| Avg Promotion Gap | **2.2 years** |
| Employees in Stagnation (>50 index) | **29.2%** |
| Medium-Risk Employees | **471** |
| ML Silhouette Score | **0.154** |
| Employee Clusters | **5 distinct segments** |

---

## 🗂️ Project Structure

```
pan_hr/
├── data/
│   ├── pan_employees.csv          # Raw dataset
│   └── processed.csv              # Feature-engineered dataset
├── src/
│   ├── feature_engineering.py     # 15 advanced HR features
│   ├── clustering.py              # K-Means + PCA pipeline
│   └── preprocessing.py           # sklearn pipeline
├── notebooks/
│   └── 01_EDA.ipynb               # Exploratory Data Analysis
├── models/                        # Serialized model artifacts
├── deployment/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/
│   └── test_features.py
├── .github/workflows/ci_cd.yml    # GitHub Actions
├── app.py                         # Streamlit application
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Local Development
```bash
# Clone the repository
git clone https://github.com/yourusername/pan-hr-analytics.git
cd pan-hr-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

### Docker
```bash
docker build -t pan-hr-analytics -f deployment/Dockerfile .
docker run -p 8501:8501 pan-hr-analytics
# Visit http://localhost:8501
```

---

## 🤖 Machine Learning

### Feature Engineering (15 Advanced Features)

| Feature | Formula | Business Logic |
|---------|---------|----------------|
| Promotion Gap Ratio | `YearsSincePromotion / (YearsAtCompany + 1)` | How overdue is a promotion |
| Role Stagnation Index | `YearsInCurrentRole / (TotalWorkingYears + 1)` | Career mobility indicator |
| Career Velocity Score | `JobLevel / (TotalWorkingYears + 1)` | Speed of advancement |
| Burnout Risk Proxy | `0.4×OT + 0.3×Travel + 0.3×Distance` | Work-life stress composite |
| Engagement Composite | Weighted avg of 4 satisfaction scores | Overall engagement health |

### Clustering Results (K-Means, k=5)

| Cluster | Size | Attrition | Description |
|---------|------|-----------|-------------|
| 🟢 Fast-Track Performers | 463 | 13% | High velocity, high engagement |
| 🔵 Stable Contributors | 389 | 15% | Consistent, satisfied workers |
| 🟣 Career Explorers | 297 | 19% | Transitioning, exploring growth |
| 🔴 High-Risk Stagnation | 167 | 24% | Urgent intervention needed |
| 🟡 Promotion-Stalled | 154 | 17% | Qualified but overlooked |

---

## 🏢 Key Business Insights

1. **Sales attrition at 20.6%** — highest of all departments; driven by promotion gaps and overtime
2. **Overtime employees leave at 2.4× base rate** — workload redistribution required urgently
3. **Year 0-1 is the most critical window** — 34-36% attrition in first year
4. **167 employees in High-Risk Stagnation** — 24% attrition, immediate HR intervention needed
5. **Fast-Track Performers are retention anchors** — protect with stretch assignments + equity

---

## 📱 Streamlit App Pages

- **Executive Summary** — KPIs, attrition trends, risk distribution
- **Career Path Analysis** — trajectory scatter, parallel categories flow
- **Promotion Gap Dashboard** — distribution, role-level breakdown
- **Retention Risk** — risk matrix, at-risk employee table
- **Department Analytics** — benchmarking across R&D, Sales, HR
- **Employee Explorer** — filterable table with CSV export
- **ML Cluster Analysis** — PCA visualization, cluster profiles

---

## 🔧 Deployment Options

| Platform | Method |
|----------|--------|
| Streamlit Cloud | Connect GitHub repo → auto-deploy |
| Docker | `docker build & run` |
| Render | `render.yaml` (add Procfile) |
| AWS EC2 | Docker on t3.medium |
| Azure | Container Instances |
| GCP | Cloud Run |

---

## 📄 Research Summary

**Abstract**: This study develops a multi-layered HR analytics system using unsupervised machine learning to identify employee career stagnation patterns, promotion gaps, and retention risks within a 1,470-employee dataset representative of Palo Alto Networks' workforce composition. We engineer 15 domain-specific features capturing career velocity, burnout risk, promotion momentum, and engagement health. K-Means clustering with PCA dimensionality reduction reveals five distinct employee archetypes with statistically significant attrition differentials (13%–24% across clusters). The High-Risk Stagnation segment (n=167) exhibits 24% attrition — 49% above baseline — and represents the primary retention intervention target.

---

## 💼 Resume / Portfolio

**Developed** an end-to-end HR analytics ML system for workforce retention optimization using Python, scikit-learn, and Streamlit; engineered 15 domain-specific features capturing career velocity, burnout risk, and promotion momentum; applied K-Means clustering + PCA to segment 1,470 employees into 5 career archetypes with 24% attrition delta across risk tiers; deployed interactive multi-page Streamlit dashboard with real-time filters and executive-level KPIs.

**Tech**: Python · Pandas · Scikit-Learn · Plotly · Streamlit · Docker · GitHub Actions

---

## 📞 Contact

Built by: [Your Name]  
LinkedIn: linkedin.com/in/yourprofile  
GitHub: github.com/yourusername
