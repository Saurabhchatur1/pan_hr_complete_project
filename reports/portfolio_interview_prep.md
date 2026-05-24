# Portfolio & Interview Preparation
## PAN HR Analytics — Career Progression & Promotion Gap Analysis

---

## ✅ ATS-Friendly Resume Bullet Points

**Data Science / ML Engineer Resume**

- Built end-to-end HR analytics ML system (Python, scikit-learn, Streamlit) analyzing 1,470 employees across 31 dimensions, identifying 5 distinct career archetypes with 24% attrition delta across risk segments
- Engineered 15 custom domain-specific features (Promotion Gap Ratio, Career Velocity Score, Burnout Risk Proxy, Engagement Composite) using pandas/numpy; increased clustering silhouette score vs. raw features
- Applied K-Means clustering with PCA dimensionality reduction; identified High-Risk Stagnation segment (n=167) showing 49% above-average attrition, enabling targeted $3M+ annual retention savings
- Designed weighted composite Risk Scoring Engine with three-tier classification (Low/Medium/High), stratifying 471 medium-risk employees for proactive HR intervention
- Deployed production-ready 7-page Streamlit dashboard with interactive filters, Plotly visualizations, and CSV export; structured as modular OOP codebase with sklearn pipelines
- Containerized application using Docker with GitHub Actions CI/CD pipeline; documented deployment guides for Streamlit Cloud, Render, AWS, and GCP
- Generated research-grade findings: Sales dept attrition 20.6% vs. R&D baseline 13.8%; overtime employees exit at 2.9× rate; Year-0 attrition at 36% — all with actionable intervention strategies

---

## 📋 LinkedIn Project Description

**Career Progression & Promotion Gap Analysis — HR Intelligence System**
*Palo Alto Networks Dataset · Python · Machine Learning · Streamlit*

Designed and built a complete AI-powered HR analytics platform that identifies employee career stagnation, promotion gaps, and retention risks using unsupervised machine learning.

Starting from a 1,470-employee dataset with 31 features, I engineered 15 advanced domain-specific features capturing career velocity, burnout risk, engagement health, and promotion momentum — then applied K-Means clustering to segment the workforce into 5 distinct archetypes.

Key findings: the High-Risk Stagnation cluster showed 24% attrition (49% above baseline), Sales department attrition reached 20.6%, and overtime employees departed at nearly 3× the rate of non-overtime peers.

Delivered a production-ready 7-page Streamlit dashboard with executive KPIs, cluster visualization, promotion gap analysis, at-risk employee tables, and department benchmarking — fully containerized with Docker and CI/CD via GitHub Actions.

**Tech stack**: Python · Pandas · NumPy · Scikit-Learn · Plotly · Streamlit · Docker · GitHub Actions

---

## 🎤 Top Interview Questions & STAR Answers

### Q1: "Walk me through your most complex data science project."

**Situation**: HR teams at large organizations struggle to identify which employees are at risk of leaving before it happens — attrition is typically analyzed only after the fact.

**Task**: Build a complete end-to-end analytics system that proactively identifies at-risk employees using career trajectory patterns and machine learning.

**Action**: I started by deeply understanding the business domain — reading HR literature on career plateauing and promotion fairness. I then engineered 15 custom features that no standard dataset provides: Promotion Gap Ratio, Career Velocity Score, Burnout Risk Proxy. Applied K-Means clustering to reveal 5 employee archetypes, built a weighted Risk Scoring Engine, and wrapped everything in a production Streamlit dashboard with 7 pages.

**Result**: Identified a High-Risk Stagnation cluster of 167 employees with 24% attrition — 49% above baseline. Created a business case for $3M+ annual retention savings. Built a fully deployable, documented system ready for portfolio presentation.

---

### Q2: "How did you choose which features to engineer?"

I applied a business-first approach rather than statistical-first. I asked: "What does an HR manager care about?" — they care about how long since someone was promoted, how fast they're advancing relative to peers, how engaged they feel, how burned out they might be. From those questions, I mapped each concept to measurable proxies from the available columns. For example, Burnout Risk = weighted combination of Overtime + Business Travel + Distance From Home. Each feature was validated against attrition correlation before inclusion.

---

### Q3: "Why K-Means and not a supervised model?"

We have no labeled "about to leave" data — only retrospective attrition flags. A supervised model would only learn who already left, not who is at risk now. K-Means allows us to discover natural groupings based on career trajectory patterns without labels. The resulting clusters have interpretable business meaning (Fast-Track Performers, Career Explorers, etc.) that an HR manager can act on immediately — something a black-box classifier score cannot provide.

---

### Q4: "How would you productionize this further?"

Three main steps: First, integrate a live HRIS data feed (Workday API or BambooHR) so the dashboard refreshes automatically. Second, add an MLflow experiment tracker to monitor feature drift and retrain the clustering model quarterly. Third, build a Slack/email alerting layer that notifies HR business partners when an employee's composite risk score crosses a threshold. This moves the system from a dashboard to an active intervention engine.

---

### Q5: "How did you measure business impact?"

I used industry-standard replacement cost estimates (1.5× median salary = ~$85K per departure) applied to the 237 annual attrition cases, yielding a $20.1M annual cost. A conservative 15% reduction from targeted interventions on the High-Risk and Medium-Risk segments yields $3M in savings. I also identified specific high-value interventions — addressing overtime, fixing year-0 onboarding, creating Sales promotion roadmaps — each with estimated impact percentages backed by the data.

---

## 📊 Business Impact One-Liner (for interviews)

"I built an ML-powered HR system that identified 167 employees with 24% attrition risk — 49% above baseline — creating a business case for $3M+ in annual retention savings through targeted career interventions."

---

## 🔑 Key Technical Talking Points

| Topic | Your Answer |
|-------|-------------|
| Why 5 clusters? | Elbow method + silhouette analysis; k=5 maximized interpretability with silhouette=0.154 |
| Why StandardScaler? | Features have vastly different scales (years vs. dollars); k-means is distance-based so scaling is essential |
| Why PCA for viz only? | PCA loses feature interpretability needed for business insights; used only for 2D cluster visualization |
| How did you validate clusters? | Silhouette score, attrition rate differentials across clusters (13%–24%), business interpretation by HR domain |
| What would improve the model? | More recent data, NLP on performance reviews, manager network graph features |
