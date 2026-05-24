"""
Career Progression & Promotion Gap Analysis
Palo Alto Networks — HR Intelligence Platform
Streamlit Multi-Page Application
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from feature_engineering import engineer_features
from clustering import run_clustering, CLUSTER_COLORS

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PAN HR Intelligence",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0a0a0f; }
    [data-testid="stSidebar"] { background-color: #0f0f1a; border-right: 1px solid #1e1e2e; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d2d4e; border-radius: 12px;
        padding: 1.2rem; text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { color: #8888aa; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .danger { color: #ef4444; } .warn { color: #f59e0b; } .ok { color: #22c55e; }
    .section-header {
        color: #e2e8f0; font-size: 1.1rem; font-weight: 600;
        border-left: 3px solid #f59e0b; padding-left: 0.75rem; margin: 1rem 0 0.5rem;
    }
    div[data-testid="metric-container"] { background: #1a1a2e; border-radius: 8px; padding: 0.5rem; border: 1px solid #2d2d4e; }
    .stTabs [data-baseweb="tab"] { color: #8888aa; }
    .stTabs [aria-selected="true"] { color: #f59e0b; border-bottom: 2px solid #f59e0b; }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(__file__), 'data', 'pan_employees.csv')
    df = pd.read_csv(path)
    df = engineer_features(df)
    df, sil, feats = run_clustering(df, n_clusters=5)
    return df, sil

df_raw, silhouette = load_data()

# ─── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔐 PAN HR Intelligence")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Executive Summary",
        "📈 Career Path Analysis",
        "🎯 Promotion Gap Dashboard",
        "🔴 Retention Risk",
        "🏢 Department Analytics",
        "🤖 Employee Explorer",
        "📊 ML Cluster Analysis"
    ])
    st.markdown("---")
    st.markdown("**Filters**")
    dept_filter = st.multiselect("Department", df_raw['Department'].unique(), default=list(df_raw['Department'].unique()))
    risk_filter = st.multiselect("Risk Tier", ['Low Risk', 'Medium Risk', 'High Risk'], default=['Low Risk', 'Medium Risk', 'High Risk'])
    age_range = st.slider("Age Range", 18, 60, (18, 60))

df = df_raw[
    df_raw['Department'].isin(dept_filter) &
    df_raw['RiskTier'].isin(risk_filter) &
    df_raw['Age'].between(age_range[0], age_range[1])
].copy()

# ─── Helper ───────────────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0", family="Inter"),
    margin=dict(l=10, r=10, t=30, b=10)
)

def T(fig, height=None):
    """Apply dark theme to any plotly figure."""
    fig.update_layout(**PLOTLY_THEME)
    if height:
        fig.update_layout(height=height)
    return fig

# ══════════════════════════════════════════════════════════════════════════
# PAGE: EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════
if "Executive" in page:
    st.title("Executive Summary Dashboard")
    st.markdown(f"*Showing {len(df):,} of {len(df_raw):,} employees based on current filters*")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1: st.metric("Total Employees", f"{len(df):,}")
    with col2: st.metric("Attrition Rate", f"{df['Attrition'].mean()*100:.1f}%", delta=f"{(df['Attrition'].mean() - df_raw['Attrition'].mean())*100:.1f}% vs all")
    with col3: st.metric("Avg Promo Gap", f"{df['YearsSinceLastPromotion'].mean():.1f} yrs")
    with col4: st.metric("Stagnation Index", f"{(df['StagnationScore'] > 50).mean()*100:.0f}%")
    with col5: st.metric("Medium Risk", f"{(df['RiskTier']=='Medium Risk').sum():,}")
    with col6: st.metric("High Risk", f"{(df['RiskTier']=='High Risk').sum():,}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section-header">Attrition by Tenure</p>', unsafe_allow_html=True)
        at = df.groupby('YearsAtCompany')['Attrition'].mean().reset_index()
        fig = px.line(at, x='YearsAtCompany', y='Attrition', markers=True,
                      title="", color_discrete_sequence=['#f59e0b'])
        fig.update_layout(yaxis_tickformat='.0%', height=280)
        fig = T(fig)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<p class="section-header">Risk Tier Distribution</p>', unsafe_allow_html=True)
        rd = df['RiskTier'].value_counts().reset_index()
        fig = px.pie(rd, values='count', names='RiskTier',
                     color='RiskTier', color_discrete_map={'Low Risk':'#22c55e','Medium Risk':'#f59e0b','High Risk':'#ef4444'},
                     hole=0.55)
        fig.update_layout(height=280)
        fig = T(fig)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<p class="section-header">Monthly Income by Job Level</p>', unsafe_allow_html=True)
        sl = df.groupby('JobLevel')['MonthlyIncome'].mean().reset_index()
        fig = px.bar(sl, x='JobLevel', y='MonthlyIncome', color='MonthlyIncome',
                     color_continuous_scale='Blues')
        fig.update_layout(height=250, showlegend=False)
        fig = T(fig)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        st.markdown('<p class="section-header">Cluster Distribution</p>', unsafe_allow_html=True)
        cl = df['ClusterLabel'].value_counts().reset_index()
        fig = px.bar(cl, x='ClusterLabel', y='count',
                     color='ClusterLabel', color_discrete_map=CLUSTER_COLORS)
        fig.update_layout(height=250, showlegend=False, xaxis_tickangle=-20)
        fig = T(fig)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE: CAREER PATH
# ══════════════════════════════════════════════════════════════════════════
elif "Career Path" in page:
    st.title("Career Path Analysis")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(df.sample(min(500,len(df))), x='TotalWorkingYears', y='MonthlyIncome',
                         color='JobLevel', size='PercentSalaryHike', hover_data=['JobRole','Department'],
                         title="Career Trajectory: Experience vs Income", color_continuous_scale='Viridis')
        fig = T(fig)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(df, x='JobLevel', y='YearsSinceLastPromotion', color='Department',
                     title="Promotion Gap by Level & Department")
        fig = T(fig)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.parallel_categories(
        df[['Department','JobRole','RiskTier','Attrition']].sample(min(300,len(df))),
        dimensions=['Department','JobRole','RiskTier'],
        color=df['Attrition'].sample(min(300,len(df))),
        color_continuous_scale=['#22c55e','#ef4444'],
        title="Career Flow: Department → Role → Risk"
    )
    fig = T(fig)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE: PROMOTION GAP
# ══════════════════════════════════════════════════════════════════════════
elif "Promotion Gap" in page:
    st.title("Promotion Gap Dashboard")
    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Avg Promo Gap", f"{df['YearsSinceLastPromotion'].mean():.1f} yrs")
    with c2: st.metric("Max Promo Gap", f"{df['YearsSinceLastPromotion'].max()} yrs")
    with c3: st.metric("% Stagnant >3yrs", f"{(df['YearsSinceLastPromotion']>3).mean()*100:.1f}%")

    c1,c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x='YearsSinceLastPromotion', color='Attrition',
                           barmode='overlay', nbins=16,
                           title="Promotion Gap Distribution by Attrition",
                           color_discrete_map={0:'#3b82f6', 1:'#ef4444'})
        fig = T(fig)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        role_promo = df.groupby('JobRole')['YearsSinceLastPromotion'].mean().sort_values(ascending=True).reset_index()
        fig = px.bar(role_promo, y='JobRole', x='YearsSinceLastPromotion', orientation='h',
                     title="Avg Promotion Gap by Role", color='YearsSinceLastPromotion',
                     color_continuous_scale='RdYlGn_r')
        fig = T(fig)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.violin(df, x='Department', y='YearsSinceLastPromotion', color='Department',
                    box=True, points='outliers', title="Promotion Gap Distribution by Department")
    fig = T(fig)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE: RETENTION RISK
# ══════════════════════════════════════════════════════════════════════════
elif "Retention" in page:
    st.title("Retention Risk Dashboard")
    c1,c2 = st.columns(2)
    with c1:
        fig = px.scatter(df.sample(min(500,len(df))), x='RetentionScore', y='StagnationScore',
                         color='RiskTier', size='YearsSinceLastPromotion',
                         hover_data=['JobRole','Department','MonthlyIncome'],
                         color_discrete_map={'Low Risk':'#22c55e','Medium Risk':'#f59e0b','High Risk':'#ef4444'},
                         title="Retention Risk Matrix")
        fig = T(fig)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        ot = df.groupby('OverTime')['Attrition'].mean().reset_index()
        fig = px.bar(ot, x='OverTime', y='Attrition', color='OverTime',
                     title="Attrition Rate: Overtime vs Not",
                     color_discrete_map={'Yes':'#ef4444','No':'#22c55e'})
        fig.update_layout(yaxis_tickformat='.0%')
        fig = T(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Top 20 at-risk employees
    st.markdown('<p class="section-header">Top 25 Highest Risk Employees</p>', unsafe_allow_html=True)
    risk_top = df[['JobRole','Department','Age','MonthlyIncome','YearsSinceLastPromotion',
                   'RetentionScore','StagnationScore','RiskTier','ClusterLabel']]\
        .sort_values('RetentionScore', ascending=False).head(25)
    st.dataframe(risk_top.style.background_gradient(subset=['RetentionScore','StagnationScore'], cmap='RdYlGn_r'),
                 use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE: DEPARTMENTS
# ══════════════════════════════════════════════════════════════════════════
elif "Department" in page:
    st.title("Department Analytics")
    dept_summary = df.groupby('Department').agg(
        Employees=('Attrition','count'), AttritionRate=('Attrition','mean'),
        AvgIncome=('MonthlyIncome','mean'), AvgPromoGap=('YearsSinceLastPromotion','mean'),
        AvgSatisfaction=('JobSatisfaction','mean'), AvgStagnation=('StagnationScore','mean')
    ).reset_index().round(2)
    st.dataframe(dept_summary, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        fig = px.bar(dept_summary, x='Department', y='AttritionRate',
                     color='AttritionRate', color_continuous_scale='RdYlGn_r',
                     title="Attrition Rate by Department")
        fig.update_layout(yaxis_tickformat='.1%')
        fig = T(fig)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(df, x='EngagementCompositeScore', y='MonthlyIncome',
                         color='Department', facet_col='Department',
                         title="Engagement vs Income by Department")
        fig = T(fig)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE: EMPLOYEE EXPLORER
# ══════════════════════════════════════════════════════════════════════════
elif "Employee Explorer" in page:
    st.title("Employee Explorer")
    search_cols = ['JobRole','Department','MaritalStatus','OverTime','BusinessTravel']
    filters = {col: st.selectbox(col, ['All'] + list(df[col].unique())) for col in search_cols}
    df_view = df.copy()
    for col, val in filters.items():
        if val != 'All': df_view = df_view[df_view[col] == val]
    st.dataframe(df_view[['Age','JobRole','Department','MonthlyIncome','YearsSinceLastPromotion',
                            'ClusterLabel','RiskTier','RetentionScore','StagnationScore',
                            'CareerVelocityScore','BurnoutRiskProxy']].reset_index(drop=True),
                 use_container_width=True)
    csv = df_view.to_csv(index=False).encode()
    st.download_button("⬇ Download Filtered Report", csv, "filtered_employees.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════
# PAGE: ML CLUSTER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
elif "ML Cluster" in page:
    st.title("ML Cluster Analysis")
    st.metric("Silhouette Score", f"{silhouette:.3f}", help="Higher is better (range -1 to 1)")

    fig = px.scatter(df.sample(min(600,len(df))), x='PCA1', y='PCA2',
                     color='ClusterLabel', color_discrete_map=CLUSTER_COLORS,
                     hover_data=['JobRole','Department','MonthlyIncome','YearsSinceLastPromotion'],
                     title="Employee Clusters — PCA 2D Projection", size_max=8)
    fig = T(fig)
    st.plotly_chart(fig, use_container_width=True)

    cluster_profile = df.groupby('ClusterLabel').agg(
        Count=('Attrition','count'), AttritionRate=('Attrition','mean'),
        AvgSalary=('MonthlyIncome','mean'), AvgPromoGap=('YearsSinceLastPromotion','mean'),
        AvgCareerVelocity=('CareerVelocityScore','mean'), AvgBurnout=('BurnoutRiskProxy','mean'),
        AvgEngagement=('EngagementCompositeScore','mean')
    ).reset_index().round(3)
    st.dataframe(cluster_profile, use_container_width=True)

    fig = px.bar(cluster_profile, x='ClusterLabel', y=['AttritionRate','AvgCareerVelocity','AvgBurnout'],
                 barmode='group', title="Cluster Comparison: Key Metrics",
                 color_discrete_sequence=['#ef4444','#22c55e','#f59e0b'])
    fig = T(fig)
    st.plotly_chart(fig, use_container_width=True)