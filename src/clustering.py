"""
Unsupervised ML — Employee Clustering
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

CLUSTER_LABELS = {
    0: "Fast-Track Performers",
    1: "Promotion-Stalled",
    2: "High-Risk Stagnation",
    3: "Stable Contributors",
    4: "Career Explorers"
}

CLUSTER_COLORS = {
    "Fast-Track Performers": "#22c55e",
    "Promotion-Stalled": "#f59e0b",
    "High-Risk Stagnation": "#ef4444",
    "Stable Contributors": "#3b82f6",
    "Career Explorers": "#a855f7"
}

FEATURE_COLS = [
    'PromotionGapRatio', 'RoleStagnationIndex', 'CareerVelocityScore',
    'PromotionMomentum', 'CareerSatisfactionIndex', 'EngagementCompositeScore',
    'BurnoutRiskProxy', 'CareerAccelerationScore', 'RetentionScore',
    'GrowthPotentialScore', 'ManagerStabilityIndicator'
]

def run_clustering(df: pd.DataFrame, n_clusters: int = 5):
    feats = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feats].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    df = df.copy()
    df['Cluster'] = labels
    df['ClusterLabel'] = df['Cluster'].map(CLUSTER_LABELS)

    # PCA for viz
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(Xs)
    df['PCA1'] = pca_coords[:, 0]
    df['PCA2'] = pca_coords[:, 1]

    sil = silhouette_score(Xs, labels)
    return df, sil, feats

