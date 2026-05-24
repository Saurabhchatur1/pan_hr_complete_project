"""
Production-Ready Preprocessing Pipeline
Palo Alto Networks HR Analytics
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

CAT_COLS = ['BusinessTravel', 'Department', 'EducationField', 'Gender',
            'JobRole', 'MaritalStatus', 'OverTime']
NUM_COLS = ['Age', 'DailyRate', 'DistanceFromHome', 'Education',
            'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
            'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
            'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
            'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
            'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
            'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

ENGINEERED_FEATURES = [
    'PromotionGapRatio', 'RoleStagnationIndex', 'TrainingIntensityScore',
    'ManagerStabilityIndicator', 'CareerVelocityScore', 'PromotionMomentum',
    'CompensationGrowthProxy', 'CareerSatisfactionIndex', 'OrgStabilityScore',
    'EmployeeDevelopmentScore', 'RetentionOpportunityIndex', 'LeadershipExposureScore',
    'EngagementCompositeScore', 'BurnoutRiskProxy', 'CareerAccelerationScore'
]

def build_preprocessor():
    """Build sklearn ColumnTransformer pipeline for model-ready data."""
    num_pipeline = Pipeline([('scaler', StandardScaler())])
    cat_pipeline = Pipeline([('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, NUM_COLS + ENGINEERED_FEATURES),
        ('cat', cat_pipeline, CAT_COLS)
    ], remainder='drop')
    return preprocessor

def prepare_data(df: pd.DataFrame, for_clustering: bool = True):
    """Full preprocessing pipeline. Returns X_scaled, feature_names."""
    from feature_engineering import engineer_features
    df = engineer_features(df)
    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(df)
    return X, preprocessor

