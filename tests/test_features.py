"""Unit tests for feature engineering"""
import pytest
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from feature_engineering import engineer_features

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Age': [30], 'Attrition': [0], 'BusinessTravel': ['Travel_Rarely'],
        'DailyRate': [800], 'Department': ['Sales'], 'DistanceFromHome': [5],
        'Education': [3], 'EducationField': ['Marketing'], 'EnvironmentSatisfaction': [3],
        'Gender': ['Male'], 'HourlyRate': [60], 'JobInvolvement': [3], 'JobLevel': [2],
        'JobRole': ['Sales Executive'], 'JobSatisfaction': [3], 'MaritalStatus': ['Single'],
        'MonthlyIncome': [5000], 'MonthlyRate': [15000], 'NumCompaniesWorked': [2],
        'OverTime': ['No'], 'PercentSalaryHike': [14], 'PerformanceRating': [3],
        'RelationshipSatisfaction': [3], 'StockOptionLevel': [1], 'TotalWorkingYears': [8],
        'TrainingTimesLastYear': [3], 'WorkLifeBalance': [3], 'YearsAtCompany': [5],
        'YearsInCurrentRole': [3], 'YearsSinceLastPromotion': [2], 'YearsWithCurrManager': [3]
    })

def test_feature_count(sample_df):
    result = engineer_features(sample_df)
    new_cols = set(result.columns) - set(sample_df.columns)
    assert len(new_cols) >= 15

def test_promotion_gap_ratio(sample_df):
    result = engineer_features(sample_df)
    expected = 2 / (5 + 1)
    assert abs(result['PromotionGapRatio'].iloc[0] - expected) < 0.001

def test_risk_tier_exists(sample_df):
    result = engineer_features(sample_df)
    assert 'RiskTier' in result.columns
    assert result['RiskTier'].iloc[0] in ['Low Risk', 'Medium Risk', 'High Risk']

def test_burnout_nonnegative(sample_df):
    result = engineer_features(sample_df)
    assert result['BurnoutRiskProxy'].iloc[0] >= 0

def test_no_nulls_in_engineered(sample_df):
    result = engineer_features(sample_df)
    engineered = ['PromotionGapRatio', 'RoleStagnationIndex', 'CareerVelocityScore',
                  'BurnoutRiskProxy', 'EngagementCompositeScore']
    for col in engineered:
        assert result[col].isna().sum() == 0, f"{col} has nulls"
