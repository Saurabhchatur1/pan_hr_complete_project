"""
Advanced Feature Engineering for HR Analytics
Palo Alto Networks — Career Progression & Promotion Gap Analysis
"""
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Promotion Gap Ratio — how long since last promotion relative to tenure
    df['PromotionGapRatio'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)

    # 2. Role Stagnation Index — years in current role vs total working years
    df['RoleStagnationIndex'] = df['YearsInCurrentRole'] / (df['TotalWorkingYears'] + 1)

    # 3. Training Intensity Score — training normalized by years at company
    df['TrainingIntensityScore'] = df['TrainingTimesLastYear'] / (df['YearsAtCompany'] + 1)

    # 4. Manager Stability Indicator — years with manager vs years at company
    df['ManagerStabilityIndicator'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)

    # 5. Career Velocity Score — job level per years of total experience
    df['CareerVelocityScore'] = df['JobLevel'] / (df['TotalWorkingYears'] + 1)

    # 6. Promotion Momentum — inverse of promotion gap (higher = more promotions)
    df['PromotionMomentum'] = 1 / (df['YearsSinceLastPromotion'] + 1)

    # 7. Compensation Growth Proxy — income per job level
    df['CompensationGrowthProxy'] = df['MonthlyIncome'] / (df['JobLevel'] * 1000 + 1)

    # 8. Career Satisfaction Index — composite of job satisfaction + involvement
    df['CareerSatisfactionIndex'] = (df['JobSatisfaction'] + df['JobInvolvement'] + df['WorkLifeBalance']) / 3

    # 9. Organizational Stability Score — avg of env satisfaction + relationship satisfaction
    df['OrgStabilityScore'] = (df['EnvironmentSatisfaction'] + df['RelationshipSatisfaction']) / 2

    # 10. Employee Development Score — training + percent hike + performance
    df['EmployeeDevelopmentScore'] = (
        df['TrainingTimesLastYear'] * 0.3 +
        df['PercentSalaryHike'] * 0.4 +
        df['PerformanceRating'] * 0.3
    )

    # 11. Retention Opportunity Index — higher = more at risk
    df['RetentionOpportunityIndex'] = (
        df['PromotionGapRatio'] * 0.35 +
        df['RoleStagnationIndex'] * 0.25 +
        (1 - df['CareerSatisfactionIndex'] / 4) * 0.4
    )

    # 12. Leadership Exposure Score — job level + involvement + stock options
    df['LeadershipExposureScore'] = (
        df['JobLevel'] * 0.5 +
        df['JobInvolvement'] * 0.3 +
        df['StockOptionLevel'] * 0.2
    )

    # 13. Engagement Composite Score
    df['EngagementCompositeScore'] = (
        df['JobSatisfaction'] * 0.3 +
        df['EnvironmentSatisfaction'] * 0.25 +
        df['RelationshipSatisfaction'] * 0.25 +
        df['WorkLifeBalance'] * 0.2
    )

    # 14. Burnout Risk Proxy — overtime + travel + distance
    travel_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
    ot_map = {'Yes': 1, 'No': 0}
    df['_travel_num'] = df['BusinessTravel'].map(travel_map).fillna(1)
    df['_ot_num'] = df['OverTime'].map(ot_map).fillna(0)
    df['BurnoutRiskProxy'] = (
        df['_ot_num'] * 0.4 +
        (df['_travel_num'] / 2) * 0.3 +
        (df['DistanceFromHome'] / 29) * 0.3
    )
    df.drop(columns=['_travel_num', '_ot_num'], inplace=True)

    # 15. Career Acceleration Score — salary hike * performance / stagnation
    df['CareerAccelerationScore'] = (
        df['PercentSalaryHike'] * df['PerformanceRating']
    ) / (df['YearsSinceLastPromotion'] + 1)

    # === Risk Scoring Engine ===
    # Promotion Gap Score (0–100)
    df['PromotionGapScore'] = np.clip(
        df['YearsSinceLastPromotion'] / 15 * 100, 0, 100
    )

    # Retention Opportunity Score (0–100)
    df['RetentionScore'] = np.clip(df['RetentionOpportunityIndex'] * 100, 0, 100)

    # Career Stagnation Score
    df['StagnationScore'] = np.clip(df['RoleStagnationIndex'] * 100, 0, 100)

    # Growth Potential Score (0–100, higher = more potential)
    df['GrowthPotentialScore'] = np.clip(
        (df['EmployeeDevelopmentScore'] / df['EmployeeDevelopmentScore'].max()) * 100, 0, 100
    )

    # Managerial Influence Score
    df['ManagerialInfluenceScore'] = np.clip(df['ManagerStabilityIndicator'] * 100, 0, 100)

    # Composite Risk Tier
    df['CompositeRisk'] = (
        df['PromotionGapScore'] * 0.3 +
        df['RetentionScore'] * 0.4 +
        df['StagnationScore'] * 0.3
    )
    # REPLACE with this (add .astype(str) at the end):
    df['RiskTier'] = pd.cut(
       df['CompositeRisk'],
       bins=[0, 33, 66, 100],
       labels=['Low Risk', 'Medium Risk', 'High Risk'],
       include_lowest=True
).astype(str)

    return df

