import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from xgboost import XGBClassifier

# Load the datasets
training_data = pd.read_csv('merged_dataset.csv')
challenge_data = pd.read_csv('Challenge_1_Ingame_data_updated.csv')

# Simple features for FTR
ftr_features = ['HST', 'AST', 'HS', 'AS', 'HC', 'AC']

# Advanced features for HTR
def create_advanced_features(df):
    df = df.copy()
    
    # Shot efficiency and quality
    df['home_shooting_efficiency'] = np.where(df['HS'] > 0, df['HST'] / df['HS'], 0)
    df['away_shooting_efficiency'] = np.where(df['AS'] > 0, df['AST'] / df['AS'], 0)
    df['home_shot_quality'] = df['HST'] * df['home_shooting_efficiency']
    df['away_shot_quality'] = df['AST'] * df['away_shooting_efficiency']
    
    # Attack pressure
    df['home_attack_pressure'] = df['HS'] + 2*df['HST'] + df['HC']
    df['away_attack_pressure'] = df['AS'] + 2*df['AST'] + df['AC']
    
    return df

# Create features
training_data = create_advanced_features(training_data)
challenge_data = create_advanced_features(challenge_data)

htr_features = ftr_features + [
    'home_shooting_efficiency', 'away_shooting_efficiency',
    'home_shot_quality', 'away_shot_quality',
    'home_attack_pressure', 'away_attack_pressure'
]

# Prepare data
X_ftr = training_data[ftr_features]
X_htr = training_data[htr_features]
y_ftr = training_data['FTR']
y_htr = training_data['HTR']
y_goals = training_data[['FTHG', 'FTAG', 'HTHG', 'HTAG']]

# Label Encoding
le_ftr = LabelEncoder()
le_htr = LabelEncoder()
y_ftr_encoded = le_ftr.fit_transform(y_ftr)
y_htr_encoded = le_htr.fit_transform(y_htr)

# Split data
X_ftr_train, X_ftr_test, y_ftr_train, y_ftr_test = train_test_split(
    X_ftr, y_ftr_encoded, test_size=0.2, random_state=42
)

X_htr_train, X_htr_test, y_htr_train, y_htr_test = train_test_split(
    X_htr, y_htr_encoded, test_size=0.2, random_state=42
)

_, _, _, y_goals_test = train_test_split(
    X_ftr, y_goals, test_size=0.2, random_state=42
)

# Scale features
ftr_scaler = StandardScaler()
htr_scaler = StandardScaler()

X_ftr_train_scaled = ftr_scaler.fit_transform(X_ftr_train)
X_ftr_test_scaled = ftr_scaler.transform(X_ftr_test)

X_htr_train_scaled = htr_scaler.fit_transform(X_htr_train)
X_htr_test_scaled = htr_scaler.transform(X_htr_test)

# 1. Simple Full-time Result Model
ftr_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
ftr_model.fit(X_ftr_train_scaled, y_ftr_train)

# 2. Enhanced Half-time Result Model
htr_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
htr_model.fit(X_htr_train_scaled, y_htr_train)

# 3. Goals Ensemble
class GoalsEnsemble:
    def __init__(self):
        self.rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
        self.gb = GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42)
        
    def fit(self, X, y):
        self.rf.fit(X, y)
        self.gb.fit(X, y)
        
    def predict(self, X):
        rf_pred = self.rf.predict(X)
        gb_pred = self.gb.predict(X)
        ensemble_pred = (rf_pred + gb_pred) / 2
        return np.round(ensemble_pred)

# Train goal models
goal_models = {
    'FTHG': GoalsEnsemble(),
    'FTAG': GoalsEnsemble(),
    'HTHG': GoalsEnsemble(),
    'HTAG': GoalsEnsemble()
}

for col in ['FTHG', 'FTAG', 'HTHG', 'HTAG']:
    goal_models[col].fit(X_ftr_train_scaled, y_goals[col].iloc[X_ftr_train.index])

# Make predictions
ftr_pred_encoded = ftr_model.predict(X_ftr_test_scaled)
htr_pred_encoded = htr_model.predict(X_htr_test_scaled)

ftr_pred = le_ftr.inverse_transform(ftr_pred_encoded)
htr_pred = le_htr.inverse_transform(htr_pred_encoded)

# Goals predictions
goals_pred = {col: model.predict(X_ftr_test_scaled) 
              for col, model in goal_models.items()}

# Print performance metrics
print("\nFull-time Result Performance:")
print(classification_report(le_ftr.inverse_transform(y_ftr_test), ftr_pred))
print("\nFTR Confusion Matrix:")
print(confusion_matrix(le_ftr.inverse_transform(y_ftr_test), ftr_pred))

print("\nHalf-time Result Performance:")
print(classification_report(le_htr.inverse_transform(y_htr_test), htr_pred))
print("\nHTR Confusion Matrix:")
print(confusion_matrix(le_htr.inverse_transform(y_htr_test), htr_pred))

print("\nGoals Prediction Performance:")
for col in ['FTHG', 'FTAG', 'HTHG', 'HTAG']:
    print(f"\n{col}:")
    print("RMSE:", np.sqrt(mean_squared_error(y_goals_test[col], goals_pred[col])))

# Make predictions for challenge set
X_challenge_ftr = challenge_data[ftr_features]
X_challenge_htr = challenge_data[htr_features]

X_challenge_ftr_scaled = ftr_scaler.transform(X_challenge_ftr)
X_challenge_htr_scaled = htr_scaler.transform(X_challenge_htr)

# Predict and decode results
challenge_ftr = le_ftr.inverse_transform(ftr_model.predict(X_challenge_ftr_scaled))
challenge_htr = le_htr.inverse_transform(htr_model.predict(X_challenge_htr_scaled))

challenge_goals = {col: model.predict(X_challenge_ftr_scaled) 
                  for col, model in goal_models.items()}

challenge_predictions = pd.DataFrame({
    'FTR': challenge_ftr,
    'HTR': challenge_htr,
    'FTHG': challenge_goals['FTHG'],
    'FTAG': challenge_goals['FTAG'],
    'HTHG': challenge_goals['HTHG'],
    'HTAG': challenge_goals['HTAG']
})

# Save predictions
challenge_predictions.to_csv('hybrid_match_predictions.csv', index=False)

print("\nSample of Challenge Predictions (first 5 rows):")
print(challenge_predictions.head())

# Print feature importance for both models
print("\nFTR Feature Importance:")
ftr_importance = pd.DataFrame({
    'Feature': ftr_features,
    'Importance': ftr_model.feature_importances_
})
print(ftr_importance.sort_values('Importance', ascending=False))

print("\nHTR Feature Importance:")
htr_importance = pd.DataFrame({
    'Feature': htr_features,
    'Importance': htr_model.feature_importances_
})
print(htr_importance.sort_values('Importance', ascending=False))