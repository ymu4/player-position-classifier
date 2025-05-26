import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def create_tuning_pipeline(feature_cols):
    """
    Create pipeline with hyperparameter search space
    """
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ]), feature_cols)
    ])
    
    # Define the parameter search space
    param_distributions = {
        'classifier__n_estimators': [100, 200, 300, 400],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 4, 6, 8],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', None],  # None means all features
        'classifier__max_samples': [0.7, 0.8, 0.9, None],
        'classifier__class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    base_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    # Create RandomizedSearchCV object
    search = RandomizedSearchCV(
        base_pipeline,
        param_distributions=param_distributions,
        n_iter=100,  # Number of parameter settings sampled
        cv=5,        # 5-fold cross-validation
        scoring='balanced_accuracy',
        random_state=42,
        n_jobs=-1,
        verbose=2
    )
    
    return search

def load_and_prepare_data(train_path, test_path):
    """
    Load and prepare training and test datasets
    """
    train_df = pd.read_csv(train_path, encoding='utf-8')
    test_df = pd.read_csv(test_path, encoding='cp1252')
    
    train_df['Position'] = train_df['Pos'].apply(lambda x: str(x).split(',')[0].strip())
    train_df = train_df[train_df['Position'].isin(['DF', 'MF', 'FW', 'GK'])]
    
    # Get common numeric columns
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    common_cols = train_cols.intersection(test_cols)
    
    exclude_cols = {'Player', 'Pos', 'Nation', 'Squad', 'Position'}
    numeric_common_cols = [col for col in common_cols 
                          if col not in exclude_cols 
                          and pd.api.types.is_numeric_dtype(train_df[col])]
    
    X = train_df[numeric_common_cols]
    y = train_df['Position']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, numeric_common_cols

def print_feature_importances(model, feature_names, top_n=20):
    """
    Print top N feature importances from the trained model
    """
    # Get feature importances from the best model
    importances = model.named_steps['classifier'].feature_importances_
    feature_importance_pairs = list(zip(feature_names, importances))
    sorted_pairs = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_n} Feature importances:")
    for feature, importance in sorted_pairs[:top_n]:
        print(f"{feature}: {importance:.4f}")
    
    # Plot feature importances
    plt.figure(figsize=(12, 6))
    features, values = zip(*sorted_pairs[:top_n])
    plt.barh(range(len(features)), values)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.close()

def main():
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, feature_cols = load_and_prepare_data(
        'merged_premier_league_stats.csv',
        'Challenge_2_data.csv'
    )
    
    print("\nStarting hyperparameter tuning...")
    search = create_tuning_pipeline(feature_cols)
    search.fit(X_train, y_train)
    
    # Print best parameters and score
    print("\nBest parameters found:")
    for param, value in search.best_params_.items():
        print(f"{param}: {value}")
    print(f"\nBest cross-validation score: {search.best_score_:.4f}")
    
    # Evaluate on test set
    y_pred = search.predict(X_test)
    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, y_pred))
    
    # Print and plot feature importances
    print_feature_importances(search.best_estimator_, feature_cols)
    
    return search.best_params_

if __name__ == "__main__":
    best_params = main()