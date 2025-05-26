import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def load_and_prepare_data(train_path, test_path):
    """
    Load and prepare training and test datasets with correct encodings
    """
    # Load datasets with appropriate encodings
    train_df = pd.read_csv(train_path, encoding='utf-8')
    test_df = pd.read_csv(test_path, encoding='cp1252')
    
    # Clean positions in training data
    train_df['Position'] = train_df['Pos'].apply(lambda x: str(x).split(',')[0].strip())
    train_df = train_df[train_df['Position'].isin(['DF', 'MF', 'FW', 'GK'])]
    
    # Get common numeric columns
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    common_cols = train_cols.intersection(test_cols)
    
    # Filter out non-numeric and special columns
    exclude_cols = {'Player', 'Pos', 'Nation', 'Squad', 'Position'}
    numeric_common_cols = [col for col in common_cols 
                          if col not in exclude_cols 
                          and pd.api.types.is_numeric_dtype(train_df[col])]
    
    print(f"\nNumber of common numeric features: {len(numeric_common_cols)}")
    print("\nFirst few common features:")
    for col in sorted(numeric_common_cols)[:5]:
        print(f"- {col}")
    
    # Prepare feature matrices
    X_train = train_df[numeric_common_cols]
    y_train = train_df['Position']
    X_test = test_df[numeric_common_cols]
    
    # Print shapes for verification
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    return X_train, y_train, X_test, numeric_common_cols

def create_pipeline(feature_cols):
    """
    Create preprocessing and model pipeline with the same parameters as original
    """
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ]), feature_cols)
    ])
    
    # Use the same parameters as in your original code
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=8,
        min_samples_leaf=2,
        max_features='log2',
        max_samples=0.8,
        ccp_alpha=0.0,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

def get_top_features(model, feature_names, n_features=20):
    """
    Get top n most important features from trained model
    """
    importances = model.named_steps['classifier'].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 20 features and their importance scores:")
    for i in range(n_features):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    return [feature_names[i] for i in indices[:n_features]]

def main():
    # Load and prepare data
    print("Loading and preparing data...")
    X_train, y_train, X_test, feature_cols = load_and_prepare_data(
        'merged_premier_league_stats.csv',
        'Challenge_2_data.csv'
    )
    
    print(f"\nInitial number of features: {len(feature_cols)}")
    print("\nTraining class distribution:")
    print(y_train.value_counts())
    
    # Train initial model to get feature importance
    print("\nTraining initial model for feature selection...")
    initial_pipeline = create_pipeline(feature_cols)
    initial_pipeline.fit(X_train, y_train)
    
    # Get top 20 features
    print("\nSelecting top 20 features...")
    top_features = get_top_features(initial_pipeline, feature_cols, n_features=20)
    
    # Train final model with only top features
    print("\nTraining final model with top 20 features...")
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]
    
    final_pipeline = create_pipeline(top_features)
    final_pipeline.fit(X_train_top, y_train)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = final_pipeline.predict(X_test_top)
    
    # Create output DataFrame and save
    output_df = pd.DataFrame({'Pos': predictions})
    output_df.to_csv('Challenge2_teamname1.csv', index=False)
    print("\nPredictions saved to 'Challenge2_teamname.csv'")
    
    # Print class distribution in predictions
    print("\nPredicted class distribution:")
    print(pd.Series(predictions).value_counts())

if __name__ == "__main__":
    main()