import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
#import GridSearch

from sklearn.model_selection import GridSearchCV

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectFromModel
import joblib

def load_and_merge_data(file_path):
    """
    Load and merge all sheets from Premier League stats Excel file
    """
    sheets = pd.read_excel(file_path, sheet_name=None, na_values=['', ' ', 'NA', 'N/A'])
    processed_sheets = []
    
    for sheet_name, df in sheets.items():
        print(f"\nProcessing sheet: {sheet_name}")
        print(f"Initial shape: {df.shape}")
        sheet_df = df.copy()
        
        # Basic cleaning
        sheet_df['Player'] = sheet_df['Player'].str.strip()
        sheet_df['Squad'] = sheet_df['Squad'].str.strip()
        sheet_df = sheet_df[~(sheet_df['Player'].isin(['Player', '']) |
                            sheet_df['Squad'].isin(['Squad', '']))]
        
        # Handle 90s column
        if '90s' in sheet_df.columns:
            sheet_df.rename(columns={'90s': 'Minutes_90s'}, inplace=True)
            sheet_df['Minutes_90s'] = pd.to_numeric(sheet_df['Minutes_90s'], errors='coerce')
        
        # Create unique identifier and handle column naming
        sheet_df['Player_ID'] = sheet_df.apply(
            lambda x: f"{x['Player']}_{x['Squad']}_{x['Rk']}", axis=1)
        prefix = sheet_name.split(' ')[0]
        base_cols = ['Player', 'Squad', 'Nation', 'Pos', 'Rk', 'Player_ID', 'Minutes_90s']
        cols_to_rename = [col for col in sheet_df.columns if col not in base_cols]
        sheet_df.rename(columns={col: f"{col}_{prefix}" for col in cols_to_rename},
                       inplace=True)
        
        print(f"Processed shape: {sheet_df.shape}")
        processed_sheets.append(sheet_df)
    
    # Merge all sheets
    final_df = processed_sheets[0]
    for i, right_df in enumerate(processed_sheets[1:], 1):
        right_cols = [col for col in right_df.columns if col not in final_df.columns]
        right_cols.append('Player_ID')
        right_df_slim = right_df[right_cols]
        print(f"\nMerging sheet {i+1}")
        print(f"Current shape: {final_df.shape}")
        final_df = pd.merge(
            final_df,
            right_df_slim,
            on='Player_ID',
            how='outer',
            validate='1:1'
        )
        print(f"After merge: {final_df.shape}")
    
    return final_df.drop('Player_ID', axis=1)

def clean_positions(df):
    """
    Clean position data for classification
    """
    df['Position'] = df['Pos'].apply(lambda x: str(x).split(',')[0].strip())
    return df[df['Position'].isin(['DF', 'MF', 'FW', 'GK'])]

def filter_playing_time(df, min_games=5):
    """
    Filter players based on minimum playing time
    """
    return df[df['Minutes_90s'] >= min_games]

def get_numeric_features(df):
    """
    Extract relevant numeric features with proper filtering
    """
    print(f"Initial column count: {len(df.columns)}")
    
    numeric_cols = []
    for col in df.columns:
        if col not in ['Player', 'Squad', 'Nation', 'Pos', 'Position']:
            try:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    continue
                df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
                numeric_cols.append(col)
            except:
                continue
    
    valid_cols = [col for col in numeric_cols if
                 col not in ['Rk', 'Born'] and
                 df[col].isnull().mean() < 0.5 and
                 df[col].nunique() > 1]
    
    print(f"Selected {len(valid_cols)} valid numeric features")
    print("\nExample of numeric features selected:")
    for col in valid_cols[:5]:
        print(f"- {col}")
    return valid_cols

def train_position_classifier_cv(X_train,  y_train, feature_cols):
    """
    Train and evaluate the position classifier using GridSearchCV with reduced parameter space
    """
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ]), feature_cols)
    ])

    # Define focused parameter grid (reduced but strategic values)
    param_grid = {
        'n_estimators': [150, 300],           # 3 values
        'max_depth': [7, 11, None],             # 4 values
        'min_samples_split': [ 8, 12],           # 3 values
        'min_samples_leaf': [ 4, 6],             # 3 values
        # 'max_features': ['sqrt', 'log2'],          # 2 values
        'max_samples': [ 0.9],            # 3 values
        'ccp_alpha': [0.0, 0.01]                   # 2 values
    }
    
    # Total combinations = 3 × 4 × 3 × 3 × 2 × 3 × 2 = 1,296 combinations

    # Initialize cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        Pipeline([
            ('preprocessor', preprocessor),
            ('feature_selector', SelectFromModel(
                RandomForestClassifier(n_estimators=200, random_state=42),
                max_features=80
            )),
            ('classifier', RandomForestClassifier(
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ]),
        param_grid={
            'classifier__' + key: value for key, value in param_grid.items()
        },
        cv=cv,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=3
    )

    print("Training models with grid search using 5-fold cross-validation...")
    print("Total parameter combinations to try: 1,296")
    print("Total model fits: 1,296 × 5 folds = 6,480")
    
    grid_search.fit(X_train, y_train)

    print("\nBest Cross-Validation Results:")
    print(f"Mean CV Score: {grid_search.best_score_:.4f}")
    print("\nBest Parameters Found:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")

    return grid_search.best_estimator_, grid_search.best_params_

def plot_model_results_cv(model, X_train, y_train, feature_names):
    """
    Create visualizations of model performance with cross-validation results
    """
    position_labels = sorted(np.unique(y_train))
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 2, height_ratios=[1, 1, 1])
    
    # Training confusion matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_train, model.predict(X_train), normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', ax=ax1,
               xticklabels=position_labels,
               yticklabels=position_labels)
    ax1.set_title('Training Confusion Matrix\n(Normalized by True Labels)')
    ax1.set_xlabel('Predicted Position')
    ax1.set_ylabel('True Position')
    
    # Cross-validation scores plot
    ax3 = fig.add_subplot(gs[1, :])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                              scoring='balanced_accuracy')
    x_pos = np.arange(len(cv_scores))
    ax3.bar(x_pos, cv_scores, yerr=0, capsize=5)
    ax3.axhline(y=cv_scores.mean(), color='r', linestyle='--',
                label=f'Mean CV Score: {cv_scores.mean():.3f}')
    ax3.set_xlabel('CV Fold')
    ax3.set_ylabel('Balanced Accuracy Score')
    ax3.set_title('Cross-validation Scores Across Folds')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'Fold {i+1}' for i in range(len(cv_scores))])
    ax3.legend()
    
    # Feature importance plot
    if hasattr(model, 'named_steps'):
        ax4 = fig.add_subplot(gs[2, :])
        feature_selector = model.named_steps['feature_selector']
        clf = model.named_steps['classifier']
        mask = feature_selector.get_support()
        selected_features = [f for f, m in zip(feature_names, mask) if m]
        importances = pd.DataFrame({
            'feature': selected_features,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        sns.barplot(data=importances.head(15), x='importance', y='feature', ax=ax4)
        ax4.set_title('Top 15 Most Important Features')
        ax4.set_xlabel('Feature Importance')
        ax4.set_ylabel('Feature Name')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load initial data
    df = load_and_merge_data('Premier League Players 23_24 Stats_train.xlsx')
    print("Merging all sheets\n")
    
    # Clean positions
    # df['Position'] = df['Pos'].apply(lambda x: str(x).split(',')[0].strip())
    df = df[df['Pos'].isin(['DF', 'MF', 'FW', 'GK',"MF,FW","FW,MF","DF,MF","MF,DF",])]
    
    # Apply preprocessing
    print("\nProcessing training data...")
    # df = filter_playing_time(df)
    
    # Get feature columns
    print("\nGetting features...")
    feature_cols = get_numeric_features(df)
    
    # Create feature sets
    X_train = df[feature_cols]
    y_train = df['Pos']
    
    print("\nFinal dataset dimensions:")
    print("Training set size:", X_train.shape[0])
    print("\nFeature dimensions:")
    print("Training features:", X_train.shape)
    print("\nClass distribution:")
    print(y_train.value_counts())
    
    # Train the model
    best_pipeline, best_params = train_position_classifier_cv(X_train, y_train, feature_cols)
    
    # Print the best parameters
    print("\nBest Parameters Found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # Get predictions for training set
    train_predictions = best_pipeline.predict(X_train)
    
    # Print training performance
    print("\nTraining Performance Report:")
    print(classification_report(y_train, train_predictions))
    
    # Save the model
    print("Saving the trained pipeline...")
    joblib.dump(best_pipeline, 'player_position_pipeline.pkl')
    joblib.dump(feature_cols, 'feature_columns.pkl')
    print("Model and features saved successfully!")
    
    # Plot results
    plot_model_results_cv(best_pipeline, X_train, y_train, feature_cols)