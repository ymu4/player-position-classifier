import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def merge_player_sheets(sheets):
    """
    Merge multiple sheets of player data into a single DataFrame
    
    Parameters:
    sheets (dict): Dictionary of DataFrames from Excel sheets
    
    Returns:
    DataFrame: Merged player data
    """
    processed_sheets = []
    
    for sheet_name, df in sheets.items():
        sheet_df = df.copy()
        
        # Basic cleaning
        for col in ['Player', 'Squad', 'Pos']:
            sheet_df[col] = sheet_df[col].str.strip()
        
        # Remove header rows
        sheet_df = sheet_df[~((sheet_df['Player'] == 'Player') | 
                             (sheet_df['Player'].isna()) |
                             (sheet_df['Squad'] == 'Squad') |
                             (sheet_df['Pos'] == 'Pos'))]
        
        # Keep valid positions
        sheet_df = sheet_df[sheet_df['Pos'].apply(lambda x: any(pos in str(x) for pos in ['DF', 'MF', 'FW', 'GK']))]
        
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
        
        processed_sheets.append(sheet_df)
    
    # Merge all sheets
    merged_df = processed_sheets[0]
    for right_df in processed_sheets[1:]:
        right_cols = [col for col in right_df.columns if col not in merged_df.columns]
        right_cols.append('Player_ID')
        right_df_slim = right_df[right_cols]
        merged_df = pd.merge(
            merged_df,
            right_df_slim,
            on='Player_ID',
            how='outer',
            validate='1:1'
        )
    
    return merged_df

def test_position_classifier(test_file_path):
    """
    Test the trained position classifier on new data
    
    Parameters:
    test_file_path (str): Path to the test Excel file
    
    Returns:
    tuple: (predictions, actual_labels, performance_report)
    """
    # Load the trained pipeline and feature columns
    try:
        pipeline = joblib.load('player_position_pipeline.pkl')
    except Exception:
        import sklearn
        sklearn.set_config(transform_output="default")
        pipeline = joblib.load('player_position_pipeline.pkl', mmap_mode=None)
    
    feature_cols = joblib.load('feature_columns.pkl')
    
    # Load and preprocess test data
    sheets = pd.read_excel(test_file_path, sheet_name=None,)
    test_df = merge_player_sheets(sheets)
    
    # Clean and prepare data
    test_df = test_df.dropna(subset=['Pos'])
    test_df = test_df.drop_duplicates(subset=['Player', 'Squad', 'Pos'])
    test_df['Original_Position'] = test_df['Pos']
    test_df['Position'] = test_df['Pos'].apply(lambda x: str(x).strip())
    
    # Prepare features
    X_test = test_df[feature_cols].copy()
    for col in X_test.columns:
        if not pd.api.types.is_datetime64_any_dtype(X_test[col]):
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    
    # Make predictions
    test_predictions = pipeline.predict(X_test)
    
    # Calculate and display performance metrics
    performance_report = classification_report(
        test_df['Position'],
        test_predictions,
        zero_division=0
    )
    print("\nTest Set Performance Report:")
    print(performance_report)
    
    # Plot confusion matrix
    all_positions = sorted(list(set(test_df['Position'].unique()) | set(test_predictions)))
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(
        test_df['Position'],
        test_predictions,
        normalize='true'
    )
    
    sns.heatmap(cm, 
                annot=True, 
                fmt='.2%', 
                cmap='Blues',
                xticklabels=all_positions,
                yticklabels=all_positions)
    plt.title('Test Set Confusion Matrix\n(Normalized by True Labels)')
    plt.xlabel('Predicted Position')
    plt.ylabel('True Position')
    plt.tight_layout()
    plt.show()
    
    return test_predictions, test_df['Position'], performance_report

if __name__ == "__main__":
    test_file_path = 'Premier League Players 23_24 Stats_test.xlsx'
    predictions, actual_labels, report = test_position_classifier(test_file_path)