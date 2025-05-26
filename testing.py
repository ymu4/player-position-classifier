import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def merge_player_sheets(sheets):
    """
    Merge multiple player statistics sheets into a single dataframe
    
    Parameters:
    sheets (dict): Dictionary of dataframes from Excel sheets
    
    Returns:
    pandas.DataFrame: Merged dataframe containing all player statistics
    """
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
    merged_df = processed_sheets[0]
    for i, right_df in enumerate(processed_sheets[1:], 1):
        right_cols = [col for col in right_df.columns if col not in merged_df.columns]
        right_cols.append('Player_ID')
        right_df_slim = right_df[right_cols]
        print(f"\nMerging sheet {i+1}")
        print(f"Current shape: {merged_df.shape}")
        merged_df = pd.merge(
            merged_df,
            right_df_slim,
            on='Player_ID',
            how='outer',
            validate='1:1'
        )
        print(f"After merge: {merged_df.shape}")
    
    return merged_df

def test_position_classifier(test_file_path, save_merged_data=True):
    """
    Test the trained position classifier on new data
    
    Parameters:
    test_file_path (str): Path to the test Excel file
    save_merged_data (bool): Whether to save the merged dataframe to CSV
    
    Returns:
    tuple: (test_predictions, actual_labels, performance_report, merged_df)
    """
    # Load the trained pipeline and feature columns
    print("Loading trained model and features...")
    pipeline = joblib.load('player_position_pipeline2.pkl')
    feature_cols = joblib.load('feature_columns2.pkl')
    
    # Load and merge sheets
    print("\nLoading and preprocessing test data...")
    sheets = pd.read_excel(test_file_path, sheet_name=None)
    test_df = merge_player_sheets(sheets)
    
    # Save merged data if requested
    if save_merged_data:
        output_path = test_file_path.replace('.xlsx', '_merged.csv')
        test_df.to_csv(output_path, index=False)
        print(f"\nMerged data saved to: {output_path}")
    
    # Clean positions
    test_df['Position'] = test_df['Pos'].apply(lambda x: str(x).split(',')[0].strip())
    test_df = test_df[test_df['Position'].isin(['DF', 'MF', 'FW', 'GK'])]
    
    # Convert features to numeric, explicitly handling datetime columns
    X_test = test_df[feature_cols].copy()
    for col in X_test.columns:
        try:
            # Skip if column contains datetime objects
            if pd.api.types.is_datetime64_any_dtype(X_test[col]):
                print(f"Skipping datetime column: {col}")
                continue
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
        except Exception as e:
            print(f"Error processing column {col}: {str(e)}")
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    y_test = test_df['Position']
    
    print("\nTest set dimensions:")
    print(f"Number of samples: {X_test.shape[0]}")
    print(f"Number of features: {X_test.shape[1]}")
    print("\nClass distribution in test set:")
    print(y_test.value_counts())
    
    # Make predictions
    print("\nMaking predictions...")
    test_predictions = pipeline.predict(X_test)
    
    # Calculate performance metrics
    performance_report = classification_report(y_test, test_predictions)
    print("\nTest Set Performance Report:")
    print(performance_report)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, test_predictions, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['DF', 'FW', 'GK', 'MF'],
                yticklabels=['DF', 'FW', 'GK', 'MF'])
    plt.title('Test Set Confusion Matrix\n(Normalized by True Labels)')
    plt.xlabel('Predicted Position')
    plt.ylabel('True Position')
    plt.tight_layout()
    plt.show()
    
    # Return predictions, actual labels, performance report, and merged dataframe
    return test_predictions, y_test, performance_report, test_df

# Example usage
if __name__ == "__main__":
    test_file_path = 'Premier League Players 23_24 Stats_test.xlsx'
    predictions, actual_labels, report, merged_data = test_position_classifier(test_file_path)