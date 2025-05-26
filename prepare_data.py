import pandas as pd

def clean_excel_file(input_file_path, output_file_path):
    """
    Clean the Excel file by:
    1. Removing duplicate headers (rows where Pos = 'Pos')
    2. Removing duplicate columns
    3. Cleaning up the data structure
    
    Args:
        input_file_path (str): Path to the input Excel file
        output_file_path (str): Path to save the cleaned Excel file
    """
    # Read all sheets from the Excel file
    excel_file = pd.ExcelFile(input_file_path)
    
    # Create a dictionary to store cleaned dataframes
    cleaned_sheets = {}

    # Process each sheet
    for sheet_name in excel_file.sheet_names:
        # Read the sheet
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # Remove rows where Pos = 'Pos' (duplicate headers)
        df = df[df['Pos'] != 'Pos']
        
        # Reset the index after removing rows
        df = df.reset_index(drop=True)
        
        # Get list of duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated()]
        
        # Keep only the first instance of duplicate columns
        if len(duplicate_cols) > 0:
            df = df.loc[:, ~df.columns.duplicated()]
        
        # Store the cleaned dataframe
        cleaned_sheets[sheet_name] = df

    # Save to a new Excel file
    with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
        for sheet_name, df in cleaned_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    return cleaned_sheets

# Usage example:
if __name__ == "__main__":
    input_file = "Processed_Premier_League_Stats.xlsx"
    output_file = "Cleaned_Premier_League_Stats.xlsx"
    cleaned_data = clean_excel_file(input_file, output_file)