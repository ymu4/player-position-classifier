import pandas as pd

def merge_premier_league_stats(excel_file_path):
    """
    Merges all sheets from the Premier League stats Excel file into a single dataframe.
    
    Parameters:
    excel_file_path (str): Path to the Excel file containing Premier League stats
    
    Returns:
    pandas.DataFrame: Merged dataframe containing all stats
    """
    # Dictionary to store all dataframes
    sheet_dfs = {}
    
    # Read all sheets
    with pd.ExcelFile(excel_file_path) as xls:
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name)
            
            # Clean sheet name for column prefixes
            clean_sheet_name = sheet_name.replace(" ", "_").lower()
            
            # Skip renaming for common identifier columns
            common_cols = ['Rk', 'Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born', '90s']
            
            # Rename columns to avoid conflicts, except for common columns
            rename_dict = {
                col: f"{clean_sheet_name}_{col}" 
                for col in df.columns 
                if col not in common_cols
            }
            df = df.rename(columns=rename_dict)
            
            sheet_dfs[clean_sheet_name] = df

    # Start with shooting stats as base
    merged_df = sheet_dfs['shooting_stats']

    # Merge all other sheets
    for sheet_name, df in sheet_dfs.items():
        if sheet_name != 'shooting_stats':
            # Merge on common identifying columns
            merged_df = pd.merge(
                merged_df,
                df,
                on=['Rk', 'Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born', '90s'],
                how='outer',
                suffixes=('', f'_{sheet_name}')
            )

    # Clean up the merged dataframe
    # Remove duplicate columns that might have been created during merge
    duplicate_cols = [col for col in merged_df.columns if col.endswith('_shooting_stats')]
    merged_df = merged_df.drop(columns=duplicate_cols)
    
    # Sort by Rank and reset index
    merged_df = merged_df.sort_values('Rk').reset_index(drop=True)
    
    # Add total stats columns
    merged_df['total_goals'] = merged_df['Gls']  # Goals from shooting stats
    merged_df['total_assists'] = merged_df['passing_stats_Ast']  # Assists from passing stats
    merged_df['total_minutes'] = merged_df['90s'] * 90  # Convert 90s to minutes
    
    # Calculate additional combined metrics
    merged_df['goal_contributions'] = merged_df['total_goals'] + merged_df['total_assists']
    merged_df['goals_per_90'] = merged_df['total_goals'] / merged_df['90s']
    merged_df['assists_per_90'] = merged_df['total_assists'] / merged_df['90s']
    merged_df['goal_contributions_per_90'] = merged_df['goal_contributions'] / merged_df['90s']
    
    return merged_df

def get_player_summary(merged_df, player_name):
    """
    Returns a summary of key statistics for a specific player.
    
    Parameters:
    merged_df (pandas.DataFrame): The merged dataframe containing all stats
    player_name (str): Name of the player to summarize
    
    Returns:
    dict: Dictionary containing key statistics for the player
    """
    player_data = merged_df[merged_df['Player'] == player_name].iloc[0]
    
    summary = {
        'name': player_data['Player'],
        'team': player_data['Squad'],
        'position': player_data['Pos'],
        'age': player_data['Age'],
        'minutes_played': player_data['total_minutes'],
        'goals': player_data['total_goals'],
        'assists': player_data['total_assists'],
        'goal_contributions': player_data['goal_contributions'],
        'goals_per_90': player_data['goals_per_90'],
        'assists_per_90': player_data['assists_per_90'],
        'shot_accuracy': player_data['SoT%'] if 'SoT%' in player_data else None,
        'pass_completion': player_data['passing_stats_Cmp%'] if 'passing_stats_Cmp%' in player_data else None,
        'xG': player_data['xG'] if 'xG' in player_data else None,
        'xA': player_data['passing_stats_xA'] if 'passing_stats_xA' in player_data else None
    }
    
    return summary


merged_stats = merge_premier_league_stats('Premier League Players 23_24 Stats_train.xlsx')
player_stats = get_player_summary(merged_stats, 'Erling Haaland')