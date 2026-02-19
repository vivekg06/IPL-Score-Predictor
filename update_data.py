import pandas as pd
import numpy as np
import requests
import io
import os
from datetime import datetime

# URLs for the datasets
MATCHES_URL = "https://raw.githubusercontent.com/avinashyadav16/ipl-analytics/main/matches_2008-2024.csv"
DELIVERIES_URL = "https://raw.githubusercontent.com/avinashyadav16/ipl-analytics/main/deliveries_2008-2024.csv"

def download_data():
    print("Downloading matches data...")
    response_matches = requests.get(MATCHES_URL)
    matches = pd.read_csv(io.StringIO(response_matches.content.decode('utf-8')))
    
    print("Downloading deliveries data (this may take a moment)...")
    response_deliveries = requests.get(DELIVERIES_URL)
    deliveries = pd.read_csv(io.StringIO(response_deliveries.content.decode('utf-8')))
    
    # Clean column names (remove extracted spaces)
    matches.columns = matches.columns.str.strip()
    deliveries.columns = deliveries.columns.str.strip()
    
    return matches, deliveries

def process_data(matches, deliveries):
    print("Processing data...")
    
    # Merge matches and deliveries
    # 'id' in matches corresponds to 'match_id' in deliveries
    total_df = deliveries.merge(matches, left_on='match_id', right_on='id')
    
    # Filter for columns we need and rename if necessary
    # Expected final columns: 
    # mid, date, venue, bat_team, bowl_team, batsman, bowler, runs, wickets, overs, runs_last_5, wickets_last_5, striker, non-striker, total
    
    # Rename columns to match existing ipl.csv format logic
    total_df.rename(columns={
        'match_id': 'mid',
        'venue_x': 'venue', # deliveres usually doesn't have venue, matches does. Let's check.
        # Actually matches has 'venue', deliveries usually doesn't. 
        # After merge, if both have it, it becomes venue_x, venue_y.
        # Let's assume matches has the correct venue/date.
        'batting_team': 'bat_team',
        'bowling_team': 'bowl_team',
        'batter': 'batsman',
        'bowler': 'bowler', # existing
        'date': 'date'
    }, inplace=True)

    # Some datasets use 'venue' in matches, some 'stadium'. 
    # Let's identify the correct venue column from `matches` dataframe columns which are now in total_df
    if 'venue' in matches.columns:
        total_df['venue'] = total_df['venue']
    elif 'stadium' in matches.columns:
        total_df['venue'] = total_df['stadium']
    
    # Current Score Calculation
    # We need cumulative runs and wickets for *current* state at that ball
    # 'total_runs' in deliveries is usually runs on that ball (batsman + extras)
    
    # Sort by match_id, inning, over, ball
    total_df.sort_values(['mid', 'inning', 'over', 'ball'], inplace=True)
    
    # Calculate cumulative runs and wickets per match & inning
    total_df['runs'] = total_df.groupby(['mid', 'inning'])['total_runs'].cumsum()
    total_df['wickets'] = total_df.groupby(['mid', 'inning'])['is_wicket'].cumsum()
    
    # Calculate Overs (e.g., 0.1, 0.2, ... 19.6)
    # over is 0-indexed in some datasets, 1-indexed in others.
    # In Cricsheet/IPL datasets, usually 'over' is 0-19. 
    # Let's normalize inputs. 
    # If over is 0, ball 1 -> 0.1
    # We will just use the provided 'over' and 'ball' columns directly but formatted.
    # Actually, the model expects a float: 5.4 means 5 overs and 4 balls.
    # If raw data 'over' is 0-19, we convert to 0.1..19.6
    # Let's inspect values briefly (we can't here, so we assume standard 0-19 format)
    # If over starts at 0:
    total_df['overs'] = total_df['over'] + (total_df['ball'] / 10.0) 
    # Note: 0.6 becomes 0.6, next over starts at 1.1. 
    # Ideally standard notation is 0.1 to 0.6, then 1.1. 
    # But for calculation `5.4` usually means 5.4 overs mathematical -> 5 + 4/6? 
    # Let's stick to the float representation: Match_Over.Ball
    
    # Calculate Total Score (Target)
    # Group by match and inning to find max score
    total_scores = total_df.groupby(['mid', 'inning'])['total_runs'].sum().reset_index()
    total_scores.rename(columns={'total_runs': 'total'}, inplace=True)
    total_df = total_df.merge(total_scores, on=['mid', 'inning'])
    
    # Calculate Runs and Wickets in Last 5 Overs
    # We can do this by shifting the dataframe or using rolling window on the group
    # A simplified efficient way:
    # 30 balls ago (approx 5 overs). 
    # running_score - running_score_30_balls_ago
    
    # Group by match/inning
    groups = total_df.groupby(['mid', 'inning'])
    
    # We'll rely on the sorted order.
    # rolling(30) is tricky with groups in pandas without re-indexing.
    # Vectorized approach:
    # 1. Create a temporary column 'prev_runs' = runs shifted by 30
    # 2. But this shift must respect group boundaries.
    
    total_df['prev_runs_30'] = groups['runs'].shift(30).fillna(0)
    total_df['prev_wickets_30'] = groups['wickets'].shift(30).fillna(0)
    
    total_df['runs_last_5'] = np.where(total_df['overs'] >= 5.0, total_df['runs'] - total_df['prev_runs_30'], total_df['runs'])
    total_df['wickets_last_5'] = np.where(total_df['overs'] >= 5.0, total_df['wickets'] - total_df['prev_wickets_30'], total_df['wickets'])
    
    # Handle the case where run_last_5 might be negative due to shift across innings boundaries?
    # No, we grouped by mid and inning, so shift won't cross boundaries.
    
    # Select columns for export
    # We need: mid,date,venue,bat_team,bowl_team,batsman,bowler,runs,wickets,overs,runs_last_5,wickets_last_5,striker,non-striker,total
    
    # In raw data: 'batter' is striker, 'non_striker' is non-striker.
    # Rename for consistency with old ipl.csv
    total_df.rename(columns={
        'non_striker': 'non-striker', 
        'batter': 'batsman' 
    }, inplace=True)
    
    # The old ipl.csv had both 'batsman' and 'striker'. 
    # 'batsman' was the player facing via name? 
    # 'striker' column in old csv seemed to be numeric in some rows (maybe runs?). 
    # But since they are dropped in training, we just need to provide columns that don't crash the script OR 
    # just providing 'batsman' and 'non-striker' is enough if the script uses `errors='ignore'`.
    # The script drops: ["date", "mid", "batsman", "bowler", "striker", "non-striker"]
    
    # let's add dummy columns if needed, or just map 'batsman' to 'striker' as well to be safe.
    total_df['striker'] = total_df['batsman']
    
    # Filter for First Innings only
    total_df = total_df[total_df['inning'] == 1]
    
    # Define final column order
    final_columns = [
        'mid', 'date', 'venue', 'bat_team', 'bowl_team', 
        'batsman', 'bowler', 'runs', 'wickets', 'overs', 
        'runs_last_5', 'wickets_last_5', 'striker', 'non-striker', 'total'
    ]
    
    # Filter columns
    output_df = total_df[final_columns].copy()
    
    # Clean up string columns 
    str_cols = ['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler', 'striker', 'non-striker']
    for col in str_cols:
        if col in output_df.columns:
            output_df[col] = output_df[col].astype(str).str.strip()
            
    return output_df

if __name__ == "__main__":
    if not os.path.exists("ipl.csv"):
        print("Existing ipl.csv not found... wait, we are updating it.")
    
    matches, deliveries = download_data()
    print(f"Loaded {len(matches)} matches and {len(deliveries)} deliveries.")
    
    new_ipl_df = process_data(matches, deliveries)
    print(f"Processed dataframe shape: {new_ipl_df.shape}")
    print("Columns:", new_ipl_df.columns.tolist())
    
    # Backup old file
    if os.path.exists("ipl.csv"):
        os.rename("ipl.csv", "ipl_old_backup.csv")
        print("Backed up old ipl.csv to ipl_old_backup.csv")
        
    new_ipl_df.to_csv("ipl.csv", index=False)
    print("Successfully saved new ipl.csv!")
