import pandas as pd
import numpy as np
import pickle
import dill
from elo import Elo

# Possible way to load and store the Elo object for persistence
def load_data(dirname, fname):
    team_data = dill.load(open("%s/%s"%(dirname,fname)), "rb")
    
    return team_data


if __name__ == "__main__":
    # Read the initial file
    df = pd.read_csv("Schedule.csv", header=None, names=['index',
                     'timestamp',
                     'dayOfWeek',
                     'location',
                     'opponent',
                     'outcome',
                     'pointsScored',
                     'pointsAllowed'])
    
    # Assign team names to each row of that team
    loop_idx = 0
    name_list = []
    opp_name_list = []
    for i in range(df.shape[0]):
        if loop_idx == 0:
            name = df.iloc[i, 0]
        
        # Need to check opponent names as well, not all opponents have schedules
        opp_name = df.iloc[i]['opponent']
        
        name_list.append(name)
        if not pd.isna(opp_name) and opp_name not in opp_name_list:
            opp_name_list.append(opp_name)
            
        loop_idx += 1
        # Reset loop_idx for next team name
        if loop_idx > 138:
            loop_idx = 0
            
    # Add new column for team names
    df['team'] = name_list
    # Add result values for Elo
    outcome_vals = [1 if x == 'W' else 0 if x == 'L' else 0.5 for x in df['outcome']]
    df['outcomeVals'] = outcome_vals
    
    # Delete all rows with NaN values
    df.dropna(inplace=True)
    
    full_name_list = name_list + opp_name_list
    # Get rid of duplicate names
    full_name_list = np.unique(np.array(full_name_list))
    
    elo = Elo()
    # Initialize players in Elo system
    for x in full_name_list:
        elo.add_player(x)
        
    df_time = df.sort_values(by=['timestamp'])
    
    players = elo.get_players()
    
    # Create matrix to store date, player1, and player2 combinations, along with outcome
    # Prevents repeated games
    game_list = []
    # Record every game
    for i in range(int(0.75*df_time.shape[0])):
        timestamp = df_time.iloc[i]['timestamp']
        player1 = df_time.iloc[i]['team']
        player2 = df_time.iloc[i]['opponent']
        outcomeVal = df_time.iloc[i]['outcomeVals']
        
        # Try to locate timestamp in list
        # If not found, then add and go to next game
        try:
            time_idx = game_list.index(timestamp)
        except:
            game = [timestamp, player1, player2, outcomeVal]
            game_list.append(game)
            elo.rate(player1, player2, outcomeVal)
            continue
        
        # If timestamp is found, check players to see if it is a duplicate game
        while time_idx < len(game_list) and game_list[time_idx][0] == timestamp:
            # If duplicate game is found, skip adding this game
            if player1 == game_list[time_idx][2] and player2 == game_list[time_idx][1]:
                continue
            
            time_idx += 1
            
        # If while loop finishes, no duplicates exist, add the game
        game = [timestamp, player1, player2, outcomeVal]
        game_list.append(game)
        elo.rate(player1, player2, outcomeVal)
    
    elo_players = elo.get_players()
    
    # Evaluate overall accuracy of predictions
    correct = 0
    total = 0
    for x in game_list:
        # Obtain prediction of how likely it is that player wins against opponent
        prediction = elo.expect(x[1], x[2])
        
        # If player is more likely to win and outcome was a win, correct
        if prediction > 0.5 and x[3] == 1:
            correct += 1
        # If player is more likely to lose and outcome was a loss, correct
        elif prediction < 0.5 and x[3] == 0:
            correct += 1
        # Predict close chances as a draw
        elif (prediction >= 0.45 or prediction <= 0.55) and x[3] == 0.5:
            correct += 1
        
        total += 1
    
    print("Average accuracy:", correct/total)
    print(elo_players)
