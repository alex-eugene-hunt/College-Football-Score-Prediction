# Chris and Alex
# this will read Schedule.json
# this will return data from .json as dictionary
# ------- to run in another .py file -------------
# from parser.py import get_data_as_dict()
# data = get_data_as_dict()

import json

def get_data_as_dict():
    #open the json as read only
    with open('Schedule.json', 'r') as f:
        data = json.load(f) # loads the data into python
   
    team_info = []

    for team_data in data['teams']:
    
        team_name = team_data['name']
        schedule = team_data['schedule']
        schedule.sort(key=lambda x: x['timestamp'])
        
        # Initialize variables to calculate points scored and points allowed
        total_points_scored = 0
        total_points_allowed = 0
        total_games = 0
        total_wins = 0
        last_4_games = 0
        
        # Iterate through each game in the team's schedule
        for game in schedule:
            if(game['location'] == 'BYE'):
                continue
            points_scored = int(game['pointsScored'])
            points_allowed = int(game['pointsAllowed'])
            outcome = game['outcome']

            total_points_scored += points_scored
            total_points_allowed += points_allowed

            total_games += 1

            # Check if the game was a win
            if outcome == 'W':
                total_wins += 1
                if total_games <= 4:
                    last_4_games += 1

        # Calculate win percentage (wins / total games)
        if total_games > 0:
            win_percentage = round((total_wins / total_games) * 100, 2)
            last_4_per = (last_4_games / 4) * 100
        else:
            win_percentage = 0

        # Append team information to the list
        team_info.append([team_name, total_points_scored, total_points_allowed, win_percentage, last_4_per])
    f.close()

    return_data = {}
    for row in team_info:
        name = row[0]
        info = row[1:]
        return_data[name] = info
        
    # 'Team' : [points scored, points allowed, win percentage total, win percentage last 4 games]
    return return_data

if __name__ == "__main__":
    pass