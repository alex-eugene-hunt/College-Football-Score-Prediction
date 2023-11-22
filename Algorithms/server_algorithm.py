import pandas as pd
import numpy as np
import json
import random
import math
import pickle
import dill
import os
import requests
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from google.cloud import storage
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def pull_schedule(fname):
    """
    Pulls the newest schedule JSON using the API
    :param fname: file name to save the JSON as
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket("capstoneprediction.appspot.com")
    blob = bucket.blob(fname)

    api_url = "http://www.pepplerstats.com/SportsAggregator/API/?CAPSTONE=JSON&SPORT=NCAAF"
    
    # Pull the latest schedule JSON
    response = requests.get(api_url)
    schedule = response.json()
    
    # Save the schedule file
    with blob.open('w') as f:
        json.dump(schedule, f, ensure_ascii=False, indent=2)
    f.close()

def load_dill(fname):
    """
    Load an object that was stored using dill/pickle
    :param fname: File name of the stored object
    :return obj: Object to retrieve from file
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket("capstoneprediction.appspot.com")
    blob = bucket.blob(fname)
    
    # Load the object from a pickle/dill file
    with blob.open('rb') as f:
        obj = dill.load(f)
    
    return obj

def merge_schedules(oldD, newD):
    """
    Merges two schedule files and updates leagues
    :param oldD: Old raw data filename, usually the previous full file
    :param newD: New raw data filename, usually the file to take from to add to full file,
        assumes no overlap with oldD
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket("capstoneprediction.appspot.com")
    oldD_blob = bucket.blob(oldD)
    
    
    with oldD_blob.open('r') as f:
        d = json.load(f) # loads the data into python

    newD_blob = bucket.blob(newD)
    with newD_blob.open('r') as g:
        newD = json.load(g) # loads the new data into python
    
    # Make copy of old data to append to
    full_d = d
    
    # Iterate through new data and append to old data
    for team_data in newD['teams']:
        team_name = team_data['name']
        league = team_data['league']
        
        # Find existing entry for the team in full_df, otherwise it is None
        existing_team = next((item for item in full_d['teams'] if item['name'] == team_name), None)
        
        # If team is not in full_df, add it
        if existing_team is None:
            full_d['teams'].append(team_data)
        else:
            # Update to new league if needed
            if existing_team['league'] != league:
                existing_team['league'] = league
                
            existing_schedule = existing_team['schedule']
                
            # Search through schedules of current team to replace old entries
            # Iterate through each day of the current team's schedule
            for day in team_data['schedule']:
                timestamp = day['timestamp']
                # Find matching day in full_df
                existing_day = next((item for item in existing_schedule if item['timestamp'] == timestamp), None)
                
                # Update day if matching day was found and it had been updated
                if existing_day is not None:
                    if day['outcome'] != existing_day['outcome']:
                        existing_day = day
                else:  
                    # If not found, add the day to the full schedule
                    existing_schedule.append(day)
                
            # Merge schedules
            #existing['schedule'] = existing['schedule'] + team_data['schedule']
            
    # Save full data file for usage later
    fullD_blob = bucket.blob('Full_Schedule.json')
    with fullD_blob.open('w') as f:
        json.dump(full_d, f, ensure_ascii=False, indent=2)
    f.close()

def get_data_as_dict(rawData):
    """
    Obtain custom values for the dataset (total points scored, total points allowed,
                                          win percentage, win percentage for last 4 games)
    
    :param rawData: Raw data from JSON file to obtain custom values from
    :return: DataFrame showing custom values for each team
    """
    # Use merged data from the files
    data = rawData
   
    team_info = []

    for team_data in data['teams']:
    
        team_name = team_data['name']
        schedule = team_data['schedule']
        # Sort by date from latest to earliest
        schedule.sort(key=lambda x: datetime.strptime(x['timestamp'], '%m/%d/%Y'), reverse=True)
        
        # Initialize variables to calculate points scored and points allowed
        total_points_scored = 0
        total_points_allowed = 0
        total_games = 0
        total_wins = 0
        last_4_games = 0
        
        # Iterate through each game in the team's schedule
        for game in schedule:
            # Skip BYE games or games that did not occur
            if(game['location'] == 'BYE' or game['outcome'] == '-'):
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

    return_data = {}
    for row in team_info:
        name = row[0]
        info = row[1:]
        return_data[name] = info
        
    # 'Team' : [points scored, points allowed, win percentage total, win percentage last 4 games]
    labeledData = pd.DataFrame.from_dict(return_data, orient='index', columns=['pS', 'pA', 'winPerT', 'winPer4'])
    return labeledData.reset_index(names='teamName')

def map_cust_vals(rawData, df):
    """
    Maps custom values to the input DataFrame directly
    :param rawData: Raw data from JSON file to obtain custom values from
    :param df: DataFrame to map values to, will be directly modified
    """
    ### Combine parser.py
    df2 = get_data_as_dict(rawData)
    # adding total points scored per team
    mapping_pS = df2.set_index('teamName')['pS'].to_dict() 
    df['pS'] = df['name'].map(mapping_pS)
    df['OpS'] = df['opponent'].map(mapping_pS)
    # adding total points scored per team
    mapping_pA = df2.set_index('teamName')['pA'].to_dict() 
    df['pA'] = df['name'].map(mapping_pA)
    df['OpA'] = df['opponent'].map(mapping_pA)
    # adding win percentage total per team
    mapping_winPerT = df2.set_index('teamName')['winPerT'].to_dict() 
    df['winPerT'] = df['name'].map(mapping_winPerT)
    df['OwinPerT'] = df['opponent'].map(mapping_winPerT)
    # adding win percentage last 4 games per team
    mapping_winPer4 = df2.set_index('teamName')['winPer4'].to_dict() 
    df['winPer4'] = df['name'].map(mapping_winPer4)
    df['OwinPer4'] = df['opponent'].map(mapping_winPer4)
    
def get_encodings(df):
    """
    Obtain encodings for the data set non-numerical values
    :param df: Full DataFrame to set encodings for
    :return: List of all encodings in the order of day, location, outcome, league, and team
    """
    dayEncoder = LabelEncoder()
    locEncoder = LabelEncoder()
    outcomeEncoder = LabelEncoder()
    leagueEncoder = LabelEncoder()
    teamEncoder = LabelEncoder()
    
    dayEncoder.fit(df['dayOfWeek'])
    locEncoder.fit(df['location'])
    outcomeEncoder.fit(df['outcome'])
    leagueEncoder.fit(df['league'])
    teamEncoder.fit(df['name'])
    
    return [dayEncoder, locEncoder, outcomeEncoder, leagueEncoder, teamEncoder]
    

def load_data():
    """
    Loads data from JSON file, maps custom values, and returns final DataFrame
    :return: Final DataFrame with custom values and removed N/A entries
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket("capstoneprediction.appspot.com")
    blob = bucket.blob('Full_Schedule.json')
    
    with blob.open('r') as f:
        rawData = json.load(f) # loads the data into python
    f.close()
    
    df = pd.json_normalize(data=rawData['teams'], record_path='schedule',
                            meta=['name', 'league'])
    print(df)
    
    map_cust_vals(rawData, df)
    
    df = df.drop(columns=['index', 'timestamp'], axis=1)
    df = df.drop(df[df['location'] == 'BYE'].index)
    
    # Remove canceled or future games
    df = df.dropna() 
    df = df.drop(df[df['outcome'] == '-'].index)

    print(df)
    
    return df

def train_model():
    """
    Trains models to predict outcome and score, allows option to perform a test
    on a single game if uncommented. Saves the models to .pkl files
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket("capstoneprediction.appspot.com")
    
    # Load full data after mapping custom values to it
    df = load_data()

    ### SINGLE GAME SELECTION ###
    z = 0 # A number between 0 and maximum row, representing the game we want to test
    z = random.randrange(df.shape[0]) # comment out if not using random
    extract_game = df.iloc[z]
    simluate_game = pd.DataFrame([extract_game])
    print("\nSimulated Game:")
    print(simluate_game)

    # Obtain encodings for non-numerical columns
    # Encodings are in the order of day, location, outcome, league, team
    encodings_blob = bucket.blob('encodings.pkl')
    encodings = get_encodings(df)
    with encodings_blob.open('wb') as f:
        dill.dump(encodings, f)
    
    df['dayOfWeek'] = encodings[0].transform(df['dayOfWeek'])
    df['location'] = encodings[1].transform(df['location'])
    df['outcome'] = encodings[2].transform(df['outcome'])
    df['pointsScored'] = df['pointsScored'].astype(str).astype(np.int64)
    df['pointsAllowed'] = df['pointsAllowed'].astype(str).astype(np.int64)
    df['league'] = encodings[3].transform(df['league'])

    df['name'] = encodings[4].transform(df['name'])
    df['opponent'] = encodings[4].transform(df['opponent'])

    # Model Testing
    k = 5
    n = 10
    kf = RepeatedKFold(n_splits=k, n_repeats=n, random_state=None)
    clf = LogisticRegression(penalty='l2', max_iter=10000, random_state=1)
    linReg = LinearRegression(fit_intercept=True)

    outcome_flag = False # set to True if you want to see Model Performance
    if (outcome_flag):
        cv_results = []

        for train_index, test_index in kf.split(df):
            X_train, X_test = df.iloc[train_index], df.iloc[test_index]
            y_train, y_test = X_train['outcome'], X_test['outcome']

            clf.fit(X_train.drop(columns=['outcome', 'pointsScored', 'pointsAllowed']), y_train)

            y_pred = clf.predict(X_test.drop(columns=['outcome', 'pointsScored', 'pointsAllowed']))

            accuracy = accuracy_score(y_test, y_pred)
            cv_results.append(accuracy)

        accuracies = np.array(cv_results)
        average = round(accuracies.mean(), 3)
        std = round(accuracies.std(), 3)

        print("\n Model Accuracy: ", average, " +/- ", std)

        # Important Features
        
        coefficients = clf.coef_[0]
        features = df.drop(columns=['outcome', 'pointsScored', 'pointsAllowed']).columns

        feature_importance_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': coefficients
        })

        feature_importance_df['Absolute Coefficient'] = feature_importance_df['Coefficient'].abs()
        feature_importance_df = feature_importance_df.sort_values(by='Absolute Coefficient', ascending=False)
        feature_importance_df = feature_importance_df.drop(columns=['Coefficient'])

        print(feature_importance_df)

    score_flag = False # set to True if you want to see Model Performance
    if (score_flag):
        cv_results = []

        for train_index, test_index in kf.split(df):
            X_train, X_test = df.iloc[train_index], df.iloc[test_index]
            y_train, y_test = X_train[['pointsScored', 'pointsAllowed']], X_test[['pointsScored', 'pointsAllowed']]

            linReg.fit(X_train.drop(columns=['outcome', 'pointsScored', 'pointsAllowed']), y_train)

            y_pred = linReg.predict(X_test.drop(columns=['outcome', 'pointsScored', 'pointsAllowed']))

            accuracy = r2_score(y_test, y_pred)
            cv_results.append(accuracy)

        accuracies = np.array(cv_results)
        average = round(accuracies.mean(), 3)
        std = round(accuracies.std(), 3)

        print("\n Model Accuracy: ", average, " +/- ", std)

        # Important Features
        
        coefficients = linReg.coef_[0]
        features = df.drop(columns=['outcome', 'pointsScored', 'pointsAllowed']).columns

        feature_importance_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': coefficients
        })

        feature_importance_df['Absolute Coefficient'] = feature_importance_df['Coefficient'].abs()
        feature_importance_df = feature_importance_df.sort_values(by='Absolute Coefficient', ascending=False)
        feature_importance_df = feature_importance_df.drop(columns=['Coefficient'])

        print(feature_importance_df)

    # Comment out if not doing random test prediction
    ## Single Game Testing ##

# =============================================================================
#     test_game = df.iloc[z]
#     test_match = pd.DataFrame([test_game])
#     df = df.drop(df.index[z])
#     print("\nSimulated Game (Formatted):")
#     print(test_match)
# =============================================================================

    y_train = df['outcome']
    x_train = df.drop(columns=['outcome', 'pointsScored', 'pointsAllowed']) 
    #y_test = test_match['outcome']
    #x_test = test_match.drop(columns=['outcome', 'pointsScored', 'pointsAllowed']) 


    #x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=None)
    # x_train2, x_val2, y_train2, y_val2 = train_test_split(x_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

    clf.fit(x_train, y_train)
    
    clf_blob = bucket.blob('clf_model.pkl')
    with clf_blob.open('wb') as f:
        dill.dump(clf, f)

    # Comment out if not doing random test prediction
# =============================================================================
#     y_val_pred = clf.predict(x_test)
# 
#     # Confidence Score
#     probability = clf.predict_proba(x_test)
#     instance_prob = probability[0]
#     max_prob = max(instance_prob)
#     confidence_score = round((max_prob / sum(instance_prob)) * 100, 2)
# 
#     if y_val_pred == encodings[2].transform(['W']):
#         finalOutcome = "WIN"
#     elif y_val_pred == encodings[2].transform(['L']):
#         finalOutcome = "LOSE"
#     else:
#         finalOutcome = "DRAW"
# 
#     print("\nPrediction: The", str(simluate_game.iloc[0]['name']), "will", finalOutcome, "against the",
#           str(simluate_game.iloc[0]['opponent']), "at the", str(simluate_game.iloc[0]['league']), 
#           "with", confidence_score, "percent confidence.\n")
# =============================================================================

    # Testing Score Prediction

    y_train_score = df[['pointsScored', 'pointsAllowed']]
    y_train_score = y_train_score.to_numpy()
    #y_test_score = test_match[['pointsScored', 'pointsAllowed']]
    #y_test_score = y_test_score.to_numpy()

    linReg.fit(x_train, y_train_score)
    
    linReg_blob = bucket.blob('linReg_model.pkl')
    with linReg_blob.open('wb') as f:
        dill.dump(linReg, f)
    
    # Comment out if not doing random test prediction
# =============================================================================
#     y_val_pred = linReg.predict(x_test)
# 
#     # If a team wins, it must have a higher score
#     if y_val_pred[0][0] > y_val_pred[0][1]:
#         y_val_pred_0 = math.ceil(y_val_pred[0][0])
#         y_val_pred_1 = math.floor(y_val_pred[0][1])
#     elif y_val_pred[0][0] < y_val_pred[0][1]:
#         y_val_pred_0 = math.floor(y_val_pred[0][0])
#         y_val_pred_1 = math.ceil(y_val_pred[0][1])
#     else:
#         y_val_pred_0 = round(y_val_pred[0][0])
#         y_val_pred_1 = round(y_val_pred[0][1])
# 
#     print("\nPrediction: The ", str(simluate_game.iloc[0]['name']), " will have a score of ",
#           y_val_pred_0, " and the ", str(simluate_game.iloc[0]['opponent']),
#           " will have a score of ", y_val_pred_1, ".\n", sep="")   
# =============================================================================

def predict1(dayOfWeek, location, name, opponent):
    """
    Loads models and makes prediction for outcome and scores of a game
    :param dayOfWeek: Day of the week of the game
    :param location: Whether the game was at home or at the opponent's location or neither
    :param name: name of the first team
    :param opponent: name of the second team/opponent
    :return output_str: output string with outcome and score predictions
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket("capstoneprediction.appspot.com")
    blob = bucket.blob('Full_Schedule.json')
    
    # Need to obtain league of current player
    with blob.open('r') as f:
        d = json.load(f) # loads the data into python
        
    df = pd.json_normalize(data=d['teams'], record_path='schedule',
                            meta=['name', 'league'])
    
    league = df[df.name == name].iloc[0]['league']
    
    x_init = [[dayOfWeek, location, opponent, name, league]]
    # Convert input into DataFrame with column names
    x_test = pd.DataFrame(x_init, columns=['dayOfWeek', 'location', 'opponent', 'name', 'league'])
    # Map custom values
    map_cust_vals(d, x_test)
    
    # Encode the input game values to numbers
    encodings = load_dill("encodings.pkl")
    x_test['dayOfWeek'] = encodings[0].transform(x_test['dayOfWeek'])
    x_test['location'] = encodings[1].transform(x_test['location'])
    x_test['league'] = encodings[3].transform(x_test['league'])
    x_test['name'] = encodings[4].transform(x_test['name'])
    x_test['opponent'] = encodings[4].transform(x_test['opponent'])
    print(x_test)
    
    
    # Load the models
    clf = load_dill("clf_model.pkl")
    linReg = load_dill("linReg_model.pkl")
    
    # Predict the outcome
    y_val_pred = clf.predict(x_test)
    
    # Confidence Score
    probability = clf.predict_proba(x_test)
    instance_prob = probability[0]
    max_prob = max(instance_prob)
    confidence_score = round((max_prob / sum(instance_prob)) * 100, 2)

    if y_val_pred == encodings[2].transform(['W']):
        finalOutcome = "WIN"
    elif y_val_pred == encodings[2].transform(['L']):
        finalOutcome = "LOSE"
    else:
        finalOutcome = "DRAW"

    print("\nPrediction: The", name, "will", finalOutcome, "against the",
          opponent, "at the", league, "with", confidence_score, "percent confidence.\n")
    
    # Predict the score
    y_val_pred = linReg.predict(x_test)

    # If a team wins, it must have a higher score
    if y_val_pred[0][0] > y_val_pred[0][1]:
        y_val_pred_0 = math.ceil(y_val_pred[0][0])
        y_val_pred_1 = math.floor(y_val_pred[0][1])
    elif y_val_pred[0][0] < y_val_pred[0][1]:
        y_val_pred_0 = math.floor(y_val_pred[0][0])
        y_val_pred_1 = math.ceil(y_val_pred[0][1])
    else:
        y_val_pred_0 = round(y_val_pred[0][0])
        y_val_pred_1 = round(y_val_pred[0][1])

    print("\nPrediction: The ", name, " will have a score of ",
          y_val_pred_0, " and the ", opponent,
          " will have a score of ", y_val_pred_1, ".\n", sep="") 
    
    output_str = ""
    
    if confidence_score > 70:
        output_str = (f"Prediction: The {name} will {finalOutcome} against the {opponent} "
                      f"at the {league} with {confidence_score} percent confidence.\n\n"
                      f"The {name} will have a score of {y_val_pred_0} and the "
                      f"{opponent} will have a score of {y_val_pred_1}.\n"
                      )
    else:
        output_str = (f"Prediction: The {name} will {finalOutcome} against the {opponent} "
                      f"at the {league} with {confidence_score} percent confidence.\n\n"
                      f"Score predictions are not reliable with a lower confidence score.\n"
                      )

    return output_str

def get_list_of_teams(data_filename):
    """
    Extracts a list of team names from the JSON data file.
    
    :param data_filename: The filename of the JSON data file.
    :return unique_team_names: A list of team names.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket("capstoneprediction.appspot.com")
    blob = bucket.blob(data_filename)
    
    # Load the JSON data from the file
    with blob.open('r') as f:
        data = json.load(f)
    
    # Extract the list of team names
    team_names = [team['name'] for team in data['teams']]
    
    # Use a set to remove duplicates and convert back to a list
    unique_team_names = list(set(team_names))
    
    return unique_team_names

if __name__ == "__main__":
    # Run this to pull the latest schedule
    pull_schedule("New_Schedule.json")
    
    # Must run this first to merge data and obtain file that will be used
    #merge_schedules("Full_Schedule.json", "Schedule(10-2-2023).json")
    
    data_filename = 'Full_Schedule.json'
    
    # Call the function to get the list of teams
    team_list = get_list_of_teams(data_filename)
    
    # Must run this to train a model before doing predictions
    #train_model()
    
    # Pick arguments and run this to make a prediction
    #predict1("Sat", "N", "TCU Horned Frogs", "Michigan Wolverines")
