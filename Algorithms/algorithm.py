import pandas as pd
import numpy as np
import json
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
    labeledData = pd.DataFrame.from_dict(return_data, orient='index', columns=['pS', 'pA', 'winPerT', 'winPer4'])
    return labeledData.reset_index(names='teamName')


with open('Schedule.json', 'r') as f:
    d = json.load(f) # loads the data into python

with open('Schedule(10-2-2023).json', 'r') as g:
    newD = json.load(g) # loads the new data into python


df = pd.json_normalize(data=d['teams'], record_path='schedule',
                        meta=['name', 'league'])
print(df)
df = df.drop(columns=['index', 'timestamp'], axis=1)
df = df.drop(df[df['location'] == 'BYE'].index)

newDF = pd.json_normalize(data=newD['teams'], record_path='schedule',
                        meta=['name', 'league'])
print(newDF)

### Combine parser.py
df2 = get_data_as_dict()
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

df = df.dropna() # remove canceled games
print(df)


### SINGLE GAME SELECTION ###
z = 0 # A number between 0 and 2900, representing the game we want to test
z = random.randrange(2900) # comment out if not using random
extract_game = df.iloc[z]
simluate_game = pd.DataFrame([extract_game])
print("\nSimulated Game:")
print(simluate_game)



# DataFrame Cleaning
df['dayOfWeek'], labelDay = pd.factorize(df.dayOfWeek)
df['location'], labelLocation = pd.factorize(df.location)
df['outcome'], labelOutcome = pd.factorize(df.outcome)
df['pointsScored'] = df['pointsScored'].astype(str).astype(np.int64)
df['pointsAllowed'] = df['pointsAllowed'].astype(str).astype(np.int64)
df['league'], labelLeague = pd.factorize(df.league)

# mapping unique team names
uniqueTeams = np.unique(df[['name', 'opponent']])
factors = np.arange(len(uniqueTeams))
df[['name', 'opponent']] = df[['name', 'opponent']].replace(uniqueTeams, factors).astype(np.int64)

# print(df.dtypes, "\n")

# Model Testing
k = 5
n = 10
kf = RepeatedKFold(n_splits=k, n_repeats=n, random_state=None)
clf = LogisticRegression(penalty='l2', max_iter=10000, random_state=1)
#clf = RandomForestClassifier(random_state=1)

flag = False # set to True if you want to see Model Performance
if (flag):
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



## Single Game Testing ##

test_game = df.iloc[z]
test_match = pd.DataFrame([test_game])
df = df.drop(df.index[z])
print("\nSimulated Game (Formatted):")
print(test_match)

y_train = df['outcome']
x_train = df.drop(columns=['outcome', 'pointsScored', 'pointsAllowed']) 
y_test = test_match['outcome']
x_test = test_match.drop(columns=['outcome', 'pointsScored', 'pointsAllowed']) 



#x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=None)
# x_train2, x_val2, y_train2, y_val2 = train_test_split(x_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

clf.fit(x_train, y_train)

y_val_pred = clf.predict(x_test)

# Confidence Score
probability = clf.predict_proba(x_test)
instance_prob = probability[0]
max_prob = max(instance_prob)
confidence_score = round((max_prob / sum(instance_prob)) * 100, 2)



if y_val_pred == 0:
    finalOutcome = "WIN"
elif y_val_pred == 1:
    finalOutcome = "LOSE"
else:
    finalOutcome = "DRAW"

print("\nPrediction: The", str(simluate_game.iloc[0]['name']), "will", finalOutcome, "against the",
      str(simluate_game.iloc[0]['opponent']), "at the", str(simluate_game.iloc[0]['league']), 
      "with", confidence_score, "percent confidence.\n")
