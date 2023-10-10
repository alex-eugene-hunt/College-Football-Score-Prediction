import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data from the JSON file
with open('./Data Parser/Schedule.json', 'r') as f:
    data = json.load(f)

# Normalize JSON data into a DataFrame
df = pd.json_normalize(data=data['teams'], record_path='schedule',
                       meta=['name', 'league'])
print(df)

# Drop unnecessary columns
df = df.drop(columns=['index', 'timestamp'], axis=1)
df = df.drop(df[df['location'] == 'BYE'].index)

# DataFrame Cleaning
df['dayOfWeek'], labelDay = pd.factorize(df.dayOfWeek)
df['location'], labelLocation = pd.factorize(df.location)
df['outcome'], labelOutcome = pd.factorize(df.outcome)
df['pointsScored'] = df['pointsScored'].astype(str).astype(np.int64)
df['pointsAllowed'] = df['pointsAllowed'].astype(str).astype(np.int64)
df['league'], labelLeague = pd.factorize(df.league)

# Mapping unique team names
uniqueTeams = np.unique(df[['name', 'opponent']])
factors = np.arange(len(uniqueTeams))
df[['name', 'opponent']] = df[['name', 'opponent']].replace(uniqueTeams, factors).astype(np.int64)

labels = df['outcome']  # y-labels
df = df.drop(columns=['outcome', 'pointsScored', 'pointsAllowed'])  # x-data
print(df)
print(labels)
print(df.dtypes, "\n")

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=None)

# Create and train the RandomForestClassifier with a different variable name
random_forest_classifier = RandomForestClassifier(random_state=1)
random_forest_classifier.fit(x_train, y_train)

# Predict on the test set
y_val_pred = random_forest_classifier.predict(x_test)

# Actual vs prediction
accuracyScore = accuracy_score(y_test, y_val_pred)
print(f'\nConfusion Matrix: \n{confusion_matrix(y_test, y_val_pred)}\n')
print("\nAccuracy: ", accuracyScore, "\n")
