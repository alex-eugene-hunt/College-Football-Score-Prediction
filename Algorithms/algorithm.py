import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline


with open('Schedule.json', 'r') as f:
    d = json.load(f) # loads the data into python

df = pd.json_normalize(data=d['teams'], record_path='schedule',
                        meta=['name', 'league'])
print(df)
df = df.drop(columns=['index', 'timestamp'], axis=1)
df = df.drop(df[df['location'] == 'BYE'].index)

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

labels = df['outcome'] # y-labels
df = df.drop(columns=['outcome', 'pointsScored', 'pointsAllowed']) # x-data
print(df)
print(labels)
print(df.dtypes, "\n")



x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=None)
# x_train2, x_val2, y_train2, y_val2 = train_test_split(x_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2


clf = LogisticRegression(penalty='l2', max_iter=10000, random_state=1).fit(x_train, y_train)

y_val_pred = clf.predict(x_test)

# actual vs prediction
accuracyScore = accuracy_score(y_test, y_val_pred)
print(f'\nConfusion Matrix: \n{confusion_matrix(y_test, y_val_pred)}\n')
print("\nAccuracy: ", accuracyScore, "\n")