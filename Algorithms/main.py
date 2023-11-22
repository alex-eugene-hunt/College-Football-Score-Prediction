import numpy as np
from flask import Flask, request, render_template
# Use algorithm.py if testing locally
#from algorithm import train_model
#from algorithm import predict1
#from algorithm import pull_schedule
#from algorithm import merge_schedules
#from algorithm import get_list_of_teams

# Use server_algorithm.py if using Google Cloud server
from server_algorithm import train_model
from server_algorithm import predict1
from server_algorithm import pull_schedule
from server_algorithm import merge_schedules
from server_algorithm import get_list_of_teams

from google.cloud import storage


app = Flask(__name__)

@app.route('/')
def home():
    data_filename = 'Full_Schedule.json'

    # Call the function to get the list of unique team names
    unique_team_list = get_list_of_teams(data_filename)

    return render_template('index.html', unique_team_list=unique_team_list)

@app.route('/predict', methods = ['POST'])
def predict():
    text_features = list(request.form.values())
    print(text_features)
    # Assume that the model can handle text input directly
    prediction = predict1(text_features[0], text_features[1], text_features[2], text_features[3])

    return render_template('index.html', prediction_text='Output {}'.format(prediction))

@app.route('/apipull')
def apipull():
    # This route should generally only be called by the Cron scheduler
    storage_client = storage.Client()
    bucket = storage_client.bucket("capstoneprediction.appspot.com")
    blob = bucket.blob('Testing.txt')

    # Use API to pull the newest schedule and save it, overwrite the file
    pull_schedule('New_Schedule.json')

    # Use the newest schedule to update the full JSON
    merge_schedules('Full_Schedule.json', 'New_Schedule.json')

    # Train the model again on the newest data
    train_model()

    return "Successfully pulled and updated schedule"

if __name__ == "__main__":
    app.run(debug=True)