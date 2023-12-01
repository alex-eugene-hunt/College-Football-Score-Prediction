import numpy as np
from flask import Flask, request, render_template
from algorithm import train_model
from algorithm import predict1
import pickle
from algorithm import get_list_of_teams

app = Flask(__name__)
@app.route('/')
def home():
    data_filename = 'Full_Schedule.json'
    
    # Call the function to get the list of unique team names
    unique_team_list = get_list_of_teams(data_filename)
    #print(unique_team_list)
    return render_template('index.html', unique_team_list=unique_team_list)

@app.route('/predict', methods = ['POST'])
def predict():
    text_features = list(request.form.values())
    print(text_features)
    # Assume that the model can handle text input directly
    train_model()
    prediction = predict1(text_features[0], text_features[1], text_features[2], text_features[3])

    return render_template('index.html', prediction_text='Output {}'.format(prediction))

if __name__ == "__main__":
    app.run()