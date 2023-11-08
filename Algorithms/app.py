import numpy as np
from flask import Flask, request, render_template
from algorithm import train_model
from algorithm import predict1
import pickle


app = Flask(__name__)

##model = pickle.load(open('models/clf_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    text_features = list(request.form.values())
    print(text_features)
    # Assume that the model can handle text input directly
    train_model()
    prediction = predict1(text_features[0], text_features[1], text_features[2], text_features[3])

    ##output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Output {}'.format(prediction))

if __name__ == "__main__":
    app.run()