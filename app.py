# Importing essential libraries and modules

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, Markup
import os
import tensorflow
import numpy as np
import pandas as pd
import requests
import pickle
import io
import torch
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# define a list of all possible states and crops
all_states = ['Andaman and Nicobar Islands', 'Andhra Pradesh',
       'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh',
       'Chhattisgarh', 'Dadra and Nagar Haveli', 'Goa', 'Gujarat',
       'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir ', 'Jharkhand',
       'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
       'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Puducherry',
       'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana ',
       'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']
all_crops = ['Arecanut', 'Other Kharif pulses', 'Rice', 'Banana', 'Cashewnut',
       'Coconut ', 'Dry ginger', 'Sugarcane', 'Sweet potato', 'Tapioca',
       'Black pepper', 'Dry chillies', 'other oilseeds', 'Turmeric',
       'Maize', 'Moong(Green Gram)', 'Urad', 'Arhar/Tur', 'Groundnut',
       'Sunflower', 'Bajra', 'Castor seed', 'Cotton(lint)', 'Horse-gram',
       'Jowar', 'Korra', 'Ragi', 'Tobacco', 'Gram', 'Wheat', 'Masoor',
       'Sesamum', 'Linseed', 'Safflower', 'Onion', 'other misc. pulses',
       'Samai', 'Small millets', 'Coriander', 'Potato',
       'Other  Rabi pulses', 'Soyabean', 'Beans & Mutter(Vegetable)',
       'Bhindi', 'Brinjal', 'Citrus Fruit', 'Cucumber', 'Grapes', 'Mango',
       'Orange', 'other fibres', 'Other Fresh Fruits', 'Other Vegetables',
       'Papaya', 'Pome Fruit', 'Tomato', 'Mesta', 'Cowpea(Lobia)',
       'Lemon', 'Pome Granet', 'Sapota', 'Cabbage', 'Rapeseed &Mustard',
       'Peas  (vegetable)', 'Niger seed', 'Bottle Gourd', 'Varagu',
       'Garlic', 'Ginger', 'Oilseeds total', 'Pulses total', 'Jute',
       'Peas & beans (Pulses)', 'Blackgram', 'Paddy', 'Pineapple',
       'Barley', 'Sannhamp', 'Khesari', 'Guar seed', 'Moth',
       'Other Cereals & Millets', 'Cond-spcs other', 'Turnip', 'Carrot',
       'Redish', 'Arcanut (Processed)', 'Atcanut (Raw)',
       'Cashewnut Processed', 'Cashewnut Raw', 'Cardamom', 'Rubber',
       'Bitter Gourd', 'Drum Stick', 'Jack Fruit', 'Snak Guard', 'Tea',
       'Coffee', 'Cauliflower', 'Other Citrus Fruit', 'Water Melon',
       'Total foodgrain', 'Kapas', 'Colocosia', 'Lentil', 'Bean',
       'Jobster', 'Perilla', 'Rajmash Kholar', 'Ricebean (nagadal)',
       'Ash Gourd', 'Beet Root', 'Lab-Lab', 'Ribed Guard', 'Yam',
       'Pump Kin', 'Apple', 'Peach', 'Pear', 'Plums', 'Litchi', 'Ber',
       'Other Dry Fruit', 'Jute & mesta']


# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

#Loading the Trained Random Forest Model For Crop Prediction
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
REC_MODEL = pickle.load(open(os.path.join(MODEL_DIR, 'RFprediction.pkl'), 'rb'))

# Loading the Trained Random Forest Model For Crop Yield Prediction
with open('models/RFyield.pkl', 'rb') as f:
    model = pickle.load(f)

# render home page


@ app.route('/')
def home():
    title = 'AgroMate - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'AgroMate - Crop Prediction'
    return render_template('crop.html', title=title)

@app.route('/crop-recommend', methods=['GET', 'POST'])
def cr():
    title = 'AgroMate - Crop Prediction'
    if request.method == 'POST':
        X = []
        if request.form.get('nitrogen'):
            X.append(float(request.form.get('nitrogen')))
        if request.form.get('phosphorous'):
            X.append(float(request.form.get('phosphorous')))
        if request.form.get('potassium'):
            X.append(float(request.form.get('potassium')))
        if request.form.get('temperature'):
            X.append(float(request.form.get('temperature')))
        if request.form.get('humidity'):
            X.append(float(request.form.get('humidity')))
        if request.form.get('ph'):
            X.append(float(request.form.get('ph')))
        if request.form.get('rainfall'):
            X.append(float(request.form.get('rainfall')))
        X = np.array(X)
        X = X.reshape(1, -1)
        res = REC_MODEL.predict(X)[0]
        # print(res)
        return render_template('crop-result.html', prediction=res, title=title)
    return render_template('crop.html')

@ app.route('/crop_yield')
def crop_yield():
    title = 'AgroMate - Crop Yield Prediction'
    return render_template('yield.html', title=title, all_states=all_states, all_crops=all_crops)


@app.route('/crop_yield', methods=['POST'])
def predict():
    # get the input values from the HTML form
    state = request.form['State']
    crop = request.form['Crop']
    area = float(request.form['Area'])
    production = float(request.form['Production'])


    # perform one-hot encoding on the state input
    state_encoded = [1 if state ==  s.title().replace(" ", "_") else 0 for s in all_states]

    # perform one-hot encoding on the crop input
    crop_encoded = [1 if crop ==  c.title().replace(" ", "_") else 0 for c in all_crops]

    # combine the encoded state and crop inputs with the numerical inputs
    input_variables = [area, production] + state_encoded + crop_encoded


    # perform prediction using the trained model
    yield_prediction = model.predict(np.array(input_variables).reshape(1, -1))[0]
    yield_prediction1=round(yield_prediction,4)
    # render the prediction in a new HTML page
    return render_template('yield-result.html', yield_prediction=yield_prediction1,crop=crop,state=state)

#video page
@app.route('/help')
def help():
    title = 'AgroMate - Help'
    return render_template('help.html',title=title)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
