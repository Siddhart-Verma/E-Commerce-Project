'''
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from pymongo import MongoClient
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

client = MongoClient('mongodb://localhost:27017/')
db = client['demand_forecasting_db']
users_collection = db['users']
logs_collection = db['prediction_logs']


store_df = pd.read_csv('Cleaned_raw_dataset_2/clean_store_dataset.csv')
item_df = pd.read_csv('Cleaned_raw_dataset_2/clean_item_dataset.csv')


model = joblib.load('final_lgbm_model.pkl')


def get_dropdown_data():
    store_ids = sorted(store_df['store_nbr'].unique())
    cities = sorted(store_df['city'].unique())
    items = sorted(item_df['item_nbr'].unique())  # or 'family' if you use family
    onpromotion_options = ['0', '1']
    return store_ids, cities, items, onpromotion_options


def calculate_features(store_id, item_nbr, date, onpromotion, city):
    date = pd.to_datetime(date)

    # Basic features (you can extend more if needed)
    features = {
        'store_nbr': int(store_id),
        'item_nbr': int(item_nbr),
        'onpromotion': int(onpromotion),
        'city': city,
        'day': date.day,
        'month': date.month,
        'year': date.year,
        'day_of_week': date.weekday()
    }

    # You can also add rolling means, lags etc. if needed
    return pd.DataFrame([features])


# Home Page (Prediction)
@app.route('/')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    store_ids, cities, items, onpromotion_options = get_dropdown_data()
    return render_template('index.html',
                           store_ids=store_ids,
                           cities=cities,
                           items=items,
                           onpromotion_options=onpromotion_options)

# Handle Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    # Get form data
    store_id = request.form['store_id']
    item_nbr = request.form['item_nbr']
    city = request.form['city']
    onpromotion = request.form['onpromotion']
    date = request.form['date']

    # Feature generation
    input_features = calculate_features(store_id, item_nbr, date, onpromotion, city)

    # Predict
    prediction = model.predict(input_features)[0]

    # Save to MongoDB
    log = {
        'username': session['username'],
        'store_id': store_id,
        'item_nbr': item_nbr,
        'city': city,
        'onpromotion': onpromotion,
        'date': date,
        'predicted_sales': float(prediction),
        'timestamp': datetime.now()
    }
    logs_collection.insert_one(log)

    # Render result
    store_ids, cities, items, onpromotion_options = get_dropdown_data()
    return render_template('index.html',
                           store_ids=store_ids,
                           cities=cities,
                           items=items,
                           onpromotion_options=onpromotion_options,
                           prediction=round(prediction, 2),
                           input_data=log)

# Signup
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        existing_user = users_collection.find_one({'username': username})
        if existing_user:
            return 'Username already exists. Try logging in.'

        hashed_pw = generate_password_hash(password)
        users_collection.insert_one({'username': username, 'password': hashed_pw})
        return redirect(url_for('login'))
    
    return render_template('signup.html')

# Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = users_collection.find_one({'username': username})
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return 'Invalid credentials. Try again.'

    return render_template('login.html')

# Logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
'''


from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import joblib
from datetime import datetime, timedelta
import os

# -------------------------------
# Step 1: Initialize app and DB
# -------------------------------
app = Flask(__name__)
app.secret_key = "mysecretkey"

client = MongoClient("mongodb://localhost:27017/")
db = client['forecast_db']
users_collection = db['users']
predictions_collection = db['predictions']

# -------------------------------
# Step 2: Load model and data
# -------------------------------
model = joblib.load('final_lgbm_model.pkl')
item_df = pd.read_csv('Cleaned_raw_dataset_2/clean_item_dataset.csv')
store_df = pd.read_csv('Cleaned_raw_dataset_2/clean_store_dataset.csv')

# Dropdown options
store_ids = sorted(store_df['store_nbr'].unique().tolist())
onpromotion_options = [0, 1]
family_list = sorted(item_df['family'].unique().tolist())

# -------------------------------
# Step 3: Routes
# -------------------------------

@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html',
                           store_ids=store_ids,
                           families=family_list,
                           onpromotion_values=onpromotion_options)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        if users_collection.find_one({'email': email}):
            return render_template('signup.html', message='User already exists!')

        hashed_password = generate_password_hash(password)
        users_collection.insert_one({'email': email, 'password': hashed_password})
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = users_collection.find_one({'email': email})

        if user and check_password_hash(user['password'], password):
            session['user'] = email
            return redirect(url_for('index'))
        return render_template('login.html', message='Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# -------------------------------
# Step 4: Prediction Logic
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    # Get form inputs
    store_id = int(request.form['store_id'])
    family = request.form['family']
    date_str = request.form['date']
    onpromotion = int(request.form['onpromotion'])

    # Convert date to datetime
    date = pd.to_datetime(date_str)

    # ---------------------------
    # Build feature dataframe
    # ---------------------------
    # Example simple features (you can add more historical logic here)
    features = pd.DataFrame([{
        'store_nbr': store_id,
        'family': family,
        'onpromotion': onpromotion,
        'day': date.day,
        'month': date.month,
        'year': date.year,
        'day_of_week': date.weekday()
    }])

    # If label encoding was used, do the same here
    # Example: encode family
    from sklearn.preprocessing import LabelEncoder
    le_family = LabelEncoder()
    le_family.fit(item_df['family'])
    features['family'] = le_family.transform(features['family'])

    # Predict
    prediction = model.predict(features, predict_disable_shape_check=True)[0]

    # Save to MongoDB 
    predictions_collection.insert_one({
        'user_email': session['user'],
        'store_id': store_id,
        'family': family,
        'date': date_str,
        'onpromotion': onpromotion,
        'predicted_sales': float(prediction)
    })

    return render_template('index.html',
                           prediction=round(prediction, 2),
                           store_ids=store_ids,
                           families=family_list,
                           onpromotion_values=onpromotion_options,
                           selected_store=store_id,
                           selected_family=family,
                           selected_date=date_str,
                           selected_onpromotion=onpromotion)

# -------------------------------
# Run the app
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)
