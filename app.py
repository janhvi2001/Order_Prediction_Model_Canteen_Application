# app.py

from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model.pkl')

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    category = request.form['category']
    menu_item = request.form['menu_item']
    day_of_week = int(request.form['day_of_week'])
    time_of_day = int(request.form['time_of_day'])
    prediction = predict_orders(model, category, menu_item, day_of_week, time_of_day)
    return render_template('index.html', prediction=prediction)

def predict_orders(model, category, menu_item, day_of_week, time_of_day):
    # Create input data for prediction
    input_data = {'Category': category, 'Menu_Item': menu_item, 'Price': 0, 'Quantity': 0, 'day_of_week': day_of_week, 'time_of_day': time_of_day}
    input_df = pd.DataFrame([input_data])

    # Make prediction using the trained model
    predicted_orders = model.predict(input_df)[0]
    return round(predicted_orders, 2)  # Round the prediction to two decimal places

if __name__ == '__main__':
    app.run(debug=True)
