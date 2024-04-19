# train_model.py

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
data = pd.read_csv('canteen_data.csv')

# Convert date and time columns to datetime objects
data['date'] = pd.to_datetime(data['date'])
data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S').dt.time

# Extract day of the week and time of the day (hour) from datetime
data['day_of_week'] = data['date'].dt.dayofweek
data['time_of_day'] = data['time'].apply(lambda x: x.hour)

# Selecting features and target variable
X = data[['Category', 'Menu_Item', 'day_of_week', 'time_of_day']]
y = data['Total_Item_Price']

# Preprocessing steps
categorical_features = ['Category', 'Menu_Item']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)],
    remainder='passthrough')

# Define model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# Define parameter grid for GridSearchCV
param_grid = {
    'regressor__n_estimators': [50, 100, 150],
    'regressor__learning_rate': [0.05, 0.1, 0.2],
    'regressor__max_depth': [3, 5, 7]
}

# Grid search for hyperparameter tuning
grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=2)

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
grid_search.fit(X_train, y_train)

# Evaluate model performance
train_mae = mean_absolute_error(y_train, grid_search.best_estimator_.predict(X_train))
test_mae = mean_absolute_error(y_test, grid_search.best_estimator_.predict(X_test))
print(f"Train MAE: {train_mae}")
print(f"Test MAE: {test_mae}")

# Save the trained model
joblib.dump(grid_search.best_estimator_, 'trained_model.pkl')
