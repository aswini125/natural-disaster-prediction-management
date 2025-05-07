#!/usr/bin/env python
# coding: utf-8

# In[14]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load your dataset (example assumes it's a CSV file)
df = pd.read_csv('disasters.csv')  # Replace with your dataset path

# 2. Check the column names and data
print(df.columns)  # To ensure we have the correct columns
print(df.head())  # Preview the first few rows of data

# 3. Define features (X) and target (y)
X = df[['Entity', 'Year']]  # Features are 'Entity' and 'Year'
y = df['Deaths']  # Target is 'Deaths'

# 4. Handle categorical variable 'Entity' using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# 5. Standardize numerical features (e.g., 'Year')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Initialize RandomForestRegressor (for regression)
model = RandomForestRegressor(random_state=42)

# 8. Hyperparameter tuning (if needed)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],  # Adding None to allow for deeper trees
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [1.0, 'sqrt']
}

# Run grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# 9. Get best model from grid search
best_model = grid_search.best_estimator_

# 10. Evaluate the model
y_pred = best_model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# 11. Optionally: Cross-validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("Cross-Validation MSE Scores: ", -cv_scores)  # Since negative MSE is returned
print("Mean CV MSE: ", -cv_scores.mean())

# 12. Check target distribution (optional)
print("Target Distribution:\n", y.describe())

# 13. Visualize the results (optional)
plt.scatter(y_test, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True Values vs Predictions")
plt.show()

# 14. Feature importance (optional)
feature_importances = best_model.feature_importances_
print("Feature Importances:", feature_importances)


# In[ ]:




