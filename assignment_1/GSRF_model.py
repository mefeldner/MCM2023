import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = pd.read_csv('wordAttributes.csv')


# establish target var, train and test split
print(data.columns)
X = data[['fWord', 'fLetter', 'rep']]
# Remove error for prediction
X = X.values
print(X)
y = data['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# set up hyperparams for the grid search
param_grid = {
   'n_estimators': [10, 50, 100],
   'max_depth': [None, 10, 20],
   'min_samples_split': [2, 5, 10],
   'min_samples_leaf': [1, 2, 4]
}


rf_model = RandomForestRegressor()


# Create GridSearchCV object
rf_grid = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)


rf_grid.fit(X_train, y_train)


# get best params
best_params = rf_grid.best_params_
print("Best Parameters:", best_params)


# predictions on test set
y_pred = rf_grid.predict(X_test)


# evaluate performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Set: {mse:.3f}')


mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error on Test Set: {mae:.3f}')


# Predict for EERIE by inputting values for fWord, fLetter, and rep
example_values = [[0.00024, 0.418799, 1.5]]
predictions = rf_grid.predict(example_values)
print("Predicted Score:", predictions)
