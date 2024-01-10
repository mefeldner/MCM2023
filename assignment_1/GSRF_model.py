from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# Read the CSV file

# Reading data in - make sure to change csv file to have correct name on local machine
data = pd.read_csv('problemcData.csv',skiprows=1)  # Replace 'data.csv' with your actual file name
data['1 try'].value_counts()
print(data['1 try'].value_counts())

# Get target data
y = data['1 try']  # only for 1 try, target acts as label we need to predict
print(y)
# Load x variables into a pandas dataframe with columns

# !! Had to drop non integer columns - how can we convert this into values
X = data.drop(['1 try','Date','Word'], axis=1)
print(X)
# Divide into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)
X_train.shape, X_test.shape

# Definitley need to look at optimizing these hyperparameters

# Number of trees in random forest
n_estimators = [100]
# Number of features to consider at every train_test_split
max_features = [1]
# Max number of levels in trees
max_depth = [2, 4]
# Minimum number of samples required to split a node
min_samples_split = [2]
# Min samples at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True, False]


# Create the param grid (a dictionary)
param_grid = {
   'n_estimators': n_estimators,
   'max_features': max_features,
   'max_depth': max_depth,
   'min_samples_split': min_samples_split,
   'min_samples_leaf': min_samples_leaf,
   'bootstrap': bootstrap
}


# Print the param grid
print(param_grid)


# Create a RandomForestClassifier
rf_model = RandomForestClassifier()


# Create a GridSearchCV object
rf_grid = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=10, verbose=2, n_jobs=-1)


# Fit the GridSearchCV object to the training data
rf_grid.fit(X_train, y_train)


# Get the best parameters
best_params = rf_grid.best_params_
print("Best Parameters:", best_params)

# Print train and test accuracies
print(f'Train Accuracy: {rf_grid.score(X_train, y_train):.3f}')
print(f'Test Accuracy: {rf_grid.score(X_test, y_test):.3f}')