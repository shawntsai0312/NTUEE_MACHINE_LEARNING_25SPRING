import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the data
train_data = pd.read_csv('ML2025Spring-hw2-public/train.csv')
test_data = pd.read_csv('ML2025Spring-hw2-public/test.csv')

# Separate features and target
X_train = train_data.drop(columns=['id', 'tested_positive_day3'])
y_train = train_data['tested_positive_day3']

# Preprocess the data
# Handle missing values (if any)
# Encode categorical variables (if any)
# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Define the models
ridge = Ridge()
lasso = Lasso()
elastic_net = ElasticNet()

# Define the parameter grids for hyperparameter tuning
ridge_params = {'alpha': np.logspace(-4, 4, 10)}
lasso_params = {'alpha': np.logspace(-4, 4, 10)}
elastic_net_params = {'alpha': np.logspace(-4, 4, 10), 'l1_ratio': np.linspace(0.1, 0.9, 9)}

# Cross-validation and hyperparameter tuning
ridge_cv = GridSearchCV(ridge, ridge_params, cv=5, scoring='neg_mean_squared_error', n_jobs=2)
lasso_cv = GridSearchCV(lasso, lasso_params, cv=5, scoring='neg_mean_squared_error', n_jobs=2)
elastic_net_cv = GridSearchCV(elastic_net, elastic_net_params, cv=5, scoring='neg_mean_squared_error', n_jobs=2)

# Train the models
ridge_cv.fit(X_train_scaled, y_train)
lasso_cv.fit(X_train_scaled, y_train)
elastic_net_cv.fit(X_train_scaled, y_train)

# Get the best models
best_ridge = ridge_cv.best_estimator_
best_lasso = lasso_cv.best_estimator_
best_elastic_net = elastic_net_cv.best_estimator_

# Preprocess the test data
X_test = test_data.drop(columns=['id'])
X_test_scaled = scaler.transform(X_test)

# Make predictions
ridge_pred = best_ridge.predict(X_test_scaled)
lasso_pred = best_lasso.predict(X_test_scaled)
elastic_net_pred = best_elastic_net.predict(X_test_scaled)

# Calculate weights based on cross-validation scores
ridge_score = -ridge_cv.best_score_
lasso_score = -lasso_cv.best_score_
elastic_net_score = -elastic_net_cv.best_score_

total_score = ridge_score + lasso_score + elastic_net_score
ridge_weight = ridge_score / total_score
lasso_weight = lasso_score / total_score
elastic_net_weight = elastic_net_score / total_score

# Ensemble prediction using weighted average
ensemble_pred = ridge_weight * ridge_pred + lasso_weight * lasso_pred + elastic_net_weight * elastic_net_pred

# Evaluate the model
mse = mean_squared_error(y_train, best_ridge.predict(X_train_scaled))
print(f'MSE: {mse}', file=sys.stdout)
sys.stdout.flush()

# Save predictions to submission.csv
submission = pd.DataFrame({'id': test_data['id'], 'tested_positive_day3': ensemble_pred})
submission.to_csv('submission.csv', index=False, header=True)