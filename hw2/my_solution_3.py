import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

# Load the data
train_df = pd.read_csv('ML2025Spring-hw2-public/train.csv')
test_df = pd.read_csv('ML2025Spring-hw2-public/test.csv')

# Separate features and target
X = train_df.drop(columns=['id', 'tested_positive_day3'])
y = train_df['tested_positive_day3']

# Try both StandardScaler and RobustScaler
scaler = RobustScaler()  # Use RobustScaler for robustness against outliers

# Define the models
ridge = Ridge()
lasso = Lasso(max_iter=5000)  # Increase max_iter for better convergence
elastic_net = ElasticNet(max_iter=5000)

# Define pipelines with scaling
pipeline_ridge = Pipeline([('scaler', scaler), ('ridge', ridge)])
pipeline_lasso = Pipeline([('scaler', scaler), ('lasso', lasso)])
pipeline_elastic_net = Pipeline([('scaler', scaler), ('elastic_net', elastic_net)])

# Expanded hyperparameter grid
param_grid_ridge = {'ridge__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
param_grid_lasso = {'lasso__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
param_grid_elastic_net = {
    'elastic_net__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'elastic_net__l1_ratio': [0.1, 0.5, 0.7, 0.9]
}

from sklearn.model_selection import KFold

# 建立具有洗牌功能的交叉驗證分割器
cv_splitter = KFold(n_splits=5, shuffle=True, random_state=42)

# 在 GridSearchCV 中使用這個分割器
grid_ridge = GridSearchCV(pipeline_ridge, param_grid_ridge, cv=cv_splitter, n_jobs=-1, scoring='neg_mean_squared_error')
grid_lasso = GridSearchCV(pipeline_lasso, param_grid_lasso, cv=cv_splitter, n_jobs=-1, scoring='neg_mean_squared_error')
grid_elastic_net = GridSearchCV(pipeline_elastic_net, param_grid_elastic_net, cv=cv_splitter, n_jobs=-1, scoring='neg_mean_squared_error')
# Fit the models
grid_ridge.fit(X, y)
grid_lasso.fit(X, y)
grid_elastic_net.fit(X, y)

# Print best parameters
print(f"Best Ridge parameters: {grid_ridge.best_params_}")
print(f"Best Lasso parameters: {grid_lasso.best_params_}")
print(f"Best ElasticNet parameters: {grid_elastic_net.best_params_}")

# Predict using the best models
y_pred_ridge = grid_ridge.predict(X)
y_pred_lasso = grid_lasso.predict(X)
y_pred_elastic_net = grid_elastic_net.predict(X)

# Compute MSE
mse_ridge = mean_squared_error(y, y_pred_ridge)
mse_lasso = mean_squared_error(y, y_pred_lasso)
mse_elastic_net = mean_squared_error(y, y_pred_elastic_net)

print(f"MSE Ridge: {mse_ridge}", file=sys.stdout.flush())
print(f"MSE Lasso: {mse_lasso}", file=sys.stdout.flush())
print(f"MSE ElasticNet: {mse_elastic_net}", file=sys.stdout.flush())

# Weighted ensemble (lower MSE models get higher weight)
weights = np.array([1/mse_ridge, 1/mse_lasso, 1/mse_elastic_net])
weights /= weights.sum()  # Normalize

y_pred_ensemble = (weights[0] * y_pred_ridge) + (weights[1] * y_pred_lasso) + (weights[2] * y_pred_elastic_net)

# Compute MSE of the ensemble
mse_ensemble = mean_squared_error(y, y_pred_ensemble)
print(f"MSE Ensemble: {mse_ensemble}", file=sys.stdout.flush())

# Make predictions on the test set
X_test = test_df.drop(columns=['id'])
y_pred_test = (weights[0] * grid_ridge.predict(X_test) +
               weights[1] * grid_lasso.predict(X_test) +
               weights[2] * grid_elastic_net.predict(X_test))

# Save predictions to CSV
submission_df = pd.DataFrame({'id': test_df['id'], 'tested_positive_day3': y_pred_test})
submission_df.to_csv('submission.csv', index=False)