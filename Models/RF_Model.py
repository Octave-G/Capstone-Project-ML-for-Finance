import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np
import shap
import lime
import lime.lime_tabular

#                           Data
data = pd.read_csv('data_rf.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date', ascending=True)
data = data.set_index('Date')
data.dropna(inplace=True)

# Initialisation of variables
X = data.drop('US T-Note 10Y Yield', axis=1) # Dataset, the target data is removed from it
Y = data['US T-Note 10Y Yield'] # Target data to predict is 10Y U.S. T-Note

print('\t Data Head :', '\n', data.head(), '\n')
print('Data Shape :', data.shape)

# Train, Test Split
split_point = int(len(X) * 0.75) # 75% Train, 25% Test
X_train, X_test = X[:split_point], X[split_point:]
Y_train, Y_test = Y[:split_point], Y[split_point:]

n, p = X_test.shape[0], X_test.shape[1] # NB of samples and features in test set (Useful for R2 Adjusted)


#                           Random Forest Training
# Hyperparameters & Model Prediction
rf_model = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=1, max_features='sqrt', min_samples_split=10,
                                 bootstrap=True, max_samples=None, max_leaf_nodes=None, n_jobs=-1, oob_score=True)
rf_model.fit(X_train, Y_train)
rf_Y_pred = rf_model.predict(X_test)

# Metrics
RF_MSE = mean_squared_error(Y_test, rf_Y_pred)
RF_R2 = r2_score(Y_test, rf_Y_pred)
RF_R2_ADJ = 1 - (((1 - RF_R2) * (n - 1)) / (n - p - 1))
OOB_SCORE = rf_model.oob_score_ # Out-Of-Bag score estimates the model's generalization performance
rf_feature_importance = pd.Series(rf_model.feature_importances_, index=list(X.columns)).sort_values(ascending=False) # Table that show % of contribution of each data

print('\tRandom Forest Prediction Metrics :', '\n', f'\nOut-of-Bag Score: {round(OOB_SCORE, 3)}', f'\nMSE = {round(RF_MSE,3)} \nR2 = {round(RF_R2,3)} \nAdjusted R2 = {round(RF_R2_ADJ,3)}\n'); print('\nFeature Importance :\n', f'\n{rf_feature_importance}')

# Plot the Prediction
plt.figure(figsize=(12,6))
plt.plot(data.index, data['US T-Note 10Y Yield'], label='Historical Data', color='blue', alpha=0.4)
plt.plot(Y_test.index, rf_Y_pred, label='Prediction', color='red', alpha=1)
plt.title('Random Forest Prediction on 10Y T-Bills'); plt.xlabel('Date'); plt.ylabel('US 10Y Yield'); plt.legend(); plt.grid(True, alpha=0.3); plt.show()


#                           SHAP
print('\t SHAP :')

# Plots features contribution
rf_explainer_shap = shap.TreeExplainer(rf_model)
rf_shap_values = rf_explainer_shap(X_test)
shap.summary_plot(rf_shap_values.values, X_test, feature_names=X.columns, title='Summary Plot')

# Shows impact of a single feature on model prediction
feature_number = 5 # Choose the feature, ranges 0 to 10
shap.dependence_plot(feature_number, rf_shap_values.values, X_test, feature_names=X.columns, title='Dependence Plot')


#                           LIME
rf_explainer_lime = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, verbose=True, mode='regression')
i = np.random.randint(0, X_test.shape[0])
exp = rf_explainer_lime.explain_instance(X_test.values[i], rf_model.predict, num_features=len(X_test.columns))
exp.show_in_notebook(show_all=True)


#                         TREE PLOT
tree_position = 150 # Number specifies the tree position, goes up to 300
tree_to_plot = rf_model.estimators_[tree_position]

plt.figure(figsize=(20, 10))
plot_tree(decision_tree=tree_to_plot, feature_names=X_train.columns, filled=True, rounded=True, impurity=True, proportion=True, node_ids=True, fontsize=5)
plt.title(f'Decision Tree N.{tree_position} from Random Forest')
plt.show()