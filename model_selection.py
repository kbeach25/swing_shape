import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv("tracking_data.csv")

# copied ttc formula from final.py
def calculate_time_to_contact(bat_speed, swing_length):
    if pd.isnull(bat_speed) or pd.isnull(swing_length) or bat_speed == 0:
        return np.nan
    ft_per_sec = bat_speed * 5280 / 3600
    return round(2 * swing_length / ft_per_sec, 3)

# engineer features again
df['vertical_bat_angle'] = df['vertical_swing_path'] * -1
df['ttc'] = df.apply(lambda row: calculate_time_to_contact(row['avg_swing_speed'], row['avg_swing_length']), axis=1)

df_filtered = df.dropna(subset=[
    'vertical_bat_angle', 'attack_angle', 'ttc', 'avg_swing_speed',
    'batting_avg', 'slg_percent', 'on_base_percent', 'k_percent', 'bb_percent'
])

X = df_filtered[['attack_angle', 'ttc', 'avg_swing_speed', 'vertical_bat_angle']]
y = df_filtered[['batting_avg', 'slg_percent', 'on_base_percent', 'k_percent', 'bb_percent']]
target_names = y.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# need performance scores for each target
def get_metrics_per_target(y_true, y_pred, model_name, formula):
    metrics = []
    for i, target in enumerate(target_names):
        metrics.append({
            'model': model_name,
            'target': target,
            'formula': formula,
            'R2': r2_score(y_true.iloc[:, i], y_pred[:, i]),
            'MAE': mean_absolute_error(y_true.iloc[:, i], y_pred[:, i]),
            'RMSE': np.sqrt(mean_squared_error(y_true.iloc[:, i], y_pred[:, i]))
        })
    return metrics

results = []

# Linear Regression
lr = MultiOutputRegressor(LinearRegression())
lr.fit(X_train, y_train)
results += get_metrics_per_target(y_test, lr.predict(X_test), 'Linear Regression', 'Multiple output linear regression')

# Rf
rf = MultiOutputRegressor(RandomForestRegressor(random_state=42))
rf.fit(X_train, y_train)
results += get_metrics_per_target(y_test, rf.predict(X_test), 'Random Forest', 'Ensemble of decision trees')

# svm
svr = MultiOutputRegressor(SVR(kernel='rbf'))
svr.fit(X_train, y_train)
results += get_metrics_per_target(y_test, svr.predict(X_test), 'Support Vector Regression', 'Non-linear SVR (RBF)')

# gradient boosting
gbr = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
gbr.fit(X_train, y_train)
results += get_metrics_per_target(y_test, gbr.predict(X_test), 'Gradient Boosting', 'Boosted decision trees')

# K-Nearest Neighbors
knn = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=5))
knn.fit(X_train, y_train)
results += get_metrics_per_target(y_test, knn.predict(X_test), 'K-Nearest Neighbors', 'Instance-based learning')

results_df = pd.DataFrame(results)
print("### Model Performance Per Target")
print(results_df)
results_df.to_csv("results_df_detailed.csv", index=False)

# see if different features can be predicted better by different models
metrics_df = results_df.copy()

# obp and walk rate are both too reliant on swing decision tendencies
key_targets = ['batting_avg', 'slg_percent', 'k_percent']

print("\n### Best Model Per Target (based on R2):")
for target in key_targets:
    best_row = metrics_df[metrics_df['target'] == target].sort_values(by='R2', ascending=False).iloc[0]
    print(f"\nTarget: {target}")
    print(f"  Best Model: {best_row['model']}")
    print(f"  Formula: {best_row['formula']}")
    print(f"  R2: {best_row['R2']:.4f}")
    print(f"  MAE: {best_row['MAE']:.4f}")
    print(f"  RMSE: {best_row['RMSE']:.4f}")
