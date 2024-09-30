import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from ucimlrepo import fetch_ucirepo


combined_cycle_power_plant = fetch_ucirepo(id=294)


X = combined_cycle_power_plant.data.features
y = combined_cycle_power_plant.data.targets


print("Metadata:\n", combined_cycle_power_plant.metadata)
print("\nVariables:\n", combined_cycle_power_plant.variables)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2 Score): {r2:.4f}")


feature_importance = model.feature_importances_
features = X.columns
for i, v in enumerate(feature_importance):
    print(f"Feature: {features[i]}, Score: {v:.5f}")
