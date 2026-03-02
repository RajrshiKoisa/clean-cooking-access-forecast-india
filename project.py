import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

file_path = "P_Data_Extract_From_World_Development_Indicators (1).xlsx"

df = pd.read_excel(r"C:\Users\rajrs\OneDrive\Desktop\AI for sustainability\dataset.xlsx")
print(df.head())
print(df.columns)

df = df[df["Country Name"] == "India"].copy()

id_cols = ["Series Name", "Series Code"]
year_cols = [c for c in df.columns if "[YR" in c]

df = df[id_cols + year_cols]
long_df = df.melt(
    id_vars=["Series Name", "Series Code"],
    value_vars=year_cols,
    var_name="Year",
    value_name="Value"
)

long_df["Year"] = long_df["Year"].str.extract(r"(\d{4})").astype(int)
long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
wide_df = long_df.pivot_table(
    index="Year",
    columns="Series Name",
    values="Value"
).reset_index()


print(wide_df.head())

TARGET = "Access to clean fuels and technologies for cooking (% of population)"

print("Target present:", TARGET in wide_df.columns)


wide_df = wide_df.sort_values("Year")
wide_df = wide_df.dropna(subset=[TARGET])

missing_ratio = wide_df.isna().mean()
keep_cols = missing_ratio[missing_ratio < 0.4].index

wide_df = wide_df[keep_cols]

wide_df = wide_df.sort_values("Year")
wide_df = wide_df.set_index("Year")

wide_df = wide_df.interpolate(method="linear")

wide_df = wide_df.dropna()

print(wide_df.shape)


plt.figure(figsize=(8,5))
plt.plot(wide_df.index, wide_df[TARGET])
plt.xlabel("Year")
plt.ylabel("Clean cooking access (%)")
plt.title("Access to clean cooking fuels in India")
plt.show()

X = wide_df.drop(columns=[TARGET])
y = wide_df[TARGET]

years = wide_df.index.sort_values()
split_year = years[int(len(years)*0.8)]

X_train = X[X.index <= split_year]
X_test  = X[X.index > split_year]

y_train = y[y.index <= split_year]
y_test  = y[y.index > split_year]



lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("\nLinear Regression results")
print("MAE :", mean_absolute_error(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("R2  :", r2_score(y_test, y_pred_lr))



rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\nRandom Forest results")
print("MAE :", mean_absolute_error(y_test, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("R2  :", r2_score(y_test, y_pred_rf))



importances = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 important indicators:")
print(importances.head(10))


plt.figure(figsize=(8,5))
importances.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 important indicators")
plt.xlabel("Importance")
plt.show()

base_year = X.iloc[[-1]].copy()

top_feature = importances.index[0]
print("Most important feature:", top_feature)

scenario = base_year.copy()
scenario[top_feature] = scenario[top_feature] * 1.10

base_pred = rf.predict(base_year)[0]
scenario_pred = rf.predict(scenario)[0]

print("\nBase prediction:", base_pred)
print("After 10% increase in", top_feature, ":", scenario_pred)
print("Change:", scenario_pred - base_pred)