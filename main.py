import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


# save filepath to variable for easier access
trendyol_file_path = "trendyol_data.csv"

# read the data and store data in DataFrame titled melbourne_data
trendyol_data = pd.read_csv(trendyol_file_path)

# dropna drops missing values (think of na as "not available")
trendyol_data = trendyol_data.dropna(axis=0)

trendyol_product_popularity = [
    "ModelCode",
    "Color",
    "TotalViews",
    "GrossFavoriteNumber",
    "TotalAddedtoCart",
    "GrossOrderQuantity",
]
trendyol_customer_behavior = [
    "ModelCode",
    "Color",
    "CancelledbyCustomer",
    "NumberofReturns",
]
trendyol_product_sales_performance = [
    "ModelCode",
    "Color",
    "GrossSalesQuantity",
    "GrossTurnover",
]

y = trendyol_data.GrossSalesQuantity
X = trendyol_data[trendyol_product_popularity]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Make copy to avoid changing original data
label_X_train = train_X.copy()
label_X_valid = val_X.copy()

# Categorical columns
object_cols = ["ModelCode", "Color"]

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
label_X_train[object_cols] = ordinal_encoder.fit_transform(train_X[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(val_X[object_cols])

# Try different leaf nodes to find the best MAE
leaf_node_mae = {}
for max_leaf_nodes in range(2, 200, 2):
    mae = get_mae(max_leaf_nodes, label_X_train, label_X_valid, train_y, val_y)
    leaf_node_mae[max_leaf_nodes] = mae

best_leaf_nodes_value = min(leaf_node_mae, key=leaf_node_mae.get)

forest_model = RandomForestRegressor(
    max_leaf_nodes=best_leaf_nodes_value, random_state=1
)
forest_model.fit(label_X_train, train_y)
trendyol_preds = forest_model.predict(label_X_valid)


# Evaluate model performance
mae = mean_absolute_error(val_y, trendyol_preds)
mse = mean_squared_error(val_y, trendyol_preds)
rmse = np.sqrt(mse)

print("Making predictions for the following 5 products:")
print(X.head())

print("Forest predictions are:")
print(trendyol_preds[:5])

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")