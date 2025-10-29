from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor 
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib


# Load the dataset to examine its structure
file_path = 'bangkok-2016.csv'
data = pd.read_csv(file_path)
print(data)

# Remove leading/trailing whitespaces from column names for consistency
data.columns = data.columns.str.strip()

# Drop unnecessary columns: 'ICT' (date), 'Events' (irrelevant for regression), and 'Max Gust SpeedKm/h' (has many NaN values)
data_cleaned = data.drop(columns=['ICT', 'Events', 'Max Gust SpeedKm/h'])

# Fill remaining missing values with the mean of each column
data_cleaned = data_cleaned.fillna(data_cleaned.mean())

# Confirm no missing values remain and check data types
missing_values_after_cleaning = data_cleaned.isnull().sum()
data_cleaned_info = data_cleaned.info()

missing_values_after_cleaning, data_cleaned_info

X = data_cleaned.drop(columns=['Mean TemperatureC'])
y = data_cleaned['Mean TemperatureC']

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra (80% huấn luyện, 20% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Hiển thị dữ liệu đã được làm sạch và hình dạng của tập huấn luyện/kiểm tra
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Khởi tạo mô hình hồi quy tuyến tính
linear_model = LinearRegression()

# Huấn luyện mô hình
linear_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_linear = linear_model.predict(X_test)

# Đánh giá mô hình
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print(f"Linear Regression MSE: {mse_linear}\n")
print(f"Linear Regression R-square: {r2_linear}\n")
print(f"Linear Regression MAE: {mae_linear}\n")
print(f"Linear Regression RMSE: {rmse_linear}\n")

joblib.dump(linear_model, 'Linear.pkl')

# Khởi tạo mô hình Lasso
lasso_model = Lasso(alpha=0.01)

# Huấn luyện mô hình
lasso_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_lasso = lasso_model.predict(X_test)

# Đánh giá mô hình
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f"Lasso Regression MSE: {mse_lasso}\n")
print(f"Lasso Regression R-square: {r2_lasso}\n")
print(f"Lasso Regression MAE: {mae_lasso}\n")
print(f"Lasso Regression RMSE: {rmse_lasso}\n")

joblib.dump(lasso_model, 'lasso.pkl')

# Khởi tạo MLPRegressor với các tham số phù hợp
mlp_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000,  early_stopping=True, validation_fraction=0.01, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Huấn luyện mô hình
mlp_model.fit(X_train_scaled, y_train)

# Dự đoán trên tập kiểm tra

y_pred_mlp = mlp_model.predict(X_test_scaled)

# Đánh giá mô hình
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
rmse_mlp = np.sqrt(mse_mlp)
mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)

# In kết quả đánh giá
print(f"MLPRegressor MSE: {mse_mlp}\n")
print(f"MLPRegressor RMSE: {rmse_mlp}\n")
print(f"MLPRegressor MAE: {mae_mlp}\n")
print(f"MLPRegressor R-square: {r2_mlp}\n")

joblib.dump(mlp_model, 'MLP.pkl')

# Sử dụng Stacking với các mô hình Linear Regression, Lasso, và MLPRegressor
estimators = [
    ('linear', LinearRegression()),
    ('lasso', Lasso(alpha=0.1)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(100,), max_iter=2500,  early_stopping=True, validation_fraction=0.01, random_state=42))
]
# Sử dụng lại scaler đã được fit từ MLP model
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())

# Huấn luyện mô hình Stacking
stacking_model.fit(X_train_scaled, y_train)

# Dự đoán trên tập kiểm tra bằng Stacking
y_pred_stacking = stacking_model.predict(X_test_scaled)
joblib.dump(scaler, 'scaler.pkl')

# Đánh giá mô hình Stacking
mse_stacking = mean_squared_error(y_test, y_pred_stacking)
rmse_stacking = np.sqrt(mse_stacking)
mae_stacking = mean_absolute_error(y_test, y_pred_stacking)
r2_stacking = r2_score(y_test, y_pred_stacking)

print(f"Stacking Regression MSE: {mse_stacking}\n")
print(f"Stacking Regression RMSE: {rmse_stacking}\n")
print(f"Stacking Regression MAE: {mae_stacking}\n")
print(f"Stacking Regression R-Square: {r2_stacking}\n")

joblib.dump(stacking_model, 'Stacking.pkl')

joblib.dump(X.columns, 'columns.pkl')  # Lưu thứ tự cột
