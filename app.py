from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

# Tạo ứng dụng Flask
app = Flask(__name__)

# Tải mô hình, scaler, và cột gốc đã lưu
linear_model = joblib.load('Linear.pkl')
lasso_model = joblib.load('lasso.pkl')
mlp_model = joblib.load('MLP.pkl')
stacking_model = joblib.load('Stacking.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')  # Tải cột gốc

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu từ form
        data = {
            'Max TemperatureC': float(request.form.get('Max TemperatureC', 0)),
            'Min TemperatureC': float(request.form.get('Min TemperatureC', 0)),
            'Dew PointC': float(request.form.get('Dew PointC', 0)),
            'Humidity': float(request.form.get('Humidity', 0)),
            'Sea Level PressurehPa': float(request.form.get('Sea Level PressurehPa', 0)),
            'VisibilityKm': float(request.form.get('VisibilityKm', 0)),
            'Wind SpeedKm/h': float(request.form.get('Wind SpeedKm/h', 0)),
            'Precipitationmm': float(request.form.get('Precipitationmm', 0)),
            'CloudCover': 0,  # Đặt giá trị mặc định nếu cần
            'Max Humidity': 0,
            'Max Sea Level PressurehPa': 0,
            'Max VisibilityKm': 0,
            'Max Wind SpeedKm/h': 0
        }
        data_df = pd.DataFrame([data])

        # Đảm bảo đúng thứ tự cột như khi huấn luyện
        data_df = data_df.reindex(columns=expected_columns, fill_value=0)

        # Chuẩn hóa dữ liệu
        data_scaled = scaler.transform(data_df)

        # Dự đoán với từng mô hình
        prediction = linear_model.predict(data_scaled)[0]
        prediction2 = lasso_model.predict(data_scaled)[0]
        prediction3 = mlp_model.predict(data_scaled)[0]
        prediction4 = stacking_model.predict(data_scaled)[0]


        return render_template('index.html', prediction=prediction, prediction2=prediction2, prediction3=prediction3, prediction4=prediction4)

    except ValueError as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
