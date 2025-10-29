from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os

# Tạo ứng dụng Flask
app = Flask(__name__)

# Tải mô hình, scaler, và cột gốc đã lưu
try:
    # Kiểm tra file tồn tại trước khi load
    model_files = ['Linear.pkl', 'lasso.pkl', 'MLP.pkl', 'Stacking.pkl', 'scaler.pkl', 'columns.pkl']
    for file in model_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"File {file} không tồn tại!")
    
    linear_model = joblib.load('Linear.pkl')
    lasso_model = joblib.load('lasso.pkl')
    mlp_model = joblib.load('MLP.pkl')
    stacking_model = joblib.load('Stacking.pkl')
    scaler = joblib.load('scaler.pkl')
    expected_columns = joblib.load('columns.pkl')  # Tải cột gốc
    print("Tất cả models đã được load thành công!")
except Exception as e:
    print(f"Lỗi khi load models: {e}")
    # Có thể thêm logic để tạo models mặc định hoặc exit

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Kiểm tra xem models đã được load chưa
        if 'linear_model' not in globals():
            return render_template('index.html', error="Models chưa được load. Vui lòng chạy lại hello.py trước!")
        
        # Lấy dữ liệu từ form và validate
        form_data = {}
        required_fields = [
            'Max TemperatureC', 'Min TemperatureC', 'Dew PointC', 'Humidity',
            'Sea Level PressurehPa', 'VisibilityKm', 'Wind SpeedKm/h', 'Precipitationmm'
        ]
        
        for field in required_fields:
            value = request.form.get(field, '').strip()
            if not value:
                return render_template('index.html', error=f"Vui lòng nhập {field}")
            
            try:
                form_data[field] = float(value)
            except ValueError:
                return render_template('index.html', error=f"{field} phải là số hợp lệ")
        
        # Tạo data dictionary với các giá trị mặc định
        data = {
            'Max TemperatureC': form_data['Max TemperatureC'],
            'Min TemperatureC': form_data['Min TemperatureC'],
            'Dew PointC': form_data['Dew PointC'],
            'Humidity': form_data['Humidity'],
            'Sea Level PressurehPa': form_data['Sea Level PressurehPa'],
            'VisibilityKm': form_data['VisibilityKm'],
            'Wind SpeedKm/h': form_data['Wind SpeedKm/h'],
            'Precipitationmm': form_data['Precipitationmm'],
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
        prediction = round(linear_model.predict(data_scaled)[0], 2)
        prediction2 = round(lasso_model.predict(data_scaled)[0], 2)
        prediction3 = round(mlp_model.predict(data_scaled)[0], 2)
        prediction4 = round(stacking_model.predict(data_scaled)[0], 2)

        return render_template('index.html', prediction=prediction, prediction2=prediction2, prediction3=prediction3, prediction4=prediction4)

    except Exception as e:
        return render_template('index.html', error=f"Lỗi khi dự đoán: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
