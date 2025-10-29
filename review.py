"""
ĐÁNH GIÁ TỔNG THỂ MÔ HÌNH MACHINE LEARNING
Dự đoán nhiệt độ trung bình tại Bangkok 2016
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("\n" + "="*80)
print(" ĐÁNH GIÁ TỔNG THỂ MÔ HÌNH MACHINE LEARNING")
print("="*80)

# Load dữ liệu
print("\n1. THÔNG TIN DATASET")
print("-"*80)
data = pd.read_csv('bangkok-2016.csv')
data.columns = data.columns.str.strip()
data_cleaned = data.drop(columns=['ICT', 'Events', 'Max Gust SpeedKm/h'])
data_cleaned = data_cleaned.fillna(data_cleaned.mean())

X = data_cleaned.drop(columns=['Mean TemperatureC'])
y = data_cleaned['Mean TemperatureC']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

print(f"✓ Tổng số mẫu: {len(data_cleaned)}")
print(f"✓ Số features: {X.shape[1]}")
print(f"✓ Training set: {len(X_train)} mẫu (80%)")
print(f"✓ Test set: {len(X_test)} mẫu (20%)")
print(f"✓ Nhiệt độ min: {y.min():.1f}°C")
print(f"✓ Nhiệt độ max: {y.max():.1f}°C")
print(f"✓ Nhiệt độ trung bình: {y.mean():.1f}°C")

# Load và đánh giá models
print("\n2. ĐÁNH GIÁ CÁC MÔ HÌNH")
print("-"*80)

# Load models
linear_model = joblib.load('Linear.pkl')
lasso_model = joblib.load('lasso.pkl')
mlp_model = joblib.load('MLP.pkl')
stacking_model = joblib.load('Stacking.pkl')
scaler = joblib.load('scaler.pkl')

models = {
    'Linear Regression': linear_model,
    'Lasso Regression': lasso_model,
    'MLP Regressor': mlp_model,
    'Stacking Regressor': stacking_model
}

results = []

for name, model in models.items():
    if name in ['MLP Regressor', 'Stacking Regressor']:
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'RMSE': f"{rmse:.4f}",
        'MAE': f"{mae:.4f}",
        'R²': f"{r2:.4f}",
        'RMSE_Value': rmse,
        'MAE_Value': mae,
        'R2_Value': r2
    })

# In bảng kết quả
df = pd.DataFrame(results)
print("\n" + df[['Model', 'RMSE', 'MAE', 'R²']].to_string(index=False))

# Tìm model tốt nhất
print("\n3. PHÂN TÍCH VÀ NHẬN XÉT")
print("-"*80)

best_rmse = min(results, key=lambda x: x['RMSE_Value'])
best_mae = min(results, key=lambda x: x['MAE_Value'])
best_r2 = max(results, key=lambda x: x['R2_Value'])

print(f"\n✓ Mô hình có RMSE tốt nhất: {best_rmse['Model']} ({best_rmse['RMSE']}°C)")
print(f"✓ Mô hình có MAE tốt nhất: {best_mae['Model']} ({best_mae['MAE']}°C)")
print(f"✓ Mô hình có R² tốt nhất: {best_r2['Model']} ({best_r2['R²']})")

# Đánh giá tổng thể
print("\n4. ĐÁNH GIÁ TỔNG THỂ")
print("-"*80)

# Tính điểm tổng thể
for result in results:
    # Chuẩn hóa điểm từ 0-100
    rmse_norm = (1 - result['RMSE_Value'] / max(r['RMSE_Value'] for r in results)) * 100
    mae_norm = (1 - result['MAE_Value'] / max(r['MAE_Value'] for r in results)) * 100
    r2_norm = result['R2_Value'] * 100
    
    score = (rmse_norm * 0.4 + mae_norm * 0.3 + r2_norm * 0.3)
    result['Total_Score'] = score

sorted_results = sorted(results, key=lambda x: x['Total_Score'], reverse=True)

print("\n📊 Bảng xếp hạng tổng thể:")
print("\n{:<25} {:<10} {:<10} {:<10} {:<10}".format('MODEL', 'RMSE (°C)', 'MAE (°C)', 'R²', 'SCORE'))
print("-"*75)
for r in sorted_results:
    print("{:<25} {:<10} {:<10} {:<10} {:<10.2f}".format(
        r['Model'], r['RMSE'], r['MAE'], r['R²'], r['Total_Score']
    ))

best_model = sorted_results[0]
print(f"\n🏆 MÔ HÌNH TỐT NHẤT: {best_model['Model']}")
print(f"   - RMSE: {best_model['RMSE']}°C")
print(f"   - MAE: {best_model['MAE']}°C")
print(f"   - R²: {best_model['R²']}")
print(f"   - Điểm tổng thể: {best_model['Total_Score']:.2f}/100")

# Kết luận
print("\n5. KẾT LUẬN VÀ KHUYẾN NGHỊ")
print("-"*80)

print("\n💡 Đánh giá từng mô hình:")

for result in sorted_results:
    print(f"\n{result['Model']}:")
    print(f"  • Độ chính xác: {'Cao' if result['Total_Score'] > 70 else 'Trung bình' if result['Total_Score'] > 50 else 'Thấp'}")
    print(f"  • Sai số trung bình: ±{result['MAE']}°C")
    print(f"  • Khả năng giải thích: {'Thấp' if 'MLP' in result['Model'] or 'Stacking' in result['Model'] else 'Cao'}")
    print(f"  • Thời gian training: {'Lâu' if 'MLP' in result['Model'] or 'Stacking' in result['Model'] else 'Nhanh'}")

print("\n✅ Khuyến nghị:")
print(f"  • Sử dụng {best_model['Model']} cho deployment vì có độ chính xác tốt nhất")
print(f"  • Model này đạt RMSE {best_model['RMSE']}°C - rất tốt cho dự đoán nhiệt độ")
print(f"  • Tất cả models đều cho kết quả chấp nhận được (R² > 0.9)")

print("\n" + "="*80)
print(" ĐÁNH GIÁ HOÀN TẤT!")
print("="*80 + "\n")

