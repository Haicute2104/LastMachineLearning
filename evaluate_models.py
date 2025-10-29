"""
Script đánh giá và so sánh tổng thể các mô hình Machine Learning
"""
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

print("=" * 70)
print("ĐÁNH GIÁ TỔNG THỂ CÁC MÔ HÌNH MACHINE LEARNING")
print("=" * 70)

# Load dữ liệu
print("\n1. LOADING DATA...")
print("-" * 70)
data = pd.read_csv('bangkok-2016.csv')
data.columns = data.columns.str.strip()
data_cleaned = data.drop(columns=['ICT', 'Events', 'Max Gust SpeedKm/h'])
data_cleaned = data_cleaned.fillna(data_cleaned.mean())

X = data_cleaned.drop(columns=['Mean TemperatureC'])
y = data_cleaned['Mean TemperatureC']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

print(f"✓ Dataset shape: {data_cleaned.shape}")
print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")
print(f"✓ Features: {X_train.shape[1]}")
print(f"✓ Target variable: Mean Temperature (°C)")
print(f"✓ Target range: {y_test.min():.1f}°C - {y_test.max():.1f}°C")

# Load models
print("\n2. LOADING TRAINED MODELS...")
print("-" * 70)

models = {}
metrics = {}

try:
    # Load Linear Regression
    models['Linear Regression'] = joblib.load('Linear.pkl')
    print("✓ Linear Regression model loaded")
    
    # Load Lasso Regression
    models['Lasso Regression'] = joblib.load('lasso.pkl')
    print("✓ Lasso Regression model loaded")
    
    # Load MLP Regressor
    models['MLP Regressor'] = joblib.load('MLP.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✓ MLP Regressor model loaded")
    print("✓ StandardScaler loaded")
    
    # Load Stacking Regressor
    models['Stacking Regressor'] = joblib.load('Stacking.pkl')
    print("✓ Stacking Regressor model loaded")
    
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    print("Please run hello.py first to train the models!")
    exit(1)

# Evaluate models
print("\n3. EVALUATING MODELS...")
print("-" * 70)

for model_name, model in models.items():
    print(f"\n📊 Evaluating {model_name}...")
    
    # Prepare data based on model type
    if model_name in ['MLP Regressor', 'Stacking Regressor']:
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate additional metrics
    mean_abs_percent_error = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    max_error = np.max(np.abs(y_test - y_pred))
    min_error = np.min(np.abs(y_test - y_pred))
    std_error = np.std(y_test - y_pred)
    
    metrics[model_name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mean_abs_percent_error,
        'Max Error': max_error,
        'Min Error': min_error,
        'Std Error': std_error
    }
    
    print(f"  ✓ MSE:  {mse:.4f}")
    print(f"  ✓ RMSE: {rmse:.4f}°C")
    print(f"  ✓ MAE:  {mae:.4f}°C")
    print(f"  ✓ R²:   {r2:.4f}")
    print(f"  ✓ MAPE: {mean_abs_percent_error:.2f}%")

# Summary table
print("\n" + "=" * 70)
print("📈 SUMMARY TABLE - MODEL COMPARISON")
print("=" * 70)

# Create summary DataFrame
summary_df = pd.DataFrame(metrics).T

# Round values
summary_df = summary_df.round(4)

print("\n" + summary_df.to_string())

# Find best models
print("\n" + "=" * 70)
print("🏆 BEST MODELS BY METRIC")
print("=" * 70)

best_rmse = summary_df['RMSE'].idxmin()
best_mae = summary_df['MAE'].idxmin()
best_r2 = summary_df['R²'].idxmax()
best_mape = summary_df['MAPE'].idxmin()

print(f"\n✓ Best RMSE: {best_rmse} ({summary_df.loc[best_rmse, 'RMSE']:.4f}°C)")
print(f"✓ Best MAE:  {best_mae} ({summary_df.loc[best_mae, 'MAE']:.4f}°C)")
print(f"✓ Best R²:   {best_r2} ({summary_df.loc[best_r2, 'R²']:.4f})")
print(f"✓ Best MAPE: {best_mape} ({summary_df.loc[best_mape, 'MAPE']:.2f}%)")

# Model recommendations
print("\n" + "=" * 70)
print("💡 RECOMMENDATIONS")
print("=" * 70)

# Analyze which model is overall best
# Give more weight to RMSE and R²
weighted_scores = {}
for model_name in summary_df.index:
    # Normalize metrics (lower is better for RMSE, MAE, MAPE; higher is better for R²)
    rmse_score = 1 / (summary_df.loc[model_name, 'RMSE'] / summary_df['RMSE'].min())
    mae_score = 1 / (summary_df.loc[model_name, 'MAE'] / summary_df['MAE'].min())
    r2_score = summary_df.loc[model_name, 'R²'] / summary_df['R²'].max()
    mape_score = 1 / (summary_df.loc[model_name, 'MAPE'] / summary_df['MAPE'].min())
    
    # Weighted average (higher weight on RMSE and R²)
    weighted_score = (rmse_score * 0.4 + mae_score * 0.2 + r2_score * 0.3 + mape_score * 0.1)
    weighted_scores[model_name] = weighted_score

best_overall = max(weighted_scores, key=weighted_scores.get)

print(f"\n🎯 Overall Best Model: {best_overall}")
print(f"\n📝 Reasons:")
print(f"  • {best_overall} has the best balance of accuracy and robustness")
print(f"  • RMSE: {summary_df.loc[best_overall, 'RMSE']:.4f}°C")
print(f"  • R² Score: {summary_df.loc[best_overall, 'R²']:.4f}")

print(f"\n💼 Model Characteristics:")
for model_name in summary_df.index:
    print(f"\n  {model_name}:")
    print(f"    • Complexity: {'High' if 'Stacking' in model_name or 'MLP' in model_name else 'Low'}")
    print(f"    • Training Time: {'Long' if 'MLP' in model_name or 'Stacking' in model_name else 'Fast'}")
    print(f"    • Prediction Time: {'Medium' if 'Stacking' in model_name or 'MLP' in model_name else 'Fast'}")
    print(f"    • Interpretability: {'Low' if 'MLP' in model_name or 'Stacking' in model_name else 'High'}")

# Model comparison chart
print("\n" + "=" * 70)
print("📊 PERFORMANCE RANKING")
print("=" * 70)

# Sort by weighted score
sorted_models = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
for i, (model_name, score) in enumerate(sorted_models, 1):
    print(f"{i}. {model_name} (Score: {score:.4f})")

print("\n" + "=" * 70)
print("✅ EVALUATION COMPLETE!")
print("=" * 70)

