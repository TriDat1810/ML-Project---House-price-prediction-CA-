import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing #Thư viện sklearn.datasets có sẵn nhiều bộ dữ liệu mẫu, ví dụ: fetch_california_housing, load_iris, load_digits, load_boston, ...
from sklearn.model_selection import train_test_split #Model selection: train_test_split để chia dữ liệu thành tập huấn luyện và tập kiểm tra
from sklearn.linear_model import LinearRegression # Mô hình hồi quy tuyến tính
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # Các metric để đánh giá model
from sklearn.ensemble import RandomForestRegressor # Mô hình rừng ngẫu nhiên (Random Forest)

# Load dataset, as_frame=True để trả về dạng pandas DataFrame luôn
# (thay vì mảng numpy thô, khó đọc)

housing = fetch_california_housing(as_frame=True)
df = housing.frame # Vì fetch_california_housing(as_frame=True) có rất nhiều mục nên việc lấy ra DataFrame và gán cho 1 biến riêng sẽ dễ đọc hơn

#print(df.head())      # xem 5 dòng data đầu tiền, từ 0 - 4
#print(df.shape)       # Tổng số đếm cho dòng và cột, ví dụ 500, 4 có nghĩa là 500 dòng và 4 cột
#print(df.columns)     # Tên của các cột

# 1. Xem thống kê mô tả: mean, std, min, max, các mốc phần trăm (25%, 50%, 75%)
print(df.describe())
print(df.isnull().sum()) # Kiểm tra xem có giá trị null nào không, nếu có thì sẽ trả về số lượng giá trị null của từng cột

# Tách features (X) và target (y) trước
X = df.drop(columns=["MedHouseVal"])   # X = tất cả cột trừ giá nhà
y = df["MedHouseVal"]                   # y = chỉ cột giá nhà

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 #test_size=0.2 nghĩa là 20% dữ liệu sẽ được dùng làm tập kiểm tra, còn lại 80% là tập huấn luyện. random_state=42 để đảm bảo việc chia dữ liệu là ngẫu nhiên nhưng có thể tái tạo được kết quả
)

print(X_train.shape)
print(X_test.shape)

# Bước 1: tạo 1 model rỗng (chưa học gì cả)
#model = LinearRegression()

# Bước 2: cho model học từ tập train
# .fit() chính là bước "training" — model tự tìm ra w1, w2, w3... và b
# tối ưu nhất, sao cho dự đoán trên X_train gần với y_train nhất có thể
#model.fit(X_train, y_train)

#print("Model đã học xong!")

#y_pred = model.predict(X_test)

#print(y_pred[:5])   # xem thử 5 giá trị đầu model đoán
#print(y_test[:5].values)   # so với 5 giá trị thật tương ứng

#mae = mean_absolute_error(y_test, y_pred)
#rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#r2 = r2_score(y_test, y_pred)

#print(f"MAE: {mae:.3f}")
#print(f"RMSE: {rmse:.3f}")
#print(f"R²: {r2:.3f}")

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print(f"MAE: {mae_rf:.3f}")
print(f"RMSE: {rmse_rf:.3f}")
print(f"R²: {r2_rf:.3f}")

importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print(importances)