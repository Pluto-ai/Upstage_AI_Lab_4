
import pandas as pd
import numpy as np
import joblib  # 모델 로드를 위한 라이브러리
from sklearn.metrics import mean_squared_error

# 새로운 데이터를 로드 (파일 경로와 인코딩 설정에 유의)
X_train = pd.read_csv('data/X_train.5.2.csv', encoding='utf-8')
y_train = pd.read_csv('data/y_train.5.2.csv', encoding='utf-8')
X_test = pd.read_csv('data/X_test_6.csv', encoding='utf-8')

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)

# y_train를 1차원 Series로 변환
y_train = y_train.squeeze()

# 저장된 모델 불러오기
model_file = 'best_model.pkl'
loaded_model = joblib.load(model_file)

# 새로운 데이터로 예측
test_preds = np.exp(loaded_model.predict(X_train))

# 예측값 출력
print(f"Test predictions: {test_preds}")

# 예측값과 실제값 비교하여 RMSE 계산
test_rmse = mean_squared_error(y_train, test_preds, squared=False)
print(f"Test RMSE: {test_rmse}")

# 예측값을 CSV 파일로 저장
final_preds = np.exp(loaded_model.predict(X_test))
pd.DataFrame(final_preds.astype(int), columns=['target']).to_csv('final_predictions.csv', index=False)
print(f"done")
