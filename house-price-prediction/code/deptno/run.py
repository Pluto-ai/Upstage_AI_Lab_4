import optuna
import pandas as pd
import lightgbm as lgb
import joblib  # 모델 저장을 위한 라이브러리
import json  # 추가: 파라미터 저장용
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 데이터 로드
X_train = pd.read_csv('data/X_train.5.2.csv', encoding='utf-8')
y_train = pd.read_csv('data/y_train.5.2.csv', encoding='utf-8')

# train/test split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=2023)

# y_train과 y_val을 1차원 Series로 변환
y_train = y_train.squeeze()
y_val = y_val.squeeze()

# 파라미터 및 결과 저장을 위한 파일 경로
param_file = 'best_params.json'
model_file = 'best_model.pkl'
all_pred_file = 'all_predictions.csv'
best_pred_file = 'best_predictions.csv'
result_file = 'results.txt'

# 모든 시도의 예측값을 저장하기 위한 DataFrame
all_predictions = pd.DataFrame()

def run():
    # Optuna 스터디 생성
    study = optuna.create_study(direction='minimize')  # RMSE 최소화를 목표로 설정
    study.optimize(objective, n_trials=100)  # 100번의 시도를 통해 최적 하이퍼파라미터 탐색

    # 최적의 하이퍼파라미터 출력 및 저장
    best_params = study.best_params
    print("Best parameters: ", best_params)
    
    # 최적 파라미터를 JSON 파일로 저장
    with open(param_file, 'w') as f:
        json.dump(best_params, f, indent=4)

    # 최적 파라미터로 모델 재학습
    best_model = lgb.LGBMRegressor(**best_params)
    best_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100)],
    )

    # 모델 저장 (pkl 파일)
    joblib.dump(best_model, model_file)

    # 검증 데이터에 대한 예측 및 RMSE 계산
    best_preds = best_model.predict(X_val)
    best_rmse = mean_squared_error(y_val, best_preds, squared=False)
    print(f"Best RMSE: {best_rmse}")

    # 최적 모델의 예측값을 CSV 파일로 저장
    pd.DataFrame(best_preds, columns=['predictions']).to_csv(best_pred_file, index=False)

    # RMSE와 Random State를 텍스트 파일로 저장
    with open(result_file, 'w') as f:
        f.write(f"Best RMSE: {best_rmse}\n")
        f.write(f"Best Parameters: {best_params}\n")
        f.write(f"Random State: 42\n")  # Random State는 고정값으로 설정한 경우

    # 모든 시도에 대한 예측값 저장
    all_predictions.to_csv(all_pred_file, index=False)

def objective(trial):
    # 하이퍼파라미터 검색 범위 설정
    param = {
        'device_type': 'gpu',  # GPU가 있을 시 사용, 없을 경우 'cpu'로 변경 가능
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 150),
        'max_depth': trial.suggest_int('max_depth', 5, 16),
        'min_child_weight': trial.suggest_float('min_child_weight', 5, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
        'random_state': 42  # Random State 고정
    }

    # 모델 생성 및 학습
    model = lgb.LGBMRegressor(**param)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100)],
    )

    # 검증 데이터에 대한 예측 및 평가 (RMSE 사용)
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)

    # 각 시도의 예측값과 RMSE를 기록
    all_predictions[f'trial_{trial.number}_preds'] = preds

    return rmse

if __name__ == '__main__':
    run()
