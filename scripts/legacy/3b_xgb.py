from model_utils import *
from datetime import datetime

print(f"[{datetime.now().strftime('%H:%M:%S')}] ▶ XGBoost 학습 중...")
X_train, X_test, y_train, y_test, sw_train, sw_test, scale_pos, df = load_data()

model = get_xgboost(scale_pos_weight=scale_pos)
model.fit(X_train, y_train)

X_all = pd.concat([X_train, X_test])
y_all = pd.concat([y_train, y_test])
metrics, _, _ = evaluate(model, X_test, y_test, "XGBoost", X_all, y_all)
save_model(model, "XGBoost", metrics)
print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ XGBoost 완료!")
