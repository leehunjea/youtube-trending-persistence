from model_utils import *
from datetime import datetime

print(f"[{datetime.now().strftime('%H:%M:%S')}] ▶ LightGBM 학습 중...")
X_train, X_test, y_train, y_test, sw_train, sw_test, scale_pos, df = load_data()

model = get_lightgbm()
model.fit(X_train, y_train)

X_all = pd.concat([X_train, X_test])
y_all = pd.concat([y_train, y_test])
metrics, _, _ = evaluate(model, X_test, y_test, "LightGBM", X_all, y_all)
save_model(model, "LightGBM", metrics)
print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ LightGBM 완료!")
