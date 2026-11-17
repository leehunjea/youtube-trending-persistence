from model_utils import *
from datetime import datetime

print(f"[{datetime.now().strftime('%H:%M:%S')}] ▶ CatBoost 학습 중...")
X_train, X_test, y_train, y_test, sw_train, sw_test, scale_pos, df = load_data()

cat_cols = ['category_id']
for col in cat_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col]  = X_test[col].astype(str)

model = get_catboost(cat_features=cat_cols)
model.fit(X_train, y_train)

X_all = pd.concat([X_train, X_test])
y_all = pd.concat([y_train, y_test])
for col in cat_cols:
    X_all[col] = X_all[col].astype(str)

# ✅ CatBoost는 CV 없이 평가 (clone 에러 우회)
metrics, _, _ = evaluate(model, X_test, y_test, "CatBoost",
                         X_all=None, y_all=None)
save_model(model, "CatBoost", metrics)
print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ CatBoost 완료!")
