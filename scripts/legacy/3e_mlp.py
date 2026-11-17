from model_utils import *
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime

print(f"[{datetime.now().strftime('%H:%M:%S')}] ▶ MLP 학습 중...")
X_train, X_test, y_train, y_test, sw_train, sw_test, scale_pos, df = load_data()

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

model = get_mlp()
model.fit(X_train_sc, y_train, sample_weight=sw_train)

X_all    = pd.concat([X_train, X_test])
y_all    = pd.concat([y_train, y_test])
X_all_sc = scaler.transform(X_all)

metrics, _, _ = evaluate(model, X_test_sc, y_test, "MLP", X_all_sc, y_all)
save_model(model, "MLP", metrics, scaler=scaler)
print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ MLP 완료!")
