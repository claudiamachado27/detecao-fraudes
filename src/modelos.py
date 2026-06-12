from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def treinar_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def treinar_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    return rf

def treinar_xgboost(X_train, y_train):
    xgb_model = xgb.XGBClassifier(scale_pos_weight=10, eval_metric="logloss", random_state=42)
    xgb_model.fit(X_train, y_train)
    return xgb_model