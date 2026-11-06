#!/usr/bin/env python3
"""
train.py - Tái lập pipeline huấn luyện của notebook Credit_Risk_XGBoost_Colab.ipynb
dưới dạng script (không vẽ biểu đồ), xuất ra:
    models/credit_risk_artifacts.pkl
    metrics_summary.json

Cùng SEED, cùng cách chia tập, cùng tham số với notebook. Số liệu có thể lệch
nhẹ ở chữ số thập phân cuối nếu phiên bản xgboost/lightgbm/scikit-learn khác
với bản trên Colab (XGBoost 3.4.1, LightGBM 4.6.0).

Chạy:
    python train.py                         # tự tải dữ liệu
    python train.py --data path/to/file.csv # dùng file CSV có sẵn
"""

import argparse
import glob
import json
import os
import shutil
import urllib.request

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import CAT_COLS, DROP_RAW_COLS, apply_imputation_and_feature_engineering

SEED = 42
COST_FN = 5000  # Bỏ lọt 1 khách vỡ nợ
COST_FP = 1000  # Từ chối nhầm 1 khách tốt
WEIGHTS = (0.45, 0.35, 0.20)  # XGB, LGB, RF

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_CSV = os.path.join(DATA_DIR, "credit_risk_dataset.csv")

BACKUP_URLS = [
    "https://huggingface.co/datasets/pra240patil/Credit-Risk-Data/resolve/main/credit_risk_dataset.csv",
    "https://raw.githubusercontent.com/LeHongNgoc3820/Project_Credit_Risk/main/credit_risk_dataset.csv",
    "https://raw.githubusercontent.com/LeHongNgoc3820/Project_Credit_Risk/master/credit_risk_dataset.csv",
    "https://raw.githubusercontent.com/issacchan26/CreditRiskPrediction/main/credit_risk_dataset.csv",
    "https://raw.githubusercontent.com/issacchan26/CreditRiskPrediction/master/credit_risk_dataset.csv",
]


def download_dataset(csv_path: str) -> None:
    """Tải 'laotse/credit-risk-dataset' qua kagglehub, nếu lỗi thì thử các link dự phòng."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    try:
        import kagglehub

        print("Đang tải 'laotse/credit-risk-dataset' qua KaggleHub...")
        dl_dir = kagglehub.dataset_download("laotse/credit-risk-dataset")
        found = glob.glob(os.path.join(dl_dir, "**", "*.csv"), recursive=True)
        if found:
            shutil.copy2(found[0], csv_path)
            return
    except Exception as e:  # noqa: BLE001
        print(f"KaggleHub không khả dụng ({e}), thử link dự phòng...")

    for url in BACKUP_URLS:
        try:
            print(f"Thử tải: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp, open(csv_path, "wb") as f:
                shutil.copyfileobj(resp, f)
            if os.path.getsize(csv_path) > 100_000:
                return
        except Exception as e:  # noqa: BLE001
            print(f"  Bỏ qua ({e})")
        if os.path.exists(csv_path):
            os.remove(csv_path)
    raise RuntimeError(
        "Không tải được dữ liệu. Hãy tải thủ công từ "
        "https://www.kaggle.com/datasets/laotse/credit-risk-dataset và chạy "
        "`python train.py --data <đường_dẫn_csv>`."
    )


def evaluate_probs(name, val_probs, y_v, test_probs, y_te, threshold=0.50):
    test_preds = (test_probs >= threshold).astype(int)
    return {
        "Model": name,
        "Val_ROC_AUC": round(float(roc_auc_score(y_v, val_probs)), 4),
        "Test_ROC_AUC": round(float(roc_auc_score(y_te, test_probs)), 4),
        "Test_PR_AUC": round(float(average_precision_score(y_te, test_probs)), 4),
        "Test_F1": round(float(f1_score(y_te, test_preds)), 4),
        "Test_Precision": round(float(precision_score(y_te, test_preds)), 4),
        "Test_Recall": round(float(recall_score(y_te, test_preds)), 4),
    }


def main(csv_path: str) -> None:
    np.random.seed(SEED)
    if not os.path.exists(csv_path):
        download_dataset(csv_path)

    # ---------- Bước 1: Làm sạch, chia tập, điền khuyết, feature engineering ----------
    df_raw = pd.read_csv(csv_path)
    df_raw.rename(columns={"cb_preson_cred_hist_length": "cb_person_cred_hist_length"}, inplace=True)
    print(f"Dữ liệu gốc: {len(df_raw):,} dòng")

    df_clean = df_raw.drop_duplicates().copy()
    rows_after_dedup = len(df_clean)
    df_clean = df_clean[
        (df_clean["person_age"] <= 100)
        & ((df_clean["person_emp_length"].isna()) | (df_clean["person_emp_length"] <= 60))
        & ((df_clean["person_emp_length"].isna()) | (df_clean["person_emp_length"] <= df_clean["person_age"] - 14))
    ].reset_index(drop=True)
    print(f"Loại {len(df_raw) - rows_after_dedup} dòng trùng + {rows_after_dedup - len(df_clean)} outlier "
          f"-> còn {len(df_clean):,} dòng")

    X = df_clean.drop(columns=["loan_status"])
    y = df_clean["loan_status"].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)
    print(f"Train = {len(X_train):,} | Val = {len(X_val):,} | Test = {len(X_test):,}")

    grade_int_rate_median = X_train.groupby("loan_grade")["loan_int_rate"].median().to_dict()
    global_int_rate_median = float(X_train["loan_int_rate"].median())
    emp_length_median = float(X_train["person_emp_length"].median())
    stats = (grade_int_rate_median, global_int_rate_median, emp_length_median)

    X_train_fe = apply_imputation_and_feature_engineering(X_train, *stats)
    X_val_fe = apply_imputation_and_feature_engineering(X_val, *stats)
    X_test_fe = apply_imputation_and_feature_engineering(X_test, *stats)

    cat_cols = list(CAT_COLS)
    num_cols = [c for c in X_train_fe.columns if c not in DROP_RAW_COLS]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    X_train_mat = preprocessor.fit_transform(X_train_fe)
    feature_names = num_cols + list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))
    X_train_df = pd.DataFrame(X_train_mat, columns=feature_names)
    X_val_df = pd.DataFrame(preprocessor.transform(X_val_fe), columns=feature_names)
    X_test_df = pd.DataFrame(preprocessor.transform(X_test_fe), columns=feature_names)
    print(f"Số đặc trưng: {len(feature_names)}")

    # ---------- Bước 2: Baseline vs XGBoost ----------
    results = []
    lr_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    lr_model.fit(X_train_df, y_train)
    results.append(evaluate_probs("1. Logistic Regression (Baseline)",
                                  lr_model.predict_proba(X_val_df)[:, 1], y_val,
                                  lr_model.predict_proba(X_test_df)[:, 1], y_test))

    rf_model = RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=4,
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    rf_model.fit(X_train_df, y_train)
    results.append(evaluate_probs("2. Random Forest",
                                  rf_model.predict_proba(X_val_df)[:, 1], y_val,
                                  rf_model.predict_proba(X_test_df)[:, 1], y_test))

    xgb_default = xgb.XGBClassifier(n_estimators=300, random_state=SEED, eval_metric="auc", n_jobs=-1)
    xgb_default.fit(X_train_df, y_train, eval_set=[(X_val_df, y_val)], verbose=False)
    results.append(evaluate_probs("3. XGBoost (Default)",
                                  xgb_default.predict_proba(X_val_df)[:, 1], y_val,
                                  xgb_default.predict_proba(X_test_df)[:, 1], y_test))

    xgb_tuned = xgb.XGBClassifier(
        n_estimators=800, learning_rate=0.035, max_depth=6, min_child_weight=5,
        subsample=0.85, colsample_bytree=0.85, reg_alpha=0.5, reg_lambda=2.0,
        scale_pos_weight=1.5, random_state=SEED, eval_metric="auc",
        early_stopping_rounds=40, n_jobs=-1,
    )
    xgb_tuned.fit(X_train_df, y_train, eval_set=[(X_val_df, y_val)], verbose=False)
    best_n_trees = int(getattr(xgb_tuned, "best_iteration", 400) or 400)
    results.append(evaluate_probs("4. XGBoost (Tuned + Regularized)",
                                  xgb_tuned.predict_proba(X_val_df)[:, 1], y_val,
                                  xgb_tuned.predict_proba(X_test_df)[:, 1], y_test))
    print(f"XGBoost Tuned best_iteration: {best_n_trees}")

    # ---------- Bước 3.1: LightGBM + Soft-Voting + Stacking ----------
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.035, num_leaves=31, max_depth=6,
        colsample_bytree=0.85, subsample=0.85, reg_alpha=0.5, reg_lambda=2.0,
        class_weight={0: 1.0, 1: 1.5}, random_state=SEED, n_jobs=-1, verbose=-1,
    )
    lgb_model.fit(X_train_df, y_train)

    val_p = {m: model.predict_proba(X_val_df)[:, 1] for m, model in
             [("xgb", xgb_tuned), ("lgb", lgb_model), ("rf", rf_model)]}
    test_p = {m: model.predict_proba(X_test_df)[:, 1] for m, model in
              [("xgb", xgb_tuned), ("lgb", lgb_model), ("rf", rf_model)]}
    results.append(evaluate_probs("5. LightGBM (Tuned)", val_p["lgb"], y_val, test_p["lgb"], y_test))

    val_probs_voting = WEIGHTS[0] * val_p["xgb"] + WEIGHTS[1] * val_p["lgb"] + WEIGHTS[2] * val_p["rf"]
    test_probs_voting = WEIGHTS[0] * test_p["xgb"] + WEIGHTS[1] * test_p["lgb"] + WEIGHTS[2] * test_p["rf"]
    results.append(evaluate_probs("6. Hybrid Soft-Voting (XGB + LGB + RF)",
                                  val_probs_voting, y_val, test_probs_voting, y_test))

    # Lưu ý: meta-learner học trên tập Val (blending), nên Val_ROC_AUC của Stacking bị lạc quan.
    val_meta_X = np.column_stack([val_p["xgb"], val_p["lgb"], val_p["rf"]])
    test_meta_X = np.column_stack([test_p["xgb"], test_p["lgb"], test_p["rf"]])
    meta_learner = LogisticRegression(random_state=SEED)
    meta_learner.fit(val_meta_X, y_val)
    results.append(evaluate_probs("7. Hybrid Stacking (XGB + LGB + RF -> Meta LR)",
                                  meta_learner.predict_proba(val_meta_X)[:, 1], y_val,
                                  meta_learner.predict_proba(test_meta_X)[:, 1], y_test))

    results_df = pd.DataFrame(results).sort_values(by="Test_ROC_AUC", ascending=False).reset_index(drop=True)
    print(results_df.to_string(index=False))

    # ---------- Bước 3.2: Tối ưu ngưỡng theo chi phí (chọn trên Val, kiểm tra trên Test) ----------
    thresholds = np.linspace(0.10, 0.85, 76)
    costs, f1s = [], []
    for th in thresholds:
        preds = (val_probs_voting >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        costs.append(fn * COST_FN + fp * COST_FP)
        f1s.append(f1_score(y_val, preds))
    best_cost_threshold = float(round(thresholds[int(np.argmin(costs))], 2))
    best_f1_threshold = float(round(thresholds[int(np.argmax(f1s))], 2))

    def test_cost(th):
        tn, fp, fn, tp = confusion_matrix(y_test, (test_probs_voting >= th).astype(int)).ravel()
        return {"threshold": th, "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
                "recall": round(tp / (tp + fn), 4), "total_cost_usd": int(fn * COST_FN + fp * COST_FP)}

    cost_default = test_cost(0.50)
    cost_opt = test_cost(best_cost_threshold)
    saved_money = cost_default["total_cost_usd"] - cost_opt["total_cost_usd"]
    print(f"Ngưỡng tối ưu chi phí: {best_cost_threshold} | Ngưỡng tối ưu F1: {best_f1_threshold} | "
          f"Tiết kiệm trên Test: ${saved_money:,}")

    # ---------- Bước 4: Đóng gói ----------
    os.makedirs(MODELS_DIR, exist_ok=True)
    artifacts = {
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "grade_int_rate_median": grade_int_rate_median,
        "global_int_rate_median": global_int_rate_median,
        "emp_length_median": emp_length_median,
        "xgb_model": xgb_tuned,
        "lgb_model": lgb_model,
        "rf_model": rf_model,
        "voting_weights": WEIGHTS,
        "meta_learner": meta_learner,
        "best_cost_threshold": best_cost_threshold,
        "best_f1_threshold": best_f1_threshold,
        "results_table": results_df.to_dict(orient="records"),
    }
    joblib.dump(artifacts, os.path.join(MODELS_DIR, "credit_risk_artifacts.pkl"))

    summary = {
        "dataset_rows_raw": int(len(df_raw)),
        "duplicates_removed": int(len(df_raw) - rows_after_dedup),
        "outliers_removed": int(rows_after_dedup - len(df_clean)),
        "dataset_rows_clean": int(len(df_clean)),
        "default_rate_pct": round(float(y.mean() * 100), 2),
        "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "num_features_engineered": int(len(feature_names)),
        "xgb_best_iteration": best_n_trees,
        "cost_assumptions_usd": {"false_negative": COST_FN, "false_positive": COST_FP},
        "best_f1_threshold": best_f1_threshold,
        "best_cost_threshold": best_cost_threshold,
        "test_cost_at_default_threshold": cost_default,
        "test_cost_at_best_threshold": cost_opt,
        "cost_saved_on_test_usd": int(saved_money),
        "models_benchmark": results_df.to_dict(orient="records"),
    }
    with open(os.path.join(BASE_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Đã lưu models/credit_risk_artifacts.pkl và metrics_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_CSV, help="Đường dẫn file credit_risk_dataset.csv")
    main(parser.parse_args().data)
